"""AcoustID enrichment: best-effort, rate-limited, cached, resumable.

Local matching is authoritative. Enrichment may fail, be throttled, or return
nothing, and dedup results are unchanged either way.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from functools import lru_cache

import numpy as np

from riffle import fingerprint, store

ACOUSTID_RATE = 3.0  # requests per second, per the service's guidelines
META = "recordings+releasegroups+compress"
LOOKUP_SECONDS = 120.0  # fpcalc's default, which populated the index
_COMPAT_REFERENCE_SECONDS = 150  # comfortably past the 120s lookup window


class RateLimiter:
    def __init__(
        self, rate_per_second: float, sleeper=time.sleep, clock=time.monotonic
    ):
        self._interval = 1.0 / rate_per_second
        self._sleep = sleeper
        self._clock = clock
        self._last = None

    def wait(self) -> None:
        now = self._clock()
        if self._last is not None:
            remaining = self._interval - (now - self._last)
            if remaining > 0:
                self._sleep(remaining)
                # Refresh: sleeping advanced the clock, so the pre-sleep
                # `now` is stale. Any real per-request latency between
                # calls otherwise compounds against this staleness and lets
                # the effective rate drift above the configured cap -- see
                # test_rate_limiter_accounts_for_time_spent_sleeping.
                now = self._clock()
        self._last = now


def encode(raw: np.ndarray, algorithm: int) -> str:
    """Re-encode a raw fingerprint. `algorithm` is fpcalc's CLI number (as
    stored in `fingerprint.algorithm`), translated to chromaprint's
    internal enum before calling the C API -- see
    `fingerprint.internal_algorithm` for why the two are not the same
    number, and the Task 20 review's finding I5.
    """
    import chromaprint

    return chromaprint.encode_fingerprint(
        [int(x) for x in raw], fingerprint.internal_algorithm(algorithm), base64=True
    ).decode("ascii")


def lookup_key(algorithm: int, encoded_fp: str, duration: float, meta: str) -> str:
    blob = json.dumps(
        {"algorithm": algorithm, "fp": encoded_fp, "duration": duration, "meta": meta},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode()).hexdigest()


@lru_cache(maxsize=1)
def _prefix_slicing_is_valid() -> bool:
    """Whether slicing the canonical array and re-encoding matches a real
    `fpcalc -length 120` run, checked once per process.

    Chromaprint fingerprints are not *guaranteed* prefix-compatible by any
    documented property of the public API, so this is verified rather than
    assumed. It empirically holds on this build (algorithm 2, fpcalc 1.5.1):
    an earlier apparent failure here was actually a bug in
    `item_duration_seconds()` computing the wrong slice length, not genuine
    prefix instability -- see the Task 18 ledger entry.
    """
    import subprocess
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        ref = Path(tmp) / "ref.flac"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"aevalsrc={fingerprint._REFERENCE_TONE_EXPR}:s=44100:"
                f"d={_COMPAT_REFERENCE_SECONDS}",
                "-ac",
                "1",
                "-c:a",
                "flac",
                str(ref),
            ],
            check=True,
        )
        full = fingerprint.fingerprint_file(ref).raw
        n = min(len(full), fingerprint.lookup_window_items(LOOKUP_SECONDS))
        native = fingerprint.fingerprint_file(
            ref, dict(fingerprint.DEFAULT_CONFIG, length=LOOKUP_SECONDS)
        ).raw

    return encode(full[:n], 2) == encode(native, 2)


def _create_dedicated_artifact(conn, content_id: int) -> tuple[str, int]:
    """Fallback: a real fpcalc run at length=120, stored and reused.

    Only reached when `_prefix_slicing_is_valid()` is false. The canonical
    full-track artifact is never touched.
    """
    from pathlib import Path

    track = conn.execute(
        "SELECT path FROM track WHERE audio_content_id = ? AND present = 1 LIMIT 1",
        (content_id,),
    ).fetchone()
    if track is None:
        raise LookupError(f"no present track for content {content_id}")

    version = fingerprint.fpcalc_version()
    lookup_config = dict(fingerprint.DEFAULT_CONFIG, length=LOOKUP_SECONDS)
    chash = fingerprint.config_hash(lookup_config)
    result = fingerprint.fingerprint_file(Path(track["path"]), lookup_config)

    conn.execute(
        "INSERT INTO fingerprint (audio_content_id, analyzer, "
        " analyzer_version, config_hash, purpose, algorithm, fp_raw, "
        " fp_length, computed_at) "
        "VALUES (?, ?, ?, ?, 'acoustid_lookup', ?, ?, ?, ?)",
        (
            content_id,
            fingerprint.ANALYZER,
            version,
            chash,
            result.algorithm,
            store.pack_fingerprint(result.raw),
            len(result.raw),
            datetime.now(UTC).isoformat(),
        ),
    )
    return encode(result.raw, result.algorithm), len(result.raw)


def lookup_fingerprint(conn, content_id: int, config: dict) -> tuple[str, int]:
    """The base64 fingerprint to send to AcoustID, and its item count.

    Preferred: slice the canonical array already in the database and
    re-encode -- no file access, no new row. Used whenever the canonical is
    already within the lookup window, or slicing is verified compatible.
    Fallback (only if `_prefix_slicing_is_valid()` is false): a dedicated
    `purpose = 'acoustid_lookup'` fingerprint from a real fpcalc run,
    created once per content and reused after that.
    """
    lookup_row = conn.execute(
        "SELECT fp_raw, fp_length, algorithm FROM fingerprint "
        "WHERE audio_content_id = ? AND purpose = 'acoustid_lookup' "
        "ORDER BY id DESC LIMIT 1",
        (content_id,),
    ).fetchone()
    if lookup_row is not None:
        raw = store.unpack_fingerprint(lookup_row["fp_raw"], lookup_row["fp_length"])
        return encode(raw, lookup_row["algorithm"]), len(raw)

    canonical = conn.execute(
        "SELECT fp_raw, fp_length, algorithm FROM fingerprint "
        "WHERE audio_content_id = ? AND purpose = 'canonical' "
        "ORDER BY id DESC LIMIT 1",
        (content_id,),
    ).fetchone()
    if canonical is None:
        raise LookupError(f"no canonical fingerprint for content {content_id}")

    canonical_raw = store.unpack_fingerprint(
        canonical["fp_raw"], canonical["fp_length"]
    )
    n = min(len(canonical_raw), fingerprint.lookup_window_items(LOOKUP_SECONDS))

    # A fingerprint short enough to already be within the lookup window needs
    # no slicing decision at all: the canonical array itself, taken whole, is
    # exactly what a default-length fpcalc run over the same short audio
    # would give, regardless of prefix-compatibility.
    if len(canonical_raw) <= n:
        return encode(canonical_raw, canonical["algorithm"]), len(canonical_raw)

    if _prefix_slicing_is_valid():
        sliced = canonical_raw[:n]
        return encode(sliced, canonical["algorithm"]), len(sliced)

    return _create_dedicated_artifact(conn, content_id)


def _default_client(apikey, fp, duration, meta):
    import acoustid

    return acoustid.lookup(apikey, fp, duration, meta=meta)


def enrich(conn, api_key: str, client=None, sleeper=None) -> dict:
    client = client or _default_client
    limiter = RateLimiter(ACOUSTID_RATE, sleeper=sleeper or time.sleep)

    looked_up = cached = failed = 0
    rows = conn.execute(
        "SELECT ac.id AS cid, ac.duration, f.algorithm "
        "FROM audio_content ac "
        "JOIN fingerprint f ON f.audio_content_id = ac.id "
        "                  AND f.purpose = 'canonical' "
        "WHERE ac.duration IS NOT NULL GROUP BY ac.id ORDER BY ac.id"
    ).fetchall()

    for row in rows:
        try:
            encoded, _ = lookup_fingerprint(
                conn, row["cid"], fingerprint.DEFAULT_CONFIG
            )
        except (LookupError, fingerprint.FingerprintError):
            # lookup_fingerprint may need to run fpcalc for the dedicated
            # acoustid_lookup artifact; that call can fail the same way any
            # fingerprinting can. Enrichment is best-effort and must never
            # block on it -- count it as failed and move on.
            failed += 1
            continue

        key = lookup_key(row["algorithm"], encoded, row["duration"], META)
        if conn.execute(
            "SELECT 1 FROM acoustid_cache WHERE lookup_key = ?", (key,)
        ).fetchone():
            cached += 1
            continue

        limiter.wait()
        try:
            # duration is the WHOLE file's duration, per the service's docs,
            # even though the fingerprint sent is truncated to 120 seconds.
            response = client(api_key, encoded, row["duration"], META)
        except Exception:
            failed += 1
            continue

        if not isinstance(response, dict) or response.get("status") != "ok":
            # pyacoustid's client does not raise for an API-level error (bad
            # key, rate limit, malformed request) -- it returns a normal
            # response dict with status="error", the same shape as success.
            # Caching that would poison the cache: a later run, even with a
            # corrected key, would see a cache hit and never retry. Observed
            # live against the real service with an invalid key.
            failed += 1
            continue

        conn.execute(
            "INSERT INTO acoustid_cache (lookup_key, response_json, fetched_at) "
            "VALUES (?,?,?)",
            (key, json.dumps(response), datetime.now(UTC).isoformat()),
        )
        looked_up += 1

    return {"looked_up": looked_up, "cached": cached, "failed": failed}
