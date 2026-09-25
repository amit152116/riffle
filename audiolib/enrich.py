"""AcoustID enrichment: best-effort, rate-limited, cached, resumable.

Local matching is authoritative. Enrichment may fail, be throttled, or return
nothing, and dedup results are unchanged either way.
"""
from __future__ import annotations

import base64
import hashlib
import json
import time
from datetime import datetime, timezone

import numpy as np

from audiolib import fingerprint, store

ACOUSTID_RATE = 3.0  # requests per second, per the service's guidelines
META = "recordings+releasegroups+compress"
LOOKUP_SECONDS = 120.0  # fpcalc's default, which populated the index


class RateLimiter:
    def __init__(self, rate_per_second: float, sleeper=time.sleep,
                 clock=time.monotonic):
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
        self._last = now


def encode(raw: np.ndarray, algorithm: int) -> str:
    import chromaprint

    return chromaprint.encode_fingerprint(
        [int(x) for x in raw], algorithm, base64=True
    ).decode("ascii")


def lookup_key(algorithm: int, encoded_fp: str, duration: float,
               meta: str) -> str:
    blob = json.dumps(
        {"algorithm": algorithm, "fp": encoded_fp,
         "duration": duration, "meta": meta},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def lookup_fingerprint(conn, content_id: int, config: dict) -> tuple[str, int]:
    """The base64 fingerprint to send to AcoustID, and its item count.

    Chromaprint fingerprints are not guaranteed prefix-compatible -- slicing
    the canonical array and re-encoding was found NOT to match a plain
    default `fpcalc -length 120` run on this build (verified in
    test_prefix_compatibility_check / test_lookup_fingerprint_falls_back_to
    _a_dedicated_artifact). So a dedicated `purpose = 'acoustid_lookup'`
    fingerprint is created once per content, from a real fpcalc run, and
    reused after that. The canonical full-track artifact is never touched.
    """
    lookup_row = conn.execute(
        "SELECT fp_raw, fp_length, algorithm FROM fingerprint "
        "WHERE audio_content_id = ? AND purpose = 'acoustid_lookup' "
        "ORDER BY id DESC LIMIT 1", (content_id,)).fetchone()
    if lookup_row is not None:
        raw = store.unpack_fingerprint(lookup_row["fp_raw"],
                                       lookup_row["fp_length"])
        return encode(raw, lookup_row["algorithm"]), len(raw)

    canonical = conn.execute(
        "SELECT fp_raw, fp_length, algorithm FROM fingerprint "
        "WHERE audio_content_id = ? AND purpose = 'canonical' "
        "ORDER BY id DESC LIMIT 1", (content_id,)).fetchone()
    if canonical is None:
        raise LookupError(f"no canonical fingerprint for content {content_id}")

    canonical_raw = store.unpack_fingerprint(canonical["fp_raw"],
                                             canonical["fp_length"])
    item_seconds = fingerprint.item_duration_seconds()
    n = min(len(canonical_raw),
            int(round(LOOKUP_SECONDS / item_seconds)))

    # A fingerprint short enough to already be within the lookup window needs
    # no fpcalc re-run: the canonical array itself, taken whole, is exactly
    # what a default-length fpcalc run over the same short audio would give.
    if len(canonical_raw) <= n:
        return encode(canonical_raw, canonical["algorithm"]), len(canonical_raw)

    track = conn.execute(
        "SELECT path FROM track WHERE audio_content_id = ? AND present = 1 "
        "LIMIT 1", (content_id,)).fetchone()
    if track is None:
        raise LookupError(f"no present track for content {content_id}")

    from pathlib import Path

    version = fingerprint.fpcalc_version()
    lookup_config = dict(fingerprint.DEFAULT_CONFIG, length=LOOKUP_SECONDS)
    chash = fingerprint.config_hash(lookup_config)
    result = fingerprint.fingerprint_file(Path(track["path"]), lookup_config)

    conn.execute(
        "INSERT INTO fingerprint (audio_content_id, analyzer, "
        " analyzer_version, config_hash, purpose, algorithm, fp_raw, "
        " fp_length, computed_at) "
        "VALUES (?, ?, ?, ?, 'acoustid_lookup', ?, ?, ?, ?)",
        (content_id, fingerprint.ANALYZER, version, chash, result.algorithm,
         store.pack_fingerprint(result.raw), len(result.raw),
         datetime.now(timezone.utc).isoformat()),
    )
    return encode(result.raw, result.algorithm), len(result.raw)


def _default_client(apikey, fp, duration, meta):
    import acoustid

    return acoustid.lookup(apikey, fp, duration, meta=meta)


def enrich(conn, api_key: str, client=None, sleeper=None) -> dict:
    client = client or _default_client
    limiter = RateLimiter(ACOUSTID_RATE,
                          sleeper=sleeper or time.sleep)

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
            encoded, _ = lookup_fingerprint(conn, row["cid"],
                                            fingerprint.DEFAULT_CONFIG)
        except (LookupError, fingerprint.FingerprintError):
            # lookup_fingerprint may need to run fpcalc for the dedicated
            # acoustid_lookup artifact; that call can fail the same way any
            # fingerprinting can. Enrichment is best-effort and must never
            # block on it -- count it as failed and move on.
            failed += 1
            continue

        key = lookup_key(row["algorithm"], encoded, row["duration"], META)
        if conn.execute("SELECT 1 FROM acoustid_cache WHERE lookup_key = ?",
                        (key,)).fetchone():
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

        conn.execute(
            "INSERT INTO acoustid_cache (lookup_key, response_json, fetched_at) "
            "VALUES (?,?,?)",
            (key, json.dumps(response),
             datetime.now(timezone.utc).isoformat()))
        looked_up += 1

    return {"looked_up": looked_up, "cached": cached, "failed": failed}
