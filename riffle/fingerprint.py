"""Chromaprint fingerprints via fpcalc.

fpcalc's default caps at the first 120 seconds, which would hide every trim
and edit past two minutes, so `-length 0` (unlimited) is always passed.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

from riffle import store

ANALYZER = "chromaprint"
DEFAULT_CONFIG = {"length": 0, "algorithm": 2, "raw": True}


class FingerprintError(Exception):
    """fpcalc could not fingerprint the file."""


@dataclass(frozen=True)
class FingerprintResult:
    raw: np.ndarray
    duration: float
    algorithm: int


@lru_cache(maxsize=1)
def fpcalc_version() -> str:
    out = subprocess.run(
        ["fpcalc", "-version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return out


def config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# A short frequency-swept, amplitude-modulated reference tone, matching the
# design in tests/fixtures.make_tone (duplicated rather than imported: this
# is production code and must not depend on the test suite). Used only for
# enrich.py's prefix-compatibility check, a different concern from item
# duration below.
_REFERENCE_TONE_EXPR = (
    "sin(2*PI*t*(440+220*sin(2*PI*t/7)))*(0.4+0.3*sin(2*PI*t*1.7))"
)
_REFERENCE_TONE_SECONDS = 120


def internal_algorithm(cli_algorithm: int) -> int:
    """The chromaprint library's internal algorithm enum for the number
    fpcalc's `-algorithm` CLI flag was invoked with.

    These are off by one, verified directly against a real fpcalc round
    trip: `fpcalc -algorithm 2`'s own native encoded output, when decoded,
    reports algorithm=1 -- not 2. `chromaprint_new(1)` (not `chromaprint_new
    (2)`) is what the CLI's "2" actually means to the C API's encode,
    decode, and item-duration calls, and passing the CLI number directly to
    any of those (as this project's code originally did) produces a
    mismatched fingerprint header and, for item duration specifically, a
    value roughly 3x too large (see item_duration_seconds below).
    """
    return cli_algorithm - 1


@lru_cache(maxsize=1)
def item_duration_seconds() -> float:
    """Seconds of audio per fingerprint item, from the library itself.

    Queries `chromaprint_new()` with the *internal* algorithm enum
    (`internal_algorithm(DEFAULT_CONFIG["algorithm"])`), not the fpcalc CLI
    number directly. Passing the CLI number (2) gives item_duration=4096,
    sample_rate=11025 -> 0.3715s/item, roughly 3x the true value; the
    correct enum (1) gives item_duration=1365 -> 0.12381s/item, matching
    the spec's own "about 0.124" and Task 10's hardcoded ITEM=0.1238 test
    constant exactly. An earlier version of this function assumed the
    discrepancy was unexplainable from the documented API and worked around
    it by fingerprinting a real reference tone empirically; that masked the
    true, simpler cause (see the Task 20 review's finding I4) and only
    approximated the correct value to within a couple of percent.
    """
    import chromaprint  # provided by pyacoustid

    algorithm = internal_algorithm(DEFAULT_CONFIG["algorithm"])
    ctx = chromaprint._libchromaprint.chromaprint_new(algorithm)
    try:
        item = chromaprint._libchromaprint.chromaprint_get_item_duration(ctx)
        rate = chromaprint._libchromaprint.chromaprint_get_sample_rate(ctx)
        return item / rate
    finally:
        chromaprint._libchromaprint.chromaprint_free(ctx)


def lookup_window_items(seconds: float) -> int:
    """How many fingerprint items a real `fpcalc -length <seconds>` run
    produces, matching Chromaprint's own delay-aware accounting.

    A naive `seconds / item_duration_seconds()` overcounts: Chromaprint
    reserves a fixed per-file delay (`chromaprint_get_delay()`, ~2.6s at
    the internal 11025 Hz working rate) before its first item, so the
    items actually available within the first `seconds` of decoded audio
    are `(seconds * sample_rate - delay) / item_duration`, not the naive
    division. Verified directly: for a 120s window this gives 948 items,
    matching a real `fpcalc -length 120` run exactly; the naive formula
    gives 969, which does not -- see the Task 20 review's finding I4.
    """
    import chromaprint

    algorithm = internal_algorithm(DEFAULT_CONFIG["algorithm"])
    ctx = chromaprint._libchromaprint.chromaprint_new(algorithm)
    try:
        item = chromaprint._libchromaprint.chromaprint_get_item_duration(ctx)
        rate = chromaprint._libchromaprint.chromaprint_get_sample_rate(ctx)
        delay = chromaprint._libchromaprint.chromaprint_get_delay(ctx)
    finally:
        chromaprint._libchromaprint.chromaprint_free(ctx)
    return max(0, int((seconds * rate - delay) / item))


def _parse(stdout: str) -> tuple[np.ndarray, float]:
    data = json.loads(stdout)
    fp = data["fingerprint"]
    if isinstance(fp, str):
        fp = [int(x) for x in fp.split(",") if x]
    return np.asarray(fp, dtype=np.uint32), float(data["duration"])


def fingerprint_file(path: Path, config: dict = DEFAULT_CONFIG) -> FingerprintResult:
    args = [
        "fpcalc", "-json", "-raw",
        "-length", str(config["length"]),
        "-algorithm", str(config["algorithm"]),
        str(path),
    ]
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FingerprintError(f"fpcalc failed on {path}: {proc.stderr.strip()}")
    try:
        raw, duration = _parse(proc.stdout)
    except (ValueError, KeyError) as exc:
        raise FingerprintError(
            f"unexpected fpcalc output for {path}: {proc.stdout[:200]!r}"
        ) from exc
    if duration <= 0:
        raise FingerprintError(f"{path} reports a duration of {duration}")
    if len(raw) == 0:
        raise FingerprintError(f"{path} produced an empty fingerprint")
    return FingerprintResult(raw, duration, config["algorithm"])


def fingerprint_pending(conn, config: dict = DEFAULT_CONFIG) -> int:
    version = fpcalc_version()
    chash = config_hash(config)
    rows = conn.execute(
        "SELECT ac.id AS content_id, MIN(t.path) AS path "
        "FROM audio_content ac "
        "JOIN track t ON t.audio_content_id = ac.id AND t.present = 1 "
        "WHERE NOT EXISTS ("
        "  SELECT 1 FROM fingerprint f "
        "  WHERE f.audio_content_id = ac.id AND f.purpose = 'canonical' "
        "    AND f.analyzer = ? AND f.analyzer_version = ? AND f.config_hash = ?"
        ") GROUP BY ac.id",
        (ANALYZER, version, chash),
    ).fetchall()

    written = 0
    for row in rows:
        try:
            res = fingerprint_file(Path(row["path"]), config)
        except FingerprintError as exc:
            conn.execute(
                "INSERT INTO ingest_error (path, stage, message, attempts, "
                " last_attempt_at) VALUES (?,?,?,1,?) "
                "ON CONFLICT(path) DO UPDATE SET "
                "attempts = ingest_error.attempts + 1, "
                "message = excluded.message, "
                "last_attempt_at = excluded.last_attempt_at",
                (row["path"], "fingerprint", str(exc),
                 datetime.now(timezone.utc).isoformat()),
            )
            continue
        conn.execute(
            "INSERT INTO fingerprint (audio_content_id, analyzer, "
            " analyzer_version, config_hash, purpose, algorithm, fp_raw, "
            " fp_length, computed_at) "
            "VALUES (?,?,?,?,'canonical',?,?,?,?)",
            (row["content_id"], ANALYZER, version, chash, res.algorithm,
             store.pack_fingerprint(res.raw), len(res.raw),
             datetime.now(timezone.utc).isoformat()),
        )
        conn.execute(
            "UPDATE audio_content SET duration = ? WHERE id = ?",
            (res.duration, row["content_id"]),
        )
        written += 1
    return written
