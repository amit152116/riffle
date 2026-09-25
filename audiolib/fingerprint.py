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

from audiolib import store

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


@lru_cache(maxsize=1)
def item_duration_seconds() -> float:
    """Seconds of audio per fingerprint item, from the library itself."""
    import chromaprint  # provided by pyacoustid

    ctx = chromaprint._libchromaprint.chromaprint_new(DEFAULT_CONFIG["algorithm"])
    try:
        item = chromaprint._libchromaprint.chromaprint_get_item_duration(ctx)
        rate = chromaprint._libchromaprint.chromaprint_get_sample_rate(ctx)
        return item / rate
    finally:
        chromaprint._libchromaprint.chromaprint_free(ctx)


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
