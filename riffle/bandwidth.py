"""Effective bandwidth: the highest frequency a lossy file actually keeps.

Nominal bitrate is a poor guide to quality when files have been re-encoded: a
"320 kbps" mp3 made from a 128 kbps source still cuts off near 16 kHz, while
a genuine ~250 kbps file reaches 20 kHz. The low-pass shows up as a steep
*cliff* in the average spectrum, so the estimator looks for that cliff
rather than for a level threshold, which saturates at the top of the band.
"""
from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SAMPLE_RATE = 44100
FRAME = 4096
CHUNK_SECONDS = 15
WHOLE_FILE_LIMIT_SECONDS = 40
MIN_FRAMES = 4

BAND_HZ = 500
REFERENCE_BAND = (1000, 4000)  # mid-band level the cliff is measured against
# Low-bitrate or 22 kHz-sample-rate files cut off near 9-11 kHz, so the
# search must start just above the reference band. Starting higher read
# those files as "no cliff", i.e. as the widest copy in their group.
CLIFF_SEARCH = (5000, 21000)
# LAME's soft low-pass falls only ~16-19 dB in its steepest kHz; natural
# spectral tilt stays well under 12 (see tests), so 12 separates them.
MIN_CLIFF_DB = 12.0


@dataclass(frozen=True)
class Bandwidth:
    """`cutoff_hz` is None when the file has no low-pass cliff (full band)."""
    cutoff_hz: float | None
    cliff_db: float


def _decode(path: Path, start: float, seconds: float) -> np.ndarray:
    try:
        proc = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-ss", f"{max(start, 0):.1f}",
             "-t", str(seconds), "-i", str(path), "-ac", "1",
             "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"],
            capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return np.zeros(0)
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(float)


def _frames(samples: np.ndarray) -> np.ndarray | None:
    n = (len(samples) // FRAME) * FRAME
    if n < FRAME * MIN_FRAMES:
        return None
    return samples[:n].reshape(-1, FRAME)


def _average_power_db(path: Path, duration: float | None):
    if duration is None or duration <= WHOLE_FILE_LIMIT_SECONDS:
        chunks = [_frames(_decode(path, 0, WHOLE_FILE_LIMIT_SECONDS))]
    else:
        # Three windows spread through the track, so one quiet passage does
        # not decide the answer.
        chunks = [_frames(_decode(path, duration * frac - CHUNK_SECONDS / 2,
                                  CHUNK_SECONDS))
                  for frac in (0.25, 0.5, 0.75)]
    chunks = [c for c in chunks if c is not None]
    if not chunks:
        return None
    frames = np.vstack(chunks) * np.hanning(FRAME)
    power = (np.abs(np.fft.rfft(frames, axis=1)) ** 2).mean(axis=0) + 1e-3
    return power, np.fft.rfftfreq(FRAME, 1 / SAMPLE_RATE)


def measure_cutoff(path: Path, duration: float | None) -> Bandwidth | None:
    """The file's low-pass cliff, or None when it cannot be measured."""
    spectrum = _average_power_db(Path(path), duration)
    if spectrum is None:
        return None
    power, freqs = spectrum

    edges = np.arange(REFERENCE_BAND[0], SAMPLE_RATE // 2 - BAND_HZ, BAND_HZ)
    bands = np.array([
        10 * np.log10(power[(freqs >= a) & (freqs < a + BAND_HZ)].mean())
        for a in edges])
    ref = bands[(edges >= REFERENCE_BAND[0])
                & (edges < REFERENCE_BAND[1])].mean()
    level = bands - ref

    # Steepest fall across two bands (1 kHz) inside the search range.
    best_drop, best_i = 0.0, None
    for i in range(len(edges) - 2):
        if CLIFF_SEARCH[0] <= edges[i] <= CLIFF_SEARCH[1]:
            drop = level[i] - level[i + 2]
            if drop > best_drop:
                best_drop, best_i = drop, i
    if best_i is None or best_drop < MIN_CLIFF_DB:
        return Bandwidth(None, float(best_drop))

    # The cliff sits where the level first falls halfway down the drop.
    midpoint = level[best_i] - best_drop / 2
    for j in range(best_i, len(edges)):
        if level[j] <= midpoint:
            return Bandwidth(float(edges[j]), float(best_drop))
    return Bandwidth(float(edges[best_i + 2]), float(best_drop))


def bandwidth_scan(conn, limit: int | None = None,
                   remeasure: bool = False, workers: int = 4) -> dict:
    """Measure present content that has no measurement yet.

    `remeasure` measures every present file again and replaces its stored
    value, for when the estimator itself has changed. Measurements of files
    that are no longer present are left alone.

    Unmeasurable audio is recorded with NULL cutoff *and* NULL cliff so it is
    not retried on every run; a full-band file has a cliff_db and a NULL
    cutoff.
    """
    query = (
        "SELECT ac.id AS content_id, ac.duration AS duration, "
        "       t.path AS path, MIN(t.id) AS track_id "
        "FROM audio_content ac "
        "JOIN track t ON t.audio_content_id = ac.id AND t.present = 1 "
        "LEFT JOIN audio_bandwidth ab ON ab.audio_content_id = ac.id "
        + ("" if remeasure else "WHERE ab.audio_content_id IS NULL ")
        + "GROUP BY ac.id ORDER BY ac.id"
    )
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    rows = conn.execute(query).fetchall()
    cached = 0 if remeasure else conn.execute(
        "SELECT count(DISTINCT ab.audio_content_id) AS n "
        "FROM audio_bandwidth ab JOIN track t "
        "  ON t.audio_content_id = ab.audio_content_id AND t.present = 1"
    ).fetchone()["n"]

    # Decoding is ffmpeg subprocess time, so threads overlap it well. Rows
    # are read and results written on this thread only.
    jobs = [(Path(r["path"]), r["duration"]) for r in rows]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(lambda job: measure_cutoff(*job), jobs))

    measured = unmeasurable = 0
    for row, result in zip(rows, results):
        if result is None:
            unmeasurable += 1
            values = (None, None)
        else:
            measured += 1
            values = (result.cutoff_hz, result.cliff_db)
        conn.execute(
            "INSERT OR REPLACE INTO audio_bandwidth "
            "(audio_content_id, cutoff_hz, cliff_db, measured_at) "
            "VALUES (?,?,?,?)",
            (row["content_id"], *values,
             datetime.now(timezone.utc).isoformat()))
    return {"measured": measured, "unmeasurable": unmeasurable,
            "cached": cached}
