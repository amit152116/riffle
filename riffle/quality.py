"""Audio quality analysis — clipping, silence, low volume detection."""
from __future__ import annotations

import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

CLIPPING_THRESHOLD_DB = 0.0
LOW_VOLUME_THRESHOLD_DB = -30.0
SILENCE_NOISE_DB = -50
SILENCE_DURATION_S = 5


def analyze_file(path: Path) -> dict:
    """Run ffmpeg volumedetect and silencedetect on a single file."""
    cmd = [
        "ffmpeg", "-hide_banner", "-i", str(path),
        "-af", f"volumedetect,silencedetect=noise={SILENCE_NOISE_DB}dB"
                f":d={SILENCE_DURATION_S}",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    stderr = proc.stderr

    peak_db = _parse_float(r"max_volume:\s*([-\d.]+)\s*dB", stderr)
    mean_db = _parse_float(r"mean_volume:\s*([-\d.]+)\s*dB", stderr)

    silence_ends = re.findall(r"silence_end:", stderr)
    silence_sections = len(silence_ends)

    clipping = peak_db is not None and peak_db > CLIPPING_THRESHOLD_DB
    low_volume = mean_db is not None and mean_db < LOW_VOLUME_THRESHOLD_DB

    return {
        "peak_db": peak_db,
        "mean_db": mean_db,
        "clipping": clipping,
        "low_volume": low_volume,
        "silence_sections": silence_sections,
    }


def _parse_float(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def quality_scan(conn, limit: int | None = None) -> dict:
    """Analyze present tracks not yet in quality_flag."""
    query = (
        "SELECT t.id AS track_id, t.path, t.audio_content_id "
        "FROM track t "
        "LEFT JOIN quality_flag qf ON qf.audio_content_id = t.audio_content_id "
        "WHERE t.present = 1 AND qf.id IS NULL "
        "ORDER BY t.id"
    )
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    rows = conn.execute(query).fetchall()

    analyzed = failed = issues = cached = 0

    already = conn.execute(
        "SELECT count(*) c FROM track t "
        "JOIN quality_flag qf ON qf.audio_content_id = t.audio_content_id "
        "WHERE t.present = 1"
    ).fetchone()["c"]
    cached = already

    for row in rows:
        path = Path(row["path"])
        if not path.exists():
            failed += 1
            continue

        try:
            result = analyze_file(path)
        except Exception:
            failed += 1
            continue

        has_issue = result["clipping"] or result["low_volume"] or \
            result["silence_sections"] > 0
        if has_issue:
            issues += 1

        conn.execute(
            "INSERT INTO quality_flag (audio_content_id, track_id, peak_db, "
            "mean_db, clipping, low_volume, silence_sections, analyzed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(audio_content_id) DO UPDATE SET "
            "peak_db=excluded.peak_db, mean_db=excluded.mean_db, "
            "clipping=excluded.clipping, low_volume=excluded.low_volume, "
            "silence_sections=excluded.silence_sections, "
            "analyzed_at=excluded.analyzed_at",
            (row["audio_content_id"], row["track_id"],
             result["peak_db"], result["mean_db"],
             int(result["clipping"]), int(result["low_volume"]),
             result["silence_sections"],
             datetime.now(UTC).isoformat()),
        )
        analyzed += 1

    return {"analyzed": analyzed, "failed": failed,
            "issues": issues, "cached": cached}


def render_quality(conn) -> str:
    """Text summary of quality flags in the database."""
    total = conn.execute(
        "SELECT count(*) c FROM quality_flag"
    ).fetchone()["c"]
    clipping = conn.execute(
        "SELECT count(*) c FROM quality_flag WHERE clipping = 1"
    ).fetchone()["c"]
    low_vol = conn.execute(
        "SELECT count(*) c FROM quality_flag WHERE low_volume = 1"
    ).fetchone()["c"]
    silence = conn.execute(
        "SELECT count(*) c FROM quality_flag WHERE silence_sections > 0"
    ).fetchone()["c"]
    clean = conn.execute(
        "SELECT count(*) c FROM quality_flag "
        "WHERE clipping = 0 AND low_volume = 0 AND silence_sections = 0"
    ).fetchone()["c"]

    lines = ["Audio Quality Report", "=" * 40, ""]
    lines.append(f"Analyzed:            {total}")
    lines.append(f"Clean:               {clean}")
    lines.append(f"Clipping:            {clipping}")
    lines.append(f"Low volume:          {low_vol}")
    lines.append(f"Silence detected:    {silence}")

    if clipping:
        lines.append("")
        lines.append("Clipping files:")
        for r in conn.execute(
            "SELECT t.path, qf.peak_db FROM quality_flag qf "
            "JOIN track t ON t.id = qf.track_id WHERE qf.clipping = 1"
        ).fetchall():
            lines.append(f"  {r['path']}  ({r['peak_db']:.1f} dB)")

    if low_vol:
        lines.append("")
        lines.append("Low volume files:")
        for r in conn.execute(
            "SELECT t.path, qf.mean_db FROM quality_flag qf "
            "JOIN track t ON t.id = qf.track_id WHERE qf.low_volume = 1"
        ).fetchall():
            lines.append(f"  {r['path']}  ({r['mean_db']:.1f} dB)")

    return "\n".join(lines)
