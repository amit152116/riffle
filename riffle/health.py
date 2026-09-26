"""Library health report — reads existing DB, no file processing."""
from __future__ import annotations

import os
from collections import Counter


def health_report(conn) -> dict:
    tracks = conn.execute(
        "SELECT count(*) c FROM track WHERE present = 1"
    ).fetchone()["c"]

    unique = conn.execute(
        "SELECT count(DISTINCT audio_content_id) c FROM track WHERE present = 1"
    ).fetchone()["c"]

    paths = conn.execute(
        "SELECT path FROM track WHERE present = 1"
    ).fetchall()
    formats = Counter()
    for row in paths:
        ext = os.path.splitext(row["path"])[1].lower()
        formats[ext] += 1

    codecs_rows = conn.execute(
        "SELECT ac.codec, count(*) c FROM track t "
        "JOIN audio_content ac ON ac.id = t.audio_content_id "
        "WHERE t.present = 1 AND ac.codec IS NOT NULL "
        "GROUP BY ac.codec"
    ).fetchall()
    codecs = {r["codec"]: r["c"] for r in codecs_rows}

    sr_rows = conn.execute(
        "SELECT ac.sample_rate, count(*) c FROM track t "
        "JOIN audio_content ac ON ac.id = t.audio_content_id "
        "WHERE t.present = 1 AND ac.sample_rate IS NOT NULL "
        "GROUP BY ac.sample_rate"
    ).fetchall()
    sample_rates = {r["sample_rate"]: r["c"] for r in sr_rows}

    br = conn.execute(
        "SELECT min(bitrate) mn, max(bitrate) mx, avg(bitrate) av "
        "FROM track WHERE present = 1 AND bitrate IS NOT NULL"
    ).fetchone()
    bitrate = {"min": br["mn"], "max": br["mx"], "avg": br["av"]}

    tag_dist = {}
    for r in conn.execute(
        "SELECT tag_completeness, count(*) c FROM track "
        "WHERE present = 1 GROUP BY tag_completeness"
    ).fetchall():
        tag_dist[r["tag_completeness"]] = r["c"]

    missing_title = conn.execute(
        "SELECT count(*) c FROM track WHERE present = 1 AND tag_title IS NULL"
    ).fetchone()["c"]
    missing_artist = conn.execute(
        "SELECT count(*) c FROM track WHERE present = 1 AND tag_artist IS NULL"
    ).fetchone()["c"]
    missing_album = conn.execute(
        "SELECT count(*) c FROM track WHERE present = 1 AND tag_album IS NULL"
    ).fetchone()["c"]
    missing_genre = conn.execute(
        "SELECT count(*) c FROM track WHERE present = 1 AND tag_genre IS NULL"
    ).fetchone()["c"]

    tags = {
        "fully_tagged": tag_dist.get(4, 0),
        "no_tags": tag_dist.get(0, 0),
        "missing_title": missing_title,
        "missing_artist": missing_artist,
        "missing_album": missing_album,
        "missing_genre": missing_genre,
    }

    errors = conn.execute(
        "SELECT count(*) c FROM ingest_error WHERE resolved_at IS NULL"
    ).fetchone()["c"]

    totals = conn.execute(
        "SELECT sum(ac.duration) dur, sum(t.size) sz FROM track t "
        "JOIN audio_content ac ON ac.id = t.audio_content_id "
        "WHERE t.present = 1"
    ).fetchone()
    total_duration = totals["dur"] or 0.0
    total_size = totals["sz"] or 0

    dup_groups = conn.execute(
        "SELECT count(*) c FROM dup_group"
    ).fetchone()["c"]

    enriched = conn.execute(
        "SELECT count(*) c FROM acoustid_cache"
    ).fetchone()["c"]

    return {
        "tracks": tracks,
        "unique_recordings": unique,
        "formats": dict(formats),
        "codecs": codecs,
        "sample_rates": sample_rates,
        "bitrate": bitrate,
        "tags": tags,
        "errors": errors,
        "total_duration": total_duration,
        "total_size": total_size,
        "dup_groups": dup_groups,
        "enriched": enriched,
    }


def _fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    if h > 0:
        return f"{h}h {m}m"
    return f"{m}m"


def _fmt_size(nbytes: int) -> str:
    if nbytes >= 1_000_000_000:
        return f"{nbytes / 1_000_000_000:.1f} GB"
    if nbytes >= 1_000_000:
        return f"{nbytes / 1_000_000:.1f} MB"
    return f"{nbytes / 1_000:.1f} KB"


def _fmt_bitrate(bps) -> str:
    if bps is None:
        return "-"
    return f"{bps / 1000:.0f} kbps"


def render_health(data: dict) -> str:
    lines = ["Library Health", "=" * 40, ""]
    lines.append(f"Tracks:              {data['tracks']}")
    lines.append(f"Unique recordings:   {data['unique_recordings']}")
    lines.append(f"Total duration:      {_fmt_duration(data['total_duration'])}")
    lines.append(f"Total size:          {_fmt_size(data['total_size'])}")
    lines.append("")

    lines.append("File Formats")
    lines.append("-" * 20)
    for fmt, count in sorted(data["formats"].items(), key=lambda x: -x[1]):
        lines.append(f"  {fmt:<8} {count:>5}")
    lines.append("")

    lines.append("Codecs")
    lines.append("-" * 20)
    for codec, count in sorted(data["codecs"].items(), key=lambda x: -x[1]):
        lines.append(f"  {codec:<8} {count:>5}")
    lines.append("")

    if data["sample_rates"]:
        lines.append("Sample Rates")
        lines.append("-" * 20)
        for sr, count in sorted(data["sample_rates"].items()):
            lines.append(f"  {sr:>6} Hz {count:>5}")
        lines.append("")

    lines.append("Bitrate")
    lines.append("-" * 20)
    lines.append(f"  Min: {_fmt_bitrate(data['bitrate']['min'])}")
    lines.append(f"  Max: {_fmt_bitrate(data['bitrate']['max'])}")
    lines.append(f"  Avg: {_fmt_bitrate(data['bitrate']['avg'])}")
    lines.append("")

    lines.append("Tag Completeness")
    lines.append("-" * 20)
    lines.append(f"  Fully tagged (4/4): {data['tags']['fully_tagged']}")
    lines.append(f"  No tags (0/4):      {data['tags']['no_tags']}")
    lines.append(f"  Missing title:      {data['tags']['missing_title']}")
    lines.append(f"  Missing artist:     {data['tags']['missing_artist']}")
    lines.append(f"  Missing album:      {data['tags']['missing_album']}")
    lines.append(f"  Missing genre:      {data['tags']['missing_genre']}")
    lines.append("")

    if data["errors"]:
        lines.append(f"Ingest errors:       {data['errors']}")
        lines.append("")

    if data["dup_groups"]:
        lines.append(f"Duplicate groups:    {data['dup_groups']}")
        lines.append("")

    lines.append(f"AcoustID lookups:    {data['enriched']}")

    return "\n".join(lines)
