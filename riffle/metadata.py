"""AcoustID response parsing into structured MusicBrainz metadata."""
from __future__ import annotations

import json


def parse_acoustid_response(response_json: str) -> dict | None:
    data = json.loads(response_json)
    if data.get("status") != "ok":
        return None

    results = data.get("results", [])
    best = None
    for r in results:
        recordings = r.get("recordings")
        if not recordings:
            continue
        if best is None or r.get("score", 0) > best.get("score", 0):
            best = r

    if best is None:
        return None

    rec = best["recordings"][0]
    artists = [{"name": a["name"], "mbid": a["id"]}
               for a in rec.get("artists", [])]

    rgs = rec.get("releasegroups", [])
    release_title = rgs[0]["title"] if rgs else None
    release_mbid = rgs[0]["id"] if rgs else None

    return {
        "acoustid_id": best["id"],
        "recording_mbid": rec["id"],
        "recording_title": rec.get("title"),
        "artists_json": json.dumps(artists),
        "release_title": release_title,
        "release_mbid": release_mbid,
        "score": best.get("score"),
    }


from datetime import UTC, datetime


def parse_all(conn) -> dict:
    rows = conn.execute(
        "SELECT ac.lookup_key, ac.response_json, ac.audio_content_id "
        "FROM acoustid_cache ac "
        "LEFT JOIN musicbrainz_match mm ON mm.audio_content_id = ac.audio_content_id "
        "WHERE ac.audio_content_id IS NOT NULL AND mm.id IS NULL"
    ).fetchall()

    already = conn.execute(
        "SELECT count(*) c FROM musicbrainz_match"
    ).fetchone()["c"]

    parsed = no_match = failed = 0
    for row in rows:
        try:
            result = parse_acoustid_response(row["response_json"])
        except Exception:
            failed += 1
            continue

        if result is None:
            conn.execute(
                "INSERT INTO musicbrainz_match (audio_content_id, parsed_at) "
                "VALUES (?, ?) ON CONFLICT(audio_content_id) DO NOTHING",
                (row["audio_content_id"], datetime.now(UTC).isoformat()),
            )
            no_match += 1
            continue

        conn.execute(
            "INSERT INTO musicbrainz_match (audio_content_id, acoustid_id, "
            "recording_mbid, recording_title, artists_json, release_title, "
            "release_mbid, score, parsed_at) "
            "VALUES (?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(audio_content_id) DO UPDATE SET "
            "acoustid_id=excluded.acoustid_id, recording_mbid=excluded.recording_mbid, "
            "recording_title=excluded.recording_title, artists_json=excluded.artists_json, "
            "release_title=excluded.release_title, release_mbid=excluded.release_mbid, "
            "score=excluded.score, parsed_at=excluded.parsed_at",
            (row["audio_content_id"], result["acoustid_id"],
             result["recording_mbid"], result["recording_title"],
             result["artists_json"], result["release_title"],
             result["release_mbid"], result["score"],
             datetime.now(UTC).isoformat()),
        )
        parsed += 1

    return {"parsed": parsed, "no_match": no_match, "failed": failed, "cached": already}


def render_metadata(conn) -> str:
    total = conn.execute(
        "SELECT count(*) c FROM musicbrainz_match WHERE recording_mbid IS NOT NULL"
    ).fetchone()["c"]
    cache_total = conn.execute(
        "SELECT count(*) c FROM acoustid_cache WHERE audio_content_id IS NOT NULL"
    ).fetchone()["c"]

    lines = ["Metadata Report", "=" * 40, ""]
    lines.append(f"Matched:             {total}")
    lines.append(f"Cache entries:       {cache_total}")
    if cache_total > 0:
        rate = total / cache_total * 100
        lines.append(f"Match rate:          {rate:.1f}%")

    if total == 0:
        return "\n".join(lines)

    lines.append("")
    lines.append("Score Distribution")
    lines.append("-" * 20)
    for lo, hi, label in [(0.9, 1.01, "High (≥0.9)"), (0.5, 0.9, "Medium (0.5-0.9)"),
                          (0.0, 0.5, "Low (<0.5)")]:
        n = conn.execute(
            "SELECT count(*) c FROM musicbrainz_match WHERE score >= ? AND score < ?",
            (lo, hi)
        ).fetchone()["c"]
        lines.append(f"  {label:<20} {n:>5}")

    lines.append("")
    lines.append("Top Artists")
    lines.append("-" * 20)
    for r in conn.execute(
        "SELECT artists_json, count(*) c FROM musicbrainz_match "
        "WHERE recording_mbid IS NOT NULL "
        "GROUP BY artists_json ORDER BY c DESC LIMIT 10"
    ).fetchall():
        try:
            artists = json.loads(r["artists_json"])
            name = ", ".join(a["name"] for a in artists)
        except (json.JSONDecodeError, KeyError):
            name = "(unknown)"
        lines.append(f"  {name:<30} {r['c']:>5}")

    return "\n".join(lines)
