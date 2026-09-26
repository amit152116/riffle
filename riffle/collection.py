"""Collection browsing, filtering, and statistics."""
from __future__ import annotations


_VALID_SORTS = {"artist": "t.tag_artist", "title": "t.tag_title",
                "bpm": "af.bpm", "key": "af.key_name", "loudness": "af.loudness_lufs",
                "energy": "af.energy", "danceability": "af.danceability",
                "genre": "t.tag_genre"}


def browse(conn, *, sort_by: str = "artist", genre: str | None = None,
           bpm_range: tuple[float, float] | None = None,
           key: str | None = None, cluster: int | None = None,
           limit: int = 50, offset: int = 0) -> list[dict]:
    order_col = _VALID_SORTS.get(sort_by, "t.tag_artist")

    query = (
        "SELECT t.id, t.path, t.tag_artist, t.tag_title, t.tag_genre, "
        "af.bpm, af.key_name, af.scale, af.energy, af.danceability, af.loudness_lufs "
        "FROM track t "
        "LEFT JOIN audio_features af ON af.audio_content_id = t.audio_content_id "
        "WHERE t.present = 1"
    )
    params: list = []

    if genre is not None:
        query += " AND t.tag_genre LIKE ?"
        params.append(f"%{genre}%")
    if bpm_range is not None:
        query += " AND af.bpm >= ? AND af.bpm < ?"
        params.extend(bpm_range)
    if key is not None:
        query += " AND af.key_name = ?"
        params.append(key)
    if cluster is not None:
        query += (
            " AND t.audio_content_id IN ("
            "SELECT ca.audio_content_id FROM cluster_assignment ca "
            "JOIN cluster_run cr ON cr.id = ca.run_id "
            "WHERE ca.cluster_id = ? "
            "ORDER BY cr.id DESC LIMIT 1)"
        )
        params.append(cluster)

    query += f" ORDER BY {order_col} COLLATE NOCASE"
    query += " LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    return [dict(r) for r in conn.execute(query, params).fetchall()]


def stats(conn) -> dict:
    total = conn.execute("SELECT count(*) c FROM track WHERE present = 1").fetchone()["c"]
    with_features = conn.execute(
        "SELECT count(*) c FROM track t "
        "JOIN audio_features af ON af.audio_content_id = t.audio_content_id "
        "WHERE t.present = 1"
    ).fetchone()["c"]
    with_metadata = conn.execute(
        "SELECT count(*) c FROM track t "
        "JOIN musicbrainz_match mm ON mm.audio_content_id = t.audio_content_id "
        "WHERE t.present = 1"
    ).fetchone()["c"]

    genre_rows = conn.execute(
        "SELECT tag_genre, count(*) c FROM track "
        "WHERE present = 1 AND tag_genre IS NOT NULL "
        "GROUP BY tag_genre ORDER BY c DESC"
    ).fetchall()
    genre_dist = [{"genre": r["tag_genre"], "count": r["c"]} for r in genre_rows]

    bpm_buckets = [(60, 80), (80, 100), (100, 120), (120, 140), (140, 160), (160, 200)]
    bpm_hist = []
    for lo, hi in bpm_buckets:
        n = conn.execute(
            "SELECT count(*) c FROM audio_features WHERE bpm >= ? AND bpm < ?",
            (lo, hi)
        ).fetchone()["c"]
        bpm_hist.append({"bucket": f"{lo}-{hi}", "count": n})
    over = conn.execute("SELECT count(*) c FROM audio_features WHERE bpm >= 200").fetchone()["c"]
    bpm_hist.append({"bucket": "200+", "count": over})

    key_rows = conn.execute(
        "SELECT key_name, scale, count(*) c FROM audio_features "
        "WHERE key_name IS NOT NULL GROUP BY key_name, scale ORDER BY c DESC"
    ).fetchall()
    key_dist = [{"key": r["key_name"], "scale": r["scale"], "count": r["c"]} for r in key_rows]

    artist_rows = conn.execute(
        "SELECT tag_artist, count(*) c FROM track "
        "WHERE present = 1 AND tag_artist IS NOT NULL "
        "GROUP BY tag_artist ORDER BY c DESC LIMIT 20"
    ).fetchall()
    top_artists = [{"artist": r["tag_artist"], "count": r["c"]} for r in artist_rows]

    return {
        "total_tracks": total,
        "total_with_features": with_features,
        "total_with_metadata": with_metadata,
        "genre_distribution": genre_dist,
        "bpm_histogram": bpm_hist,
        "key_distribution": key_dist,
        "top_artists": top_artists,
    }


def render_browse(rows: list[dict]) -> str:
    if not rows:
        return "No tracks found."
    lines = [f"{'#':<4} {'Artist':<25} {'Title':<30} {'Genre':<12} {'BPM':>5} {'Key':<8} {'Energy':>6}"]
    lines.append("-" * 94)
    for i, r in enumerate(rows, 1):
        bpm = f"{r.get('bpm', 0):.0f}" if r.get("bpm") else "-"
        key_str = f"{r.get('key_name', '-')} {r.get('scale', '')[:3]}" if r.get("key_name") else "-"
        energy = f"{r.get('energy', 0):.2f}" if r.get("energy") is not None else "-"
        lines.append(
            f"{i:<4} {(r.get('tag_artist') or '-'):<25.25} "
            f"{(r.get('tag_title') or '-'):<30.30} "
            f"{(r.get('tag_genre') or '-'):<12.12} "
            f"{bpm:>5} {key_str:<8} {energy:>6}"
        )
    return "\n".join(lines)


def render_stats(data: dict) -> str:
    lines = ["Collection Statistics", "=" * 40, ""]
    lines.append(f"Total tracks:        {data['total_tracks']}")
    lines.append(f"With features:       {data['total_with_features']}")
    lines.append(f"With metadata:       {data['total_with_metadata']}")

    if data["genre_distribution"]:
        lines.append("")
        lines.append("Genres")
        lines.append("-" * 20)
        for g in data["genre_distribution"][:10]:
            lines.append(f"  {g['genre']:<20} {g['count']:>5}")

    if data["bpm_histogram"]:
        lines.append("")
        lines.append("BPM Distribution")
        lines.append("-" * 20)
        for b in data["bpm_histogram"]:
            lines.append(f"  {b['bucket']:<10} {b['count']:>5}")

    if data["key_distribution"]:
        lines.append("")
        lines.append("Key Distribution (top 10)")
        lines.append("-" * 20)
        for k in data["key_distribution"][:10]:
            lines.append(f"  {k['key']} {k['scale']:<6} {k['count']:>5}")

    return "\n".join(lines)
