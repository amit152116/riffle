"""Smart shuffle and track similarity lookup."""
from __future__ import annotations

import random

from riffle import similarity


def find_similar(conn, track_id: int, n: int = 10) -> list[dict]:
    rows = conn.execute(
        "SELECT ts.*, t.path, t.tag_artist, t.tag_title, af.bpm, af.key_name, af.scale "
        "FROM track_similarity ts "
        "JOIN track t ON t.id = CASE WHEN ts.track_a_id = ? THEN ts.track_b_id "
        "                            ELSE ts.track_a_id END "
        "JOIN audio_features af ON af.audio_content_id = t.audio_content_id "
        "WHERE ts.track_a_id = ? OR ts.track_b_id = ? "
        "ORDER BY ts.combined_score ASC LIMIT ?",
        (track_id, track_id, track_id, n)
    ).fetchall()
    return [dict(r) for r in rows]


def resolve_track(conn, query: str) -> int:
    rows = conn.execute(
        "SELECT id FROM track WHERE present = 1 AND path LIKE ?",
        (f"%{query}%",)
    ).fetchall()
    if len(rows) == 0:
        raise ValueError(f"No track found matching '{query}'")
    if len(rows) > 1:
        raise ValueError(f"Ambiguous: {len(rows)} tracks match '{query}'")
    return rows[0]["id"]


def _get_duplicate_exclusion_set(conn) -> dict[int, set[int]]:
    groups = conn.execute(
        "SELECT gc.group_id, gc.audio_content_id "
        "FROM group_content gc "
        "JOIN dup_group dg ON dg.id = gc.group_id "
        "WHERE dg.decision IN ('approved', 'applied')"
    ).fetchall()

    group_members: dict[int, set[int]] = {}
    for r in groups:
        group_members.setdefault(r["group_id"], set()).add(r["audio_content_id"])

    content_to_tracks: dict[int, list[int]] = {}
    for r in conn.execute(
        "SELECT id, audio_content_id FROM track WHERE present = 1"
    ).fetchall():
        content_to_tracks.setdefault(r["audio_content_id"], []).append(r["id"])

    exclusions: dict[int, set[int]] = {}
    for members in group_members.values():
        all_track_ids: set[int] = set()
        for cid in members:
            all_track_ids.update(content_to_tracks.get(cid, []))
        for tid in all_track_ids:
            exclusions.setdefault(tid, set()).update(all_track_ids - {tid})

    return exclusions


def smart_shuffle(conn, *, seed_track_id: int | None = None,
                  n: int = 20, genre: str | None = None,
                  bpm_range: tuple[float, float] | None = None) -> list[dict]:
    query = (
        "SELECT t.id AS track_id, t.path, t.tag_artist, t.tag_title, t.tag_genre, "
        "t.audio_content_id, af.bpm, af.key_name, af.scale, af.energy "
        "FROM track t "
        "JOIN audio_features af ON af.audio_content_id = t.audio_content_id "
        "WHERE t.present = 1"
    )
    params: list = []
    if genre is not None:
        query += " AND t.tag_genre LIKE ?"
        params.append(f"%{genre}%")
    if bpm_range is not None:
        query += " AND af.bpm >= ? AND af.bpm < ?"
        params.extend(bpm_range)

    pool = {r["track_id"]: dict(r) for r in conn.execute(query, params).fetchall()}
    if not pool:
        return []

    dup_exclusions = _get_duplicate_exclusion_set(conn)

    content_siblings: dict[int, set[int]] = {}
    for tid, row in pool.items():
        content_siblings.setdefault(row["audio_content_id"], set()).add(tid)

    def _exclusions_for(track_id: int) -> set[int]:
        siblings = content_siblings.get(pool[track_id]["audio_content_id"], set()) - {track_id}
        return dup_exclusions.get(track_id, set()) | siblings

    if seed_track_id is not None and seed_track_id in pool:
        seed = seed_track_id
    else:
        seed = random.choice(list(pool.keys()))

    playlist = [pool[seed]]
    used_ids: set[int] = {seed}
    used_ids.update(_exclusions_for(seed))

    for _ in range(n - 1):
        current = playlist[-1]
        current_id = current["track_id"]

        candidates = find_similar(conn, current_id, n=40)
        best_score = float("inf")
        best_candidate = None

        for cand in candidates:
            cand_id = cand.get("track_b_id") if cand.get("track_a_id") == current_id \
                else cand.get("track_a_id")
            if cand_id is None or cand_id in used_ids or cand_id not in pool:
                continue

            score = cand["combined_score"]

            recent_artists = [p["tag_artist"] for p in playlist[-3:]]
            cand_artist = pool[cand_id].get("tag_artist")
            if cand_artist in recent_artists:
                score += 0.5

            cand_bpm = pool[cand_id].get("bpm") or 0
            curr_bpm = current.get("bpm") or 0
            if abs(cand_bpm - curr_bpm) > 15:
                score += 0.3

            cand_key = pool[cand_id].get("key_name") or "C"
            cand_scale = pool[cand_id].get("scale") or "major"
            curr_key = current.get("key_name") or "C"
            curr_scale = current.get("scale") or "major"
            if similarity.key_distance(curr_key, curr_scale, cand_key, cand_scale) <= 1:
                score -= 0.2

            if score < best_score:
                best_score = score
                best_candidate = cand_id

        if best_candidate is None:
            remaining = [tid for tid in pool if tid not in used_ids]
            if not remaining:
                break
            best_candidate = random.choice(remaining)

        playlist.append(pool[best_candidate])
        used_ids.add(best_candidate)
        used_ids.update(_exclusions_for(best_candidate))

    return playlist


def render_similar(rows: list[dict]) -> str:
    if not rows:
        return "No similar tracks found."
    lines = [f"{'#':<4} {'Score':>6} {'Artist':<25} {'Title':<30} {'BPM':>5} {'Key':<8}"]
    lines.append("-" * 80)
    for i, r in enumerate(rows, 1):
        bpm = f"{r.get('bpm', 0):.0f}" if r.get("bpm") else "-"
        key_str = f"{r.get('key_name', '-')} {r.get('scale', '')[:3]}" if r.get("key_name") else "-"
        lines.append(
            f"{i:<4} {r.get('combined_score', 0):>6.3f} "
            f"{(r.get('tag_artist') or '-'):<25.25} "
            f"{(r.get('tag_title') or '-'):<30.30} "
            f"{bpm:>5} {key_str:<8}"
        )
    return "\n".join(lines)


def render_playlist(rows: list[dict]) -> str:
    if not rows:
        return "Empty playlist."
    lines = [f"{'#':<4} {'Artist':<25} {'Title':<30} {'BPM':>5} {'Key':<8}"]
    lines.append("-" * 74)
    for i, r in enumerate(rows, 1):
        bpm = f"{r.get('bpm', 0):.0f}" if r.get("bpm") else "-"
        key_str = f"{r.get('key_name', '-')} {r.get('scale', '')[:3]}" if r.get("key_name") else "-"
        lines.append(
            f"{i:<4} {(r.get('tag_artist') or '-'):<25.25} "
            f"{(r.get('tag_title') or '-'):<30.30} "
            f"{bpm:>5} {key_str:<8}"
        )
    return "\n".join(lines)
