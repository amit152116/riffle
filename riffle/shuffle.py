"""Smart shuffle and track similarity lookup."""
from __future__ import annotations

import random

from riffle import similarity


def find_similar(conn, track_id: int, n: int = 10) -> list[dict]:
    rows = conn.execute(
        "SELECT ts.*, t.path, t.tag_artist, t.tag_title, af.bpm, af.key_name, af.scale "
        "FROM track_similarity ts "
        "JOIN track t ON t.id = ts.neighbor_id "
        "LEFT JOIN audio_features af ON af.audio_content_id = t.audio_content_id "
        "WHERE ts.track_id = ? AND t.present = 1 "
        "ORDER BY ts.combined_score ASC LIMIT ?",
        (track_id, n)
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


def _bpm_jump_threshold(bpm_scale: float) -> float:
    """A BPM difference above this between consecutive playlist tracks is
    penalized as a jarring tempo jump. Scales with the library's own BPM
    spread (see similarity.compute_bpm_spread) instead of a fixed 15 BPM --
    a fallback spread of 60.0 reproduces the old fixed threshold exactly.
    """
    return bpm_scale * 0.25


# Rules for playlists built on the embedding index. Benchmarked on 416 real
# tracks: re-ranking a pool of ~80 nearest neighbours this way kept 72% of a
# playlist in the seed's style (the tempo/key-driven walk it replaces: 52%,
# chance 50%) with 92% key-compatible transitions and a 3.7 BPM average jump.
EMBEDDING_POOL = 80
ARTIST_PENALTY = 0.5        # same artist as one of the last 3 tracks
BPM_STEP_PENALTY = 0.3      # a jump beyond the library-relative threshold
BPM_CONTINUOUS_WEIGHT = 0.3  # grows with the size of the jump
KEY_BONUS = 0.2             # same or adjacent key (circle of fifths)


def _bpm_jump(a: float, b: float) -> float:
    """BPM difference where half and double time count as the same tempo."""
    return min(abs(a - b), abs(2 * a - b), abs(a - 2 * b))


def _uses_embedding_index(conn) -> bool:
    return conn.execute(
        "SELECT 1 FROM track_similarity WHERE config_hash = ? LIMIT 1",
        (similarity.embedding_config_hash(),)).fetchone() is not None


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

    pool = {r["track_id"]: dict(r) for r in conn.execute(query, params).fetchall()}
    if not pool:
        return []

    dup_exclusions = _get_duplicate_exclusion_set(conn)
    bpm_jump_threshold = _bpm_jump_threshold(similarity.compute_bpm_spread(conn))

    content_siblings: dict[int, set[int]] = {}
    for tid, row in pool.items():
        content_siblings.setdefault(row["audio_content_id"], set()).add(tid)

    def _exclusions_for(track_id: int) -> set[int]:
        siblings = content_siblings.get(pool[track_id]["audio_content_id"], set()) - {track_id}
        return dup_exclusions.get(track_id, set()) | siblings

    if seed_track_id is not None:
        if seed_track_id not in pool:
            raise ValueError(
                f"Seed track {seed_track_id} does not satisfy the active "
                "genre/BPM filters"
            )
        seed = seed_track_id
    else:
        seed = random.choice(list(pool.keys()))

    playlist = [pool[seed]]
    used_ids: set[int] = {seed}
    used_ids.update(_exclusions_for(seed))

    embedding_mode = _uses_embedding_index(conn)

    for _ in range(n - 1):
        current = playlist[-1]
        current_id = current["track_id"]

        candidates = find_similar(conn, current_id,
                                  n=EMBEDDING_POOL if embedding_mode else 40)
        best_score = float("inf")
        best_candidate = None

        for cand in candidates:
            cand_id = cand.get("neighbor_id")
            if cand_id is None or cand_id in used_ids or cand_id not in pool:
                continue

            score = cand["combined_score"]

            recent_artists = [p["tag_artist"] for p in playlist[-3:]]
            cand_artist = pool[cand_id].get("tag_artist")
            cand_bpm = pool[cand_id].get("bpm")
            curr_bpm = current.get("bpm")
            cand_key = pool[cand_id].get("key_name")
            curr_key = current.get("key_name")
            bpms_known = (not similarity._is_missing_bpm(cand_bpm)
                          and not similarity._is_missing_bpm(curr_bpm))
            keys_known = (not similarity._is_missing_key(cand_key)
                          and not similarity._is_missing_key(curr_key))
            key_compatible = False
            if keys_known:
                key_compatible = similarity.key_distance(
                    curr_key, current.get("scale") or "major",
                    cand_key, pool[cand_id].get("scale") or "major") <= 1

            if embedding_mode:
                if cand_artist is not None and cand_artist in recent_artists:
                    score += ARTIST_PENALTY
                if bpms_known:
                    jump = _bpm_jump(cand_bpm, curr_bpm)
                    if jump > bpm_jump_threshold:
                        score += BPM_STEP_PENALTY
                    score += BPM_CONTINUOUS_WEIGHT * min(
                        jump / (2 * bpm_jump_threshold), 1.0)
                if key_compatible:
                    score -= KEY_BONUS
            else:
                if cand_artist is not None and cand_artist in recent_artists:
                    score += 0.5
                if bpms_known and abs(cand_bpm - curr_bpm) > bpm_jump_threshold:
                    score += 0.3
                if key_compatible:
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
