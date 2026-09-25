"""Keeper selection.

Only tier 0 and tier 1 groups may authorize quarantine, and a group that
cohered only through a chain never does. A radio edit is not an inferior copy
of the album version, so ranking it a loser for being shorter would be wrong.
"""
from __future__ import annotations

import json
from pathlib import Path

LOSSLESS_SUFFIXES = frozenset({".flac", ".wav", ".alac", ".ape", ".wv"})


def rank_key(row) -> tuple:
    """Higher sorts better. Path is the final, deterministic tie-break."""
    suffix = Path(row["path"]).suffix.lower()
    return (
        1 if suffix in LOSSLESS_SUFFIXES else 0,
        row["bitrate"] or 0,
        row["tag_completeness"] or 0,
        (-row["mtime"]) if row["mtime"] is not None else 0.0,
    )


def rank_group(conn, group_id: int) -> int | None:
    g = conn.execute("SELECT tier, formed_by_chain FROM dup_group WHERE id = ?",
                     (group_id,)).fetchone()
    if g is None or g["tier"] not in (0, 1) or g["formed_by_chain"]:
        return None

    rows = conn.execute(
        "SELECT t.* FROM group_member gm JOIN track t ON t.id = gm.track_id "
        "WHERE gm.group_id = ? AND t.present = 1", (group_id,)
    ).fetchall()
    if len(rows) < 2:
        return None

    by_path = sorted(rows, key=lambda r: r["path"])
    ordered = sorted(by_path, key=rank_key, reverse=True)
    keeper = ordered[0]
    keeper_inode = (keeper["dev"], keeper["inode"])

    for row in ordered:
        same_file = (row["dev"], row["inode"]) == keeper_inode
        is_keeper = row["id"] == keeper["id"] or same_file
        conn.execute(
            "UPDATE group_member SET is_keeper = ?, rank_score = ? "
            "WHERE group_id = ? AND track_id = ?",
            (1 if is_keeper else 0, json.dumps(rank_key(row)),
             group_id, row["id"]),
        )
    return keeper["id"]
