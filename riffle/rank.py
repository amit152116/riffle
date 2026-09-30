"""Keeper selection.

Only tier 0 and tier 1 groups may authorize quarantine, and a group that
cohered only through a chain never does. A radio edit is not an inferior copy
of the album version, so ranking it a loser for being shorter would be wrong.
"""
from __future__ import annotations

import json
from pathlib import Path

LOSSLESS_SUFFIXES = frozenset({".flac", ".wav", ".alac", ".ape", ".wv"})

# A copy whose low-pass sits this far below the widest copy in its group is a
# re-encode of a worse source, whatever its nominal bitrate says.
BANDWIDTH_TOLERANCE_HZ = 1500.0
# "Measured, no low-pass cliff". 21 kHz, not Nyquist: 20 vs 22 kHz is
# inaudible and must stay within BANDWIDTH_TOLERANCE_HZ.
FULL_BAND_HZ = 21000.0


def rank_key(row) -> tuple:
    """Higher sorts better. Path is the final, deterministic tie-break.

    Order: lossless > bitrate > tag completeness > longest duration >
    oldest mtime. (Effective bandwidth is applied before this key, in
    _bandwidth_contenders, because it is a group-relative comparison.) A missing mtime sorts *below* every real one -- mapping
    it to 0.0 would make it outrank every real, positive -mtime, the
    opposite of a safe default when nothing is actually known about it.
    """
    suffix = Path(row["path"]).suffix.lower()
    has_mtime = row["mtime"] is not None
    return (
        1 if suffix in LOSSLESS_SUFFIXES else 0,
        row["bitrate"] or 0,
        row["tag_completeness"] or 0,
        row["duration"] or 0.0,
        1 if has_mtime else 0,
        (-row["mtime"]) if has_mtime else 0.0,
    )


def _is_lossless(row) -> bool:
    return Path(row["path"]).suffix.lower() in LOSSLESS_SUFFIXES


def _effective_cutoff(row) -> float | None:
    """None when the file has no usable measurement (absent or unmeasurable)."""
    if row["cliff_db"] is None:
        return None
    return row["cutoff_hz"] if row["cutoff_hz"] is not None else FULL_BAND_HZ


def _bandwidth_contenders(rows: list) -> list:
    """Members still in the running once real bandwidth is considered.

    Only files within BANDWIDTH_TOLERANCE_HZ of the widest one stay; the
    normal rank_key then chooses among them. Filtering first, instead of
    comparing pairs, keeps the result independent of input order (a pairwise
    "clearly wider wins, else bitrate" rule can cycle across three files).
    If any member is unmeasured there is no fair comparison, so nobody is
    excluded. Lossless members always stay: rank_key already puts them first.
    """
    cutoffs = [_effective_cutoff(r) for r in rows]
    if any(c is None for c in cutoffs):
        return rows
    widest = max(cutoffs)
    return [r for r, c in zip(rows, cutoffs)
            if c >= widest - BANDWIDTH_TOLERANCE_HZ or _is_lossless(r)]


def rank_group(conn, group_id: int) -> int | None:
    g = conn.execute("SELECT tier, formed_by_chain FROM dup_group WHERE id = ?",
                     (group_id,)).fetchone()
    if g is None or g["tier"] not in (0, 1) or g["formed_by_chain"]:
        return None

    rows = conn.execute(
        "SELECT t.*, ac.duration AS duration, "
        "       ab.cutoff_hz AS cutoff_hz, ab.cliff_db AS cliff_db "
        "FROM group_member gm "
        "JOIN track t ON t.id = gm.track_id "
        "JOIN audio_content ac ON ac.id = t.audio_content_id "
        "LEFT JOIN audio_bandwidth ab ON ab.audio_content_id = ac.id "
        "WHERE gm.group_id = ? AND t.present = 1", (group_id,)
    ).fetchall()
    if len(rows) < 2:
        return None

    by_path = sorted(rows, key=lambda r: r["path"])
    ordered = sorted(by_path, key=rank_key, reverse=True)
    contenders = _bandwidth_contenders(by_path)
    keeper = sorted(contenders, key=rank_key, reverse=True)[0]
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
