"""Persisted confirmation.

Only tier 0 and tier 1 groups may be approved, and never one that cohered
through a chain. `apply` acts on approved groups only, so the safety boundary
lives in the data rather than in what a command is assumed to mean.
"""
from __future__ import annotations

from datetime import datetime, timezone


class ApprovalError(Exception):
    """The group cannot be approved."""


def _eligible(row) -> bool:
    return row["tier"] in (0, 1) and not row["formed_by_chain"]


def _get(conn, run_id: int, group_id: int):
    row = conn.execute(
        "SELECT * FROM dup_group WHERE id = ? AND run_id = ?",
        (group_id, run_id)).fetchone()
    if row is None:
        raise ApprovalError(f"no group {group_id} in run {run_id}")
    return row


def approve_group(conn, run_id: int, group_id: int) -> None:
    row = _get(conn, run_id, group_id)
    if not _eligible(row):
        reason = ("it formed through a chain" if row["formed_by_chain"]
                  else f"it is tier {row['tier']}")
        raise ApprovalError(
            f"group {group_id} cannot authorize quarantine: {reason}")
    conn.execute(
        "UPDATE dup_group SET decision = 'approved', decided_at = ? "
        "WHERE id = ?", (datetime.now(timezone.utc).isoformat(), group_id))


def reject_group(conn, run_id: int, group_id: int) -> None:
    _get(conn, run_id, group_id)
    conn.execute(
        "UPDATE dup_group SET decision = 'rejected', decided_at = ? "
        "WHERE id = ?", (datetime.now(timezone.utc).isoformat(), group_id))


def approve_tier(conn, run_id: int, tier: int) -> list[int]:
    """Preview: the groups `commit_tier` would approve. Changes nothing."""
    return [r["id"] for r in conn.execute(
        "SELECT id FROM dup_group WHERE run_id = ? AND tier = ? "
        "AND formed_by_chain = 0 AND decision = 'proposed' ORDER BY id",
        (run_id, tier)) if tier in (0, 1)]


def commit_tier(conn, run_id: int, tier: int) -> int:
    ids = approve_tier(conn, run_id, tier)
    for group_id in ids:
        approve_group(conn, run_id, group_id)
    return len(ids)
