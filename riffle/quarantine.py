"""Moving losers out of the library, reversibly.

Invariants:
  1. No group ever reaches a state with zero surviving copies.
  2. Nothing is ever overwritten, on move or on undo.
  3. Every move is verified against freshly read bytes, not cached state.

`os.link` is used rather than `shutil.move`: it fails atomically when the
destination exists, and fails with EXDEV across filesystems instead of
silently degrading to copy-then-delete.
"""
from __future__ import annotations

import contextlib
import os
from datetime import datetime, timezone
from pathlib import Path

from riffle import hashing, scan


class QuarantineError(Exception):
    """The move could not be performed safely."""


def _matching_root(path: Path, roots: list[Path]) -> Path | None:
    """The most specific root `path` falls under, or None."""
    for root in sorted((Path(r).resolve() for r in roots),
                       key=lambda r: len(str(r)), reverse=True):
        if path.is_relative_to(root):
            return root
    return None


def quarantine_dir_for(path: Path, roots: list[Path]) -> Path:
    """A quarantine directory on the same filesystem as `path`.

    One per root, so a library spread across drives is handled rather than
    refused. Falls back to the file's own directory when no root matches.
    """
    path = Path(path).resolve()
    dev = path.stat().st_dev if path.exists() else None
    root = _matching_root(path, roots)
    if root is not None and (dev is None or root.stat().st_dev == dev):
        return root / scan.QUARANTINE_DIRNAME
    return path.parent / scan.QUARANTINE_DIRNAME


def _quarantine_destination(src: Path, roots: list[Path]) -> Path:
    """Where a loser lands in quarantine.

    Mirrors the file's path relative to whichever scan root it falls under,
    rather than a flat `qdir / src.name`: two files sharing a basename in
    different directories -- the most common real duplicate shape, e.g. the
    same rip filed under two album folders -- would otherwise collide on a
    single destination. Falls back to the bare filename when no root
    matches (quarantine_dir_for's own fallback case), where that directory
    already lives beside the source file and a same-basename collision
    cannot occur.
    """
    src = Path(src).resolve()
    qdir = quarantine_dir_for(src, roots)
    root = _matching_root(src, roots)
    if root is not None:
        return qdir / src.relative_to(root)
    return qdir / src.name


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fresh_matches(path: Path, snapshot) -> bool:
    if not path.exists():
        return False
    try:
        ident = hashing.audio_identity(path)
    except hashing.HashError:
        return False
    return (ident.audio_hash == snapshot["audio_hash"]
            and ident.hash_method == snapshot["hash_method"])


def apply_run(conn, run_id: int, roots: list[Path]) -> dict:
    """Quarantine every loser of every approved group in this run.

    Each loser's `quarantine_log` row is written as `state='failed'`
    *before* any filesystem mutation is attempted for it, and only flipped
    to `'moved'` after the link, the destination hash check, and the source
    unlink have all succeeded. A row therefore always exists once a loser
    has been attempted, even if the process is interrupted or a later step
    fails -- the row starts pessimistic and is only upgraded on full
    success, rather than only being written after the fact (which could
    leave a moved file with no record at all if that final write failed).

    If linking succeeds but the source unlink does not, the link is rolled
    back (unlinked) so the operation is atomic in effect: either both steps
    complete, or neither survives, and a retry never trips over an orphaned
    destination.

    A group is marked `'applied'` only if every loser in it was moved. A
    group with any `failed` loser stays `'approved'`, so a later `apply`
    can retry exactly the ones that did not complete; a group is applied
    once with nothing left to retry, not partially and unrecoverably.
    """
    moved = failed = skipped_groups = modified = 0

    groups = conn.execute(
        "SELECT * FROM dup_group WHERE run_id = ? AND decision = 'approved' "
        "ORDER BY id", (run_id,)).fetchall()

    for g in groups:
        members = conn.execute(
            "SELECT gm.track_id, gm.is_keeper, rt.path, rt.audio_hash, "
            "       rt.hash_method "
            "FROM group_member gm "
            "JOIN run_track rt ON rt.track_id = gm.track_id "
            "                 AND rt.run_id = ? "
            "WHERE gm.group_id = ? ORDER BY gm.track_id",
            (run_id, g["id"])).fetchall()

        keepers = [m for m in members if m["is_keeper"]]
        losers = [m for m in members if not m["is_keeper"]]

        # Invariant 1: verify every keeper from disk BEFORE touching a loser.
        if not keepers or not all(
            _fresh_matches(Path(k["path"]), k) for k in keepers
        ):
            skipped_groups += 1
            continue

        group_failed = False
        for m in losers:
            src = Path(m["path"])
            if not _fresh_matches(src, m):
                modified += 1  # changed since the run: leave it alone
                continue

            dst = _quarantine_destination(src, roots)
            cur = conn.execute(
                "INSERT INTO quarantine_log (run_id, track_id, group_id, "
                " src_path, dst_path, src_hash, dst_hash, moved_at, state) "
                "VALUES (?,?,?,?,?,?,NULL,?, 'failed')",
                (run_id, m["track_id"], g["id"], str(src), str(dst),
                 m["audio_hash"], _now()))
            log_id = cur.lastrowid

            linked = False
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.link(src, dst)  # fails if dst exists (invariant 2)
                linked = True
                dst_hash = hashing.audio_identity(dst).audio_hash
                if dst_hash != m["audio_hash"]:
                    raise QuarantineError("destination hash mismatch")
                os.unlink(src)
            except (OSError, QuarantineError, hashing.HashError):
                if linked:
                    with contextlib.suppress(OSError):
                        os.unlink(dst)  # roll back: never leave an orphan
                failed += 1
                group_failed = True
                continue

            conn.execute(
                "UPDATE quarantine_log SET dst_hash = ?, moved_at = ?, "
                "state = 'moved' WHERE id = ?",
                (dst_hash, _now(), log_id))
            conn.execute(
                "UPDATE track SET present = 0, absent_reason = 'quarantined' "
                "WHERE id = ?", (m["track_id"],))
            moved += 1

        if not group_failed:
            conn.execute(
                "UPDATE dup_group SET decision = 'applied', decided_at = ? "
                "WHERE id = ?", (_now(), g["id"]))

    return {"moved": moved, "failed": failed,
            "skipped_groups": skipped_groups, "modified": modified}


def undo_run(conn, run_id: int) -> dict:
    """Restore quarantined files, refusing to overwrite anything."""
    restored = refused = 0
    rows = conn.execute(
        "SELECT * FROM quarantine_log WHERE run_id = ? AND state = 'moved' "
        "ORDER BY id", (run_id,)).fetchall()

    for row in rows:
        src = Path(row["dst_path"])
        dst = Path(row["src_path"])
        if not src.exists():
            continue
        if dst.exists():  # invariant 2: never overwrite
            refused += 1
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.link(src, dst)
            os.unlink(src)
        except OSError:
            refused += 1
            continue

        conn.execute("UPDATE quarantine_log SET state = 'undone' WHERE id = ?",
                     (row["id"],))
        conn.execute(
            "UPDATE track SET present = 1, absent_reason = NULL WHERE id = ?",
            (row["track_id"],))
        conn.execute(
            "UPDATE dup_group SET decision = 'approved' WHERE id = ?",
            (row["group_id"],))
        restored += 1

    return {"restored": restored, "refused": refused}
