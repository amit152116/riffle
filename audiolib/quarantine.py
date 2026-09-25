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

import os
from datetime import datetime, timezone
from pathlib import Path

from audiolib import hashing, scan


class QuarantineError(Exception):
    """The move could not be performed safely."""


def quarantine_dir_for(path: Path, roots: list[Path]) -> Path:
    """A quarantine directory on the same filesystem as `path`.

    One per root, so a library spread across drives is handled rather than
    refused. Falls back to the file's own directory when no root matches.
    """
    path = Path(path).resolve()
    dev = path.stat().st_dev if path.exists() else None
    for root in sorted((Path(r).resolve() for r in roots),
                       key=lambda r: len(str(r)), reverse=True):
        if path.is_relative_to(root):
            if dev is None or root.stat().st_dev == dev:
                return root / scan.QUARANTINE_DIRNAME
    return path.parent / scan.QUARANTINE_DIRNAME


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
    moved = failed = skipped_groups = 0

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

        group_moved = 0
        for m in losers:
            src = Path(m["path"])
            if not _fresh_matches(src, m):
                continue  # changed since the run: leave it alone

            qdir = quarantine_dir_for(src, roots)
            dst = qdir / src.name
            try:
                qdir.mkdir(parents=True, exist_ok=True)
                os.link(src, dst)  # fails if dst exists (invariant 2)
                dst_hash = hashing.audio_identity(dst).audio_hash
                if dst_hash != m["audio_hash"]:
                    os.unlink(dst)
                    raise QuarantineError("destination hash mismatch")
                os.unlink(src)
            except (OSError, QuarantineError) as exc:
                conn.execute(
                    "INSERT INTO quarantine_log (run_id, track_id, group_id, "
                    " src_path, dst_path, src_hash, dst_hash, moved_at, state) "
                    "VALUES (?,?,?,?,?,?,NULL,?, 'failed')",
                    (run_id, m["track_id"], g["id"], str(src), str(dst),
                     m["audio_hash"], _now()))
                failed += 1
                continue

            conn.execute(
                "INSERT INTO quarantine_log (run_id, track_id, group_id, "
                " src_path, dst_path, src_hash, dst_hash, moved_at, state) "
                "VALUES (?,?,?,?,?,?,?,?, 'moved')",
                (run_id, m["track_id"], g["id"], str(src), str(dst),
                 m["audio_hash"], dst_hash, _now()))
            conn.execute(
                "UPDATE track SET present = 0, absent_reason = 'quarantined' "
                "WHERE id = ?", (m["track_id"],))
            moved += 1
            group_moved += 1

        conn.execute(
            "UPDATE dup_group SET decision = 'applied', decided_at = ? "
            "WHERE id = ?", (_now(), g["id"]))

    return {"moved": moved, "failed": failed,
            "skipped_groups": skipped_groups}
