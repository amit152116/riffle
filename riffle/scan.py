"""Filesystem walk, identity assignment, tags, presence, and error state."""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import mutagen

from riffle import hashing

AUDIO_EXTENSIONS = frozenset(
    {".mp3", ".flac", ".m4a", ".ogg", ".opus", ".wav", ".wma", ".aac"}
)
QUARANTINE_DIRNAME = ".riffle-quarantine"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def walk_audio_files(roots: list[Path]) -> Iterator[Path]:
    """Yield audio files under roots.

    Directory symlinks are not followed, so a link back to an ancestor cannot
    loop. A `(dev, inode)` is yielded once, so hardlinks and file symlinks do
    not become separate tracks.
    """
    seen: set[tuple[int, int]] = set()
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = [d for d in dirnames if d != QUARANTINE_DIRNAME]
            for name in sorted(filenames):
                path = Path(dirpath) / name
                if path.suffix.lower() not in AUDIO_EXTENSIONS:
                    continue
                try:
                    st = path.stat()  # follows symlinks: same inode as target
                except OSError:
                    continue
                key = (st.st_dev, st.st_ino)
                if key in seen:
                    continue
                seen.add(key)
                yield path


def should_retry(row, st: os.stat_result) -> bool:
    """A recorded error is retried only when the file itself changed."""
    return not (
        row["dev"] == st.st_dev
        and row["inode"] == st.st_ino
        and row["size"] == st.st_size
        and row["mtime"] == st.st_mtime
    )


def _record_error(conn, path: Path, st, stage: str, message: str) -> None:
    row = conn.execute(
        "SELECT attempts FROM ingest_error WHERE path = ?", (str(path),)
    ).fetchone()
    attempts = (row["attempts"] + 1) if row else 1
    conn.execute(
        "INSERT INTO ingest_error "
        "(path, dev, inode, size, mtime, stage, message, attempts, "
        " last_attempt_at, resolved_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,NULL) "
        "ON CONFLICT(path) DO UPDATE SET "
        "dev=excluded.dev, inode=excluded.inode, size=excluded.size, "
        "mtime=excluded.mtime, stage=excluded.stage, "
        "message=excluded.message, attempts=excluded.attempts, "
        "last_attempt_at=excluded.last_attempt_at, resolved_at=NULL",
        (str(path), st.st_dev, st.st_ino, st.st_size, st.st_mtime,
         stage, message, attempts, _now()),
    )


def _read_tags(path: Path) -> dict:
    try:
        f = mutagen.File(path, easy=True)
    except Exception:
        f = None
    if f is None:
        return {"title": None, "artist": None, "album": None,
                "genre": None, "bitrate": None}

    def first(key):
        v = f.get(key)
        return v[0] if v else None

    return {
        "title": first("title"),
        "artist": first("artist"),
        "album": first("album"),
        "genre": first("genre"),
        "bitrate": getattr(f.info, "bitrate", None),
    }


def _content_id(conn, path: Path, ident) -> int:
    row = conn.execute(
        "SELECT id FROM audio_content WHERE hash_method = ? AND audio_hash = ?",
        (ident.hash_method, ident.audio_hash),
    ).fetchone()
    if row:
        return row["id"]
    cur = conn.execute(
        "INSERT INTO audio_content (audio_hash, hash_method, first_seen_at) "
        "VALUES (?,?,?)",
        (ident.audio_hash, ident.hash_method, _now()),
    )
    return cur.lastrowid


def scan(conn, roots: list[Path], verify_hashes: bool = False,
         retry_errors: bool = False) -> int:
    roots = [Path(r).resolve() for r in roots]
    cur = conn.execute(
        "INSERT INTO scan_run (started_at, roots, status) VALUES (?,?,'running')",
        (_now(), json.dumps([str(r) for r in roots])),
    )
    scan_id = cur.lastrowid

    try:
        for path in walk_audio_files(roots):
            try:
                str(path).encode("utf-8")
            except UnicodeEncodeError:
                # The OS surrogate-escapes bytes that are not valid UTF-8
                # (e.g. a filename in a stale legacy encoding). Every text
                # column downstream (ingest_error, track, ...) is UTF-8, and
                # storing this path anywhere would raise the same error --
                # letting it propagate here aborted scanning every other
                # file in the library. It cannot be tracked without a
                # storage change (e.g. an os.fsencode BLOB column), so it is
                # skipped, deterministically, on every scan until renamed.
                continue

            st = path.stat()

            err = conn.execute(
                "SELECT * FROM ingest_error WHERE path = ? AND resolved_at IS NULL",
                (str(path),),
            ).fetchone()
            if err and not retry_errors and not should_retry(err, st):
                continue

            existing = conn.execute(
                "SELECT * FROM track WHERE path = ?", (str(path),)
            ).fetchone()
            unchanged = (
                existing
                and existing["size"] == st.st_size
                and existing["mtime"] == st.st_mtime
                and existing["audio_content_id"] is not None
            )

            if unchanged and not verify_hashes:
                content_id = existing["audio_content_id"]
            else:
                try:
                    ident = hashing.audio_identity(path)
                except hashing.HashError as exc:
                    _record_error(conn, path, st, "hash", str(exc))
                    continue
                content_id = _content_id(conn, path, ident)

            tags = _read_tags(path)
            conn.execute(
                "INSERT INTO track (path, size, mtime, dev, inode, nlink, "
                " audio_content_id, bitrate, tag_title, tag_artist, tag_album, "
                " tag_genre, last_seen_scan_id, present, "
                " absent_reason, scanned_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,1,NULL,?) "
                "ON CONFLICT(path) DO UPDATE SET "
                "size=excluded.size, mtime=excluded.mtime, dev=excluded.dev, "
                "inode=excluded.inode, nlink=excluded.nlink, "
                "audio_content_id=excluded.audio_content_id, "
                "bitrate=excluded.bitrate, tag_title=excluded.tag_title, "
                "tag_artist=excluded.tag_artist, tag_album=excluded.tag_album, "
                "tag_genre=excluded.tag_genre, "
                "last_seen_scan_id=excluded.last_seen_scan_id, "
                "present=1, absent_reason=NULL, scanned_at=excluded.scanned_at",
                (str(path), st.st_size, st.st_mtime, st.st_dev, st.st_ino,
                 st.st_nlink, content_id, tags["bitrate"], tags["title"],
                 tags["artist"], tags["album"], tags["genre"],
                 scan_id, _now()),
            )
            conn.execute(
                "UPDATE ingest_error SET resolved_at = ? WHERE path = ?",
                (_now(), str(path)),
            )

        # Only tracks under this scan's roots may be marked absent, and only
        # after the scan completes. A failed scan marks nothing.
        #
        # This is an exact prefix comparison, not LIKE: LIKE treats '_' and
        # '%' in the root path as wildcards and ignores ASCII case, so a
        # root such as "my_music" could match an unrelated "myXmusic" or
        # "MY_MUSIC" directory and wrongly mark its tracks absent -- exactly
        # what scanning one root must never do to another.
        for root in roots:
            prefix = f"{root}{os.sep}"
            conn.execute(
                "UPDATE track SET present = 0, absent_reason = 'missing' "
                "WHERE substr(path, 1, ?) = ? AND present = 1 "
                "AND (last_seen_scan_id IS NULL OR last_seen_scan_id != ?)",
                (len(prefix), prefix, scan_id),
            )
        conn.execute(
            "UPDATE scan_run SET status='complete', completed_at=? WHERE id=?",
            (_now(), scan_id),
        )
    except Exception:
        conn.execute(
            "UPDATE scan_run SET status='failed', completed_at=? WHERE id=?",
            (_now(), scan_id),
        )
        raise
    return scan_id
