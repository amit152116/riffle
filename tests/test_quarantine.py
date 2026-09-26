import os
from pathlib import Path

import pytest

from riffle import hashing, quarantine, scan, store
from tests.fixtures import make_tone


def _fixture(tmp_path, approve_it=True):
    lib = tmp_path / "lib"
    keeper = make_tone(lib / "keep.flac", seconds=5.0)
    loser = lib / "dupe.flac"
    loser.write_bytes(keeper.read_bytes())

    conn = store.connect(tmp_path / "db.sqlite")
    conn.execute("INSERT INTO match_run (id, status) VALUES (1,'complete')")
    conn.execute("INSERT INTO audio_content (id, audio_hash, hash_method) "
                 "VALUES (1,?, 'streamhash')",
                 (hashing.audio_identity(keeper).audio_hash,))
    decision = "approved" if approve_it else "proposed"
    conn.execute("INSERT INTO dup_group (id, run_id, tier, decision) "
                 "VALUES (1,1,1,?)", (decision,))
    for tid, path, is_keeper in ((1, keeper, 1), (2, loser, 0)):
        st = path.stat()
        conn.execute(
            "INSERT INTO track (id, path, size, mtime, audio_content_id, "
            " present) VALUES (?,?,?,?,1,1)",
            (tid, str(path), st.st_size, st.st_mtime))
        conn.execute(
            "INSERT INTO run_track (run_id, track_id, path, size, mtime, "
            " audio_hash, hash_method) VALUES (1,?,?,?,?,?, 'streamhash')",
            (tid, str(path), st.st_size, st.st_mtime,
             hashing.audio_identity(path).audio_hash))
        conn.execute("INSERT INTO group_member (group_id, track_id, "
                     " is_keeper) VALUES (1,?,?)",
                     (tid, is_keeper))
    return conn, lib, keeper, loser


def test_apply_moves_the_loser_and_keeps_the_keeper(tmp_path):
    conn, lib, keeper, loser = _fixture(tmp_path)
    result = quarantine.apply_run(conn, 1, [lib])
    assert result["moved"] == 1
    assert keeper.exists()
    assert not loser.exists()
    moved_to = conn.execute(
        "SELECT dst_path FROM quarantine_log").fetchone()["dst_path"]
    assert os.path.exists(moved_to)


def test_apply_skips_unapproved_groups(tmp_path):
    conn, lib, keeper, loser = _fixture(tmp_path, approve_it=False)
    result = quarantine.apply_run(conn, 1, [lib])
    assert result["moved"] == 0
    assert loser.exists()


def test_apply_skips_the_group_when_the_keeper_is_gone(tmp_path):
    conn, lib, keeper, loser = _fixture(tmp_path)
    keeper.unlink()
    result = quarantine.apply_run(conn, 1, [lib])
    assert result["moved"] == 0
    assert result["skipped_groups"] == 1
    assert loser.exists()  # zero-copy state avoided


def test_apply_skips_a_modified_loser(tmp_path):
    conn, lib, keeper, loser = _fixture(tmp_path)
    make_tone(loser, seconds=5.0, freq=1700)
    result = quarantine.apply_run(conn, 1, [lib])
    assert result["moved"] == 0
    assert loser.exists()


def test_quarantined_track_is_marked_not_missing(tmp_path):
    conn, lib, keeper, loser = _fixture(tmp_path)
    quarantine.apply_run(conn, 1, [lib])
    row = conn.execute(
        "SELECT present, absent_reason FROM track WHERE id = 2").fetchone()
    assert row["present"] == 0
    assert row["absent_reason"] == "quarantined"


def test_apply_never_overwrites_an_existing_quarantine_file(tmp_path):
    conn, lib, keeper, loser = _fixture(tmp_path)
    qdir = quarantine.quarantine_dir_for(loser, [lib])
    dst = qdir / loser.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(b"occupied")
    quarantine.apply_run(conn, 1, [lib])
    assert dst.read_bytes() == b"occupied"


def test_quarantine_dir_is_on_the_same_filesystem(tmp_path):
    conn, lib, keeper, loser = _fixture(tmp_path)
    qdir = quarantine.quarantine_dir_for(loser, [lib])
    qdir.mkdir(parents=True, exist_ok=True)
    assert qdir.stat().st_dev == loser.stat().st_dev


def test_read_only_destination_records_a_failure_not_a_loss(tmp_path):
    # Review Focus 4: the move fails, the source must survive, and the run
    # must end in an explicit recorded state.
    conn, lib, keeper, loser = _fixture(tmp_path)
    qdir = quarantine.quarantine_dir_for(loser, [lib])
    qdir.mkdir(parents=True, exist_ok=True)
    os.chmod(qdir, 0o500)
    try:
        result = quarantine.apply_run(conn, 1, [lib])
    finally:
        os.chmod(qdir, 0o700)
    assert result["failed"] == 1
    assert loser.exists()
    assert conn.execute(
        "SELECT state FROM quarantine_log").fetchone()["state"] == "failed"


def test_group_decision_becomes_applied(tmp_path):
    conn, lib, keeper, loser = _fixture(tmp_path)
    quarantine.apply_run(conn, 1, [lib])
    assert conn.execute(
        "SELECT decision FROM dup_group WHERE id = 1"
    ).fetchone()["decision"] == "applied"


def _two_loser_fixture(tmp_path, loser_a_name="a/song.flac",
                       loser_b_name="b/song.flac"):
    """A keeper plus two losers, which may share a basename in different
    subdirectories -- review finding I2."""
    lib = tmp_path / "lib"
    keeper = make_tone(lib / "keep.flac", seconds=5.0)
    loser_a = lib / loser_a_name
    loser_a.parent.mkdir(parents=True, exist_ok=True)
    loser_a.write_bytes(keeper.read_bytes())
    loser_b = lib / loser_b_name
    loser_b.parent.mkdir(parents=True, exist_ok=True)
    loser_b.write_bytes(keeper.read_bytes())

    conn = store.connect(tmp_path / "db.sqlite")
    conn.execute("INSERT INTO match_run (id, status) VALUES (1,'complete')")
    conn.execute("INSERT INTO audio_content (id, audio_hash, hash_method) "
                 "VALUES (1,?, 'streamhash')",
                 (hashing.audio_identity(keeper).audio_hash,))
    conn.execute("INSERT INTO dup_group (id, run_id, tier, decision) "
                 "VALUES (1,1,1,'approved')")
    for tid, path, is_keeper in ((1, keeper, 1), (2, loser_a, 0),
                                 (3, loser_b, 0)):
        st = path.stat()
        conn.execute(
            "INSERT INTO track (id, path, size, mtime, audio_content_id, "
            " present) VALUES (?,?,?,?,1,1)",
            (tid, str(path), st.st_size, st.st_mtime))
        conn.execute(
            "INSERT INTO run_track (run_id, track_id, path, size, mtime, "
            " audio_hash, hash_method) VALUES (1,?,?,?,?,?, 'streamhash')",
            (tid, str(path), st.st_size, st.st_mtime,
             hashing.audio_identity(path).audio_hash))
        conn.execute("INSERT INTO group_member (group_id, track_id, "
                     " is_keeper) VALUES (1,?,?)",
                     (tid, is_keeper))
    return conn, lib, keeper, loser_a, loser_b


def test_duplicate_basenames_in_different_directories_both_quarantine(tmp_path):
    # Review finding I2: dst = qdir / src.name collided on same-named
    # losers from different source directories -- the most common real
    # duplicate shape (e.g. the same rip filed under two album folders).
    conn, lib, keeper, loser_a, loser_b = _two_loser_fixture(tmp_path)
    result = quarantine.apply_run(conn, 1, [lib])
    assert result["moved"] == 2
    assert result["failed"] == 0
    assert not loser_a.exists()
    assert not loser_b.exists()
    dst_paths = [r["dst_path"] for r in conn.execute(
        "SELECT dst_path FROM quarantine_log WHERE state = 'moved'")]
    assert len(dst_paths) == 2
    assert len(set(dst_paths)) == 2
    assert all(os.path.exists(p) for p in dst_paths)


def test_source_unlink_failure_rolls_back_the_link_and_is_recorded(tmp_path):
    # Review finding C1/I1a: previously, a failure between the successful
    # link and the source unlink left an orphaned hardlink in quarantine
    # with either no log row (C1) or a 'failed' row that then permanently
    # blocked retry with EEXIST (I1a), since the orphan was never cleaned
    # up. The fix must leave the source untouched, remove the orphan link,
    # and record a 'failed' row -- durable, not silent -- so a later apply
    # can retry cleanly.
    conn, lib, keeper, loser = _fixture(tmp_path)
    os.chmod(lib, 0o500)  # loser's own directory becomes unwritable
    try:
        result = quarantine.apply_run(conn, 1, [lib])
    finally:
        os.chmod(lib, 0o700)

    assert result["moved"] == 0
    assert result["failed"] == 1
    assert loser.exists()  # source untouched

    rows = conn.execute("SELECT * FROM quarantine_log").fetchall()
    assert len(rows) == 1  # a row exists -- not the 0-row black hole of C1
    assert rows[0]["state"] == "failed"
    dst = Path(rows[0]["dst_path"])
    assert not dst.exists()  # the orphaned link was rolled back

    # The group must stay approved, not applied, so it can be retried.
    assert conn.execute(
        "SELECT decision FROM dup_group WHERE id = 1"
    ).fetchone()["decision"] == "approved"

    # A second, unobstructed attempt now succeeds cleanly -- no EEXIST from
    # a leftover orphan.
    result2 = quarantine.apply_run(conn, 1, [lib])
    assert result2["moved"] == 1
    assert not loser.exists()


def test_group_stays_approved_when_any_loser_fails(tmp_path):
    # Review finding I1b: the group was marked 'applied' unconditionally,
    # even when a loser failed, which made that loser unreachable by any
    # future apply (apply only acts on 'approved' groups). Pre-occupying
    # loser_a's specific mirrored destination (not the shared qdir, which
    # would block loser_b too) makes only that one loser fail.
    conn, lib, keeper, loser_a, loser_b = _two_loser_fixture(tmp_path)
    dst_a = quarantine._quarantine_destination(loser_a, [lib])
    dst_a.parent.mkdir(parents=True, exist_ok=True)
    dst_a.write_bytes(b"occupied")

    result = quarantine.apply_run(conn, 1, [lib])

    assert result["moved"] == 1
    assert result["failed"] == 1
    assert conn.execute(
        "SELECT decision FROM dup_group WHERE id = 1"
    ).fetchone()["decision"] == "approved"

    # Clearing the obstruction and retrying now moves the one that
    # previously failed.
    dst_a.unlink()
    result2 = quarantine.apply_run(conn, 1, [lib])
    assert result2["moved"] == 1
    assert conn.execute(
        "SELECT decision FROM dup_group WHERE id = 1"
    ).fetchone()["decision"] == "applied"


def test_modified_loser_is_counted_not_silently_skipped(tmp_path):
    # Review finding I1c: a loser that changed since the run was skipped
    # with no signal at all.
    conn, lib, keeper, loser = _fixture(tmp_path)
    make_tone(loser, seconds=5.0, freq=1700)  # content drifted since the run
    result = quarantine.apply_run(conn, 1, [lib])
    assert result["moved"] == 0
    assert result["modified"] == 1
    assert loser.exists()
