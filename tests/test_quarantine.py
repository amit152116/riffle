import os

import pytest

from audiolib import hashing, quarantine, scan, store
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
                     " audio_content_id, is_keeper) VALUES (1,?,1,?)",
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
