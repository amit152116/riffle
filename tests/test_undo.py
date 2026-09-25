from pathlib import Path

from riffle import quarantine
from tests.test_quarantine import _fixture


def test_undo_restores_the_file(tmp_path):
    conn, lib, keeper, loser = _fixture(tmp_path)
    quarantine.apply_run(conn, 1, [lib])
    assert not loser.exists()
    result = quarantine.undo_run(conn, 1)
    assert result["restored"] == 1
    assert loser.exists()


def test_undo_marks_the_track_present_again(tmp_path):
    conn, lib, keeper, loser = _fixture(tmp_path)
    quarantine.apply_run(conn, 1, [lib])
    quarantine.undo_run(conn, 1)
    row = conn.execute(
        "SELECT present, absent_reason FROM track WHERE id = 2").fetchone()
    assert row["present"] == 1
    assert row["absent_reason"] is None


def test_undo_refuses_to_overwrite_a_path_taken_since(tmp_path):
    conn, lib, keeper, loser = _fixture(tmp_path)
    quarantine.apply_run(conn, 1, [lib])
    loser.write_bytes(b"something new lives here now")
    result = quarantine.undo_run(conn, 1)
    assert result["refused"] == 1
    assert loser.read_bytes() == b"something new lives here now"


def test_undo_returns_the_group_to_approved(tmp_path):
    conn, lib, keeper, loser = _fixture(tmp_path)
    quarantine.apply_run(conn, 1, [lib])
    quarantine.undo_run(conn, 1)
    assert conn.execute(
        "SELECT decision FROM dup_group WHERE id = 1"
    ).fetchone()["decision"] == "approved"


def test_undo_is_idempotent(tmp_path):
    conn, lib, keeper, loser = _fixture(tmp_path)
    quarantine.apply_run(conn, 1, [lib])
    quarantine.undo_run(conn, 1)
    second = quarantine.undo_run(conn, 1)
    assert second["restored"] == 0
