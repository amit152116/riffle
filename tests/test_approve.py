import pytest

from audiolib import approve, store


def _fixture(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    conn.execute("INSERT INTO match_run (id, status) VALUES (1,'complete')")
    conn.execute("INSERT INTO dup_group (id, run_id, tier, formed_by_chain) "
                 "VALUES (1,1,1,0)")
    conn.execute("INSERT INTO dup_group (id, run_id, tier, formed_by_chain) "
                 "VALUES (2,1,2,0)")
    conn.execute("INSERT INTO dup_group (id, run_id, tier, formed_by_chain) "
                 "VALUES (3,1,1,1)")
    conn.execute("INSERT INTO dup_group (id, run_id, tier, formed_by_chain) "
                 "VALUES (4,1,0,0)")
    return conn


def test_groups_start_proposed(tmp_path):
    conn = _fixture(tmp_path)
    assert conn.execute(
        "SELECT decision FROM dup_group WHERE id = 1"
    ).fetchone()["decision"] == "proposed"


def test_approve_sets_the_decision(tmp_path):
    conn = _fixture(tmp_path)
    approve.approve_group(conn, 1, 1)
    row = conn.execute("SELECT decision, decided_at FROM dup_group "
                       "WHERE id = 1").fetchone()
    assert row["decision"] == "approved"
    assert row["decided_at"]


def test_reject_sets_the_decision(tmp_path):
    conn = _fixture(tmp_path)
    approve.reject_group(conn, 1, 1)
    assert conn.execute(
        "SELECT decision FROM dup_group WHERE id = 1"
    ).fetchone()["decision"] == "rejected"


def test_tier_2_group_cannot_be_approved(tmp_path):
    conn = _fixture(tmp_path)
    with pytest.raises(approve.ApprovalError):
        approve.approve_group(conn, 1, 2)


def test_chain_group_cannot_be_approved(tmp_path):
    conn = _fixture(tmp_path)
    with pytest.raises(approve.ApprovalError):
        approve.approve_group(conn, 1, 3)


def test_approve_tier_previews_without_committing(tmp_path):
    conn = _fixture(tmp_path)
    ids = approve.approve_tier(conn, 1, 1)
    assert ids == [1]
    assert conn.execute(
        "SELECT decision FROM dup_group WHERE id = 1"
    ).fetchone()["decision"] == "proposed"


def test_commit_tier_approves_only_eligible_groups(tmp_path):
    conn = _fixture(tmp_path)
    assert approve.commit_tier(conn, 1, 1) == 1
    decisions = {r["id"]: r["decision"] for r in conn.execute(
        "SELECT id, decision FROM dup_group")}
    assert decisions == {1: "approved", 2: "proposed",
                         3: "proposed", 4: "proposed"}


def test_tier_0_can_be_approved(tmp_path):
    conn = _fixture(tmp_path)
    approve.approve_group(conn, 1, 4)
    assert conn.execute(
        "SELECT decision FROM dup_group WHERE id = 4"
    ).fetchone()["decision"] == "approved"


def test_unknown_group_raises(tmp_path):
    conn = _fixture(tmp_path)
    with pytest.raises(approve.ApprovalError):
        approve.approve_group(conn, 1, 99)
