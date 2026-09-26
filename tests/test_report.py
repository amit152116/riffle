import json

from riffle import report, store


def _fixture(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    conn.execute("INSERT INTO match_run (id, created_at, status) "
                 "VALUES (1,'now','complete')")
    for cid, method in ((1, "streamhash"), (2, "whole_file")):
        conn.execute("INSERT INTO audio_content (id, audio_hash, hash_method, "
                     " duration) VALUES (?,?,?,180.0)",
                     (cid, f"h{cid}", method))
    conn.execute("INSERT INTO track (id, path, audio_content_id, present, "
                 " bitrate) VALUES (1,'/m/a.flac',1,1,900000)")
    conn.execute("INSERT INTO track (id, path, audio_content_id, present, "
                 " bitrate) VALUES (2,'/m/a.mp3',2,1,320000)")
    conn.execute("INSERT INTO pair (run_id, a_content_id, b_content_id, tier, "
                 " coverage_a, coverage_b, mean_bit_error, "
                 " matched_span_seconds, peak_vote_ratio, verified_direct) "
                 "VALUES (1,1,2,1,0.98,0.97,1.2,180.0,0.6,1)")
    conn.execute("INSERT INTO dup_group (id, run_id, tier, formed_by_chain) "
                 "VALUES (1,1,1,0)")
    for cid, tid, keeper in ((1, 1, 1), (2, 2, 0)):
        conn.execute("INSERT INTO group_content (group_id, audio_content_id) "
                     "VALUES (1,?)", (cid,))
        conn.execute("INSERT INTO group_member (group_id, track_id, "
                     " is_keeper) VALUES (1,?,?)",
                     (tid, keeper))
    return conn


def test_report_lists_groups_and_members(tmp_path):
    data = report.report_data(_fixture(tmp_path), 1)
    assert len(data["groups"]) == 1
    g = data["groups"][0]
    assert g["tier"] == 1
    assert len(g["members"]) == 2
    assert [m["is_keeper"] for m in g["members"]].count(True) == 1


def test_report_includes_pairwise_evidence(tmp_path):
    data = report.report_data(_fixture(tmp_path), 1)
    ev = data["groups"][0]["evidence"]
    assert len(ev) == 1
    assert ev[0]["coverage_a"] == 0.98
    assert ev[0]["verified_direct"] is True


def test_report_flags_whole_file_identities(tmp_path):
    data = report.report_data(_fixture(tmp_path), 1)
    flagged = [m for m in data["groups"][0]["members"]
               if m["hash_method"] == "whole_file"]
    assert len(flagged) == 1
    assert data["warnings"]


def test_report_filters_by_tier(tmp_path):
    data = report.report_data(_fixture(tmp_path), 1, tier=2)
    assert data["groups"] == []


def test_report_is_json_serializable(tmp_path):
    json.dumps(report.report_data(_fixture(tmp_path), 1))


def test_render_text_mentions_the_keeper_and_the_decision(tmp_path):
    text = report.render_text(report.report_data(_fixture(tmp_path), 1))
    assert "KEEP" in text
    assert "proposed" in text


def test_render_text_survives_a_newline_in_a_path(tmp_path):
    conn = _fixture(tmp_path)
    conn.execute("UPDATE track SET path = ? WHERE id = 2",
                 ("/m/we'ird\nname.mp3",))
    text = report.render_text(report.report_data(conn, 1))
    # The path is escaped, so it cannot forge a line of its own.
    assert "\\n" in text


def test_render_text_survives_evidence_with_null_fields(tmp_path):
    # Review finding I7: a pair verified only for tier (chain detection's
    # direct verification when no full evidence was captured) carries tier
    # and verified_direct but every numeric evidence field is NULL. This is
    # exactly the group a person most needs to review, so the report must
    # not crash on it.
    conn = _fixture(tmp_path)
    conn.execute(
        "UPDATE pair SET coverage_a = NULL, coverage_b = NULL, "
        "mean_bit_error = NULL, matched_span_seconds = NULL, "
        "peak_vote_ratio = NULL, best_offset = NULL "
        "WHERE run_id = 1 AND a_content_id = 1 AND b_content_id = 2")
    text = report.render_text(report.report_data(conn, 1))
    assert "1~2" in text
