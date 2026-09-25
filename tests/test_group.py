from audiolib import fingerprint, group, matchrun, scan, store
from tests.fixtures import make_tone, transcode


def test_components_splits_disjoint_edges():
    comps = group.components({(1, 2), (2, 3), (5, 6)})
    assert sorted(sorted(c) for c in comps) == [[1, 2, 3], [5, 6]]


def test_components_of_no_edges_is_empty():
    assert group.components(set()) == []


def _library(tmp_path):
    lib = tmp_path / "lib"
    src = make_tone(lib / "song.flac", seconds=40.0)
    transcode(src, lib / "song.mp3", codec="libmp3lame", bitrate="128k")
    transcode(src, lib / "song.m4a", codec="aac", bitrate="192k")
    make_tone(lib / "other.flac", seconds=40.0, freq=1500)
    conn = store.connect(tmp_path / "db.sqlite")
    scan.scan(conn, [lib])
    fingerprint.fingerprint_pending(conn)
    return conn


def test_transcodes_form_one_clique_group(tmp_path):
    conn = _library(tmp_path)
    run_id = matchrun.run_match(conn)
    n = group.build_groups(conn, run_id)
    assert n == 1
    g = conn.execute("SELECT * FROM dup_group WHERE run_id = ?",
                     (run_id,)).fetchone()
    assert g["tier"] == 1
    assert g["formed_by_chain"] == 0
    assert g["decision"] == "proposed"
    members = conn.execute(
        "SELECT count(*) c FROM group_member WHERE group_id = ?", (g["id"],)
    ).fetchone()["c"]
    assert members == 3


def test_tier_0_identical_audio_groups_without_matching(tmp_path):
    lib = tmp_path / "lib"
    a = make_tone(lib / "a.flac", seconds=40.0)
    (lib / "copy.flac").write_bytes(a.read_bytes())
    conn = store.connect(tmp_path / "db.sqlite")
    scan.scan(conn, [lib])
    fingerprint.fingerprint_pending(conn)
    run_id = matchrun.run_match(conn)
    group.build_groups(conn, run_id)
    g = conn.execute("SELECT * FROM dup_group WHERE run_id = ?",
                     (run_id,)).fetchone()
    assert g["tier"] == 0
    assert conn.execute(
        "SELECT count(*) c FROM group_member WHERE group_id = ?", (g["id"],)
    ).fetchone()["c"] == 2


def test_chain_component_is_flagged_and_demoted(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    conn.execute("INSERT INTO match_run (id, status) VALUES (1, 'complete')")
    for cid in (1, 2, 3):
        conn.execute("INSERT INTO audio_content (id, audio_hash, hash_method) "
                     "VALUES (?,?, 'streamhash')", (cid, f"h{cid}"))
        conn.execute("INSERT INTO track (id, path, audio_content_id, present) "
                     "VALUES (?,?,?,1)", (cid, f"/x/{cid}.flac", cid))
    # A-B and B-C are tier 1; A-C is not present at all.
    for a, b in ((1, 2), (2, 3)):
        conn.execute("INSERT INTO pair (run_id, a_content_id, b_content_id, "
                     "tier, verified_direct) VALUES (1,?,?,1,1)", (a, b))
    # Force verification to report A-C as a non-match.
    group.build_groups(conn, 1, verifier=lambda a, b: 0 if (a, b) == (1, 3) else 1)
    g = conn.execute("SELECT * FROM dup_group WHERE run_id = 1").fetchone()
    assert g["formed_by_chain"] == 1
    assert g["tier"] == 2


def test_chain_group_cannot_authorize_quarantine(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    conn.execute("INSERT INTO match_run (id, status) VALUES (1, 'complete')")
    for cid in (1, 2, 3):
        conn.execute("INSERT INTO audio_content (id, audio_hash, hash_method) "
                     "VALUES (?,?, 'streamhash')", (cid, f"h{cid}"))
        conn.execute("INSERT INTO track (id, path, audio_content_id, present) "
                     "VALUES (?,?,?,1)", (cid, f"/x/{cid}.flac", cid))
    for a, b in ((1, 2), (2, 3)):
        conn.execute("INSERT INTO pair (run_id, a_content_id, b_content_id, "
                     "tier, verified_direct) VALUES (1,?,?,1,1)", (a, b))
    group.build_groups(conn, 1, verifier=lambda a, b: 0 if (a, b) == (1, 3) else 1)
    keepers = conn.execute(
        "SELECT count(*) c FROM group_member WHERE is_keeper = 1"
    ).fetchone()["c"]
    assert keepers == 0
