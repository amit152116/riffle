from riffle import fingerprint, group, matchrun, scan, store
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


def test_real_verification_stores_full_evidence_not_just_tier(tmp_path):
    # Review finding I7 (root cause): direct verification's real (non-test-
    # injected) path only wrote tier and verified_direct, leaving every
    # coverage/bit-error/span field NULL -- discarding evidence for exactly
    # the pairs a person most needs to inspect, and only surviving report
    # rendering because of report.py's separate defensive fix.
    #
    # The gap only shows up on a pair candidate generation actually missed
    # (no pre-existing row for it to merge onto), which a small, fully
    # redundant fixture like _library does not naturally produce -- every
    # pair is already found and carries full evidence before verification
    # even runs. Deleting one edge's row simulates a real capped-index miss.
    conn = _library(tmp_path)
    run_id = matchrun.run_match(conn)
    existing = conn.execute(
        "SELECT a_content_id, b_content_id FROM pair "
        "WHERE run_id = ? AND tier = 1 LIMIT 1", (run_id,)).fetchone()
    assert existing is not None  # sanity: the edge really exists first
    a, b = existing["a_content_id"], existing["b_content_id"]
    conn.execute(
        "DELETE FROM pair WHERE run_id = ? AND a_content_id = ? "
        "AND b_content_id = ?", (run_id, a, b))

    group.build_groups(conn, run_id)

    row = conn.execute(
        "SELECT * FROM pair WHERE run_id = ? AND a_content_id = ? "
        "AND b_content_id = ?", (run_id, a, b)
    ).fetchone()
    assert row is not None
    assert row["verified_direct"] == 1
    assert row["coverage_a"] is not None
    assert row["coverage_b"] is not None
    assert row["mean_bit_error"] is not None
    assert row["matched_span_seconds"] is not None


def _synthetic_run(conn, contents, tracks, pairs):
    """contents: ids; tracks: [(track_id, content_id)]; pairs: tier-1 edges."""
    conn.execute("INSERT INTO match_run (id, status) VALUES (1, 'complete')")
    for cid in contents:
        conn.execute("INSERT INTO audio_content (id, audio_hash, hash_method) "
                     "VALUES (?,?, 'streamhash')", (cid, f"h{cid}"))
    for tid, cid in tracks:
        conn.execute("INSERT INTO track (id, path, audio_content_id, present) "
                     "VALUES (?,?,?,1)", (tid, f"/x/{tid}.flac", cid))
    for a, b in pairs:
        conn.execute("INSERT INTO pair (run_id, a_content_id, b_content_id, "
                     "tier, verified_direct) VALUES (1,?,?,1,1)", (a, b))


def test_exact_copies_inside_a_fuzzy_group_are_not_grouped_twice(tmp_path):
    # Content 1 has two identical files (tracks 1, 2); content 2 is a
    # transcode of it (track 3). The fuzzy group already holds all three
    # tracks, so a second tier-0 group over tracks 1 and 2 would rank and
    # apply the same files under two independent keepers.
    conn = store.connect(tmp_path / "db.sqlite")
    _synthetic_run(conn, (1, 2), [(1, 1), (2, 1), (3, 2)], [(1, 2)])
    group.build_groups(conn, 1, verifier=lambda a, b: 1)
    groups = conn.execute("SELECT id, tier FROM dup_group").fetchall()
    assert [g["tier"] for g in groups] == [1]
    members = conn.execute(
        "SELECT track_id FROM group_member WHERE group_id = ?",
        (groups[0]["id"],)).fetchall()
    assert sorted(m["track_id"] for m in members) == [1, 2, 3]


def test_exact_copies_inside_a_chain_group_keep_their_tier_0_group(tmp_path):
    # A chain group can never authorize quarantine, so it must not swallow
    # the exact copies' own tier-0 group: that is their only actionable one.
    conn = store.connect(tmp_path / "db.sqlite")
    _synthetic_run(conn, (1, 2, 3), [(1, 1), (2, 1), (3, 2), (4, 3)],
                   [(1, 2), (2, 3)])
    group.build_groups(conn, 1, verifier=lambda a, b: 0 if (a, b) == (1, 3) else 1)
    tiers = sorted(r["tier"] for r in conn.execute("SELECT tier FROM dup_group"))
    assert tiers == [0, 2]
