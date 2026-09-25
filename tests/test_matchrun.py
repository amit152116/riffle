from audiolib import fingerprint, matchrun, scan, store
from tests.fixtures import make_tone, transcode


def _library(tmp_path):
    lib = tmp_path / "lib"
    src = make_tone(lib / "song.flac", seconds=40.0)
    transcode(src, lib / "song.mp3", codec="libmp3lame", bitrate="128k")
    make_tone(lib / "other.flac", seconds=40.0, freq=1500)
    conn = store.connect(tmp_path / "db.sqlite")
    scan.scan(conn, [lib])
    fingerprint.fingerprint_pending(conn)
    return conn


def test_run_match_records_a_run_and_snapshot(tmp_path):
    conn = _library(tmp_path)
    run_id = matchrun.run_match(conn)
    run = conn.execute("SELECT * FROM match_run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "complete"
    assert run["match_config"]
    snap = conn.execute(
        "SELECT count(*) c FROM run_track WHERE run_id = ?", (run_id,)
    ).fetchone()["c"]
    assert snap == 3


def test_transcoded_copy_is_found_as_tier_1(tmp_path):
    conn = _library(tmp_path)
    run_id = matchrun.run_match(conn)
    tiers = [r["tier"] for r in conn.execute(
        "SELECT tier FROM pair WHERE run_id = ?", (run_id,))]
    assert 1 in tiers


def test_unrelated_track_is_not_paired_at_tier_1(tmp_path):
    conn = _library(tmp_path)
    run_id = matchrun.run_match(conn)
    rows = conn.execute(
        "SELECT count(*) c FROM pair WHERE run_id = ? AND tier = 1", (run_id,)
    ).fetchone()
    assert rows["c"] == 1


def test_pairs_are_stored_in_canonical_order(tmp_path):
    conn = _library(tmp_path)
    run_id = matchrun.run_match(conn)
    for row in conn.execute("SELECT * FROM pair WHERE run_id = ?", (run_id,)):
        assert row["a_content_id"] < row["b_content_id"]


def test_absent_tracks_are_excluded(tmp_path):
    conn = _library(tmp_path)
    conn.execute("UPDATE track SET present = 0, absent_reason = 'missing'")
    run_id = matchrun.run_match(conn)
    assert conn.execute(
        "SELECT count(*) c FROM run_track WHERE run_id = ?", (run_id,)
    ).fetchone()["c"] == 0
