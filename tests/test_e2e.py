import json

from audiolib import (fingerprint, group, matchrun, quarantine, rank,
                      report, scan, store, approve)
from tests.fixtures import make_tone, transcode, trim, retag


def _full_run(tmp_path, name="db.sqlite"):
    lib = tmp_path / "lib"
    src = make_tone(lib / "song.flac", seconds=60.0)
    transcode(src, lib / "song.mp3", codec="libmp3lame", bitrate="128k")
    copy = lib / "song-copy.flac"
    copy.write_bytes(src.read_bytes())
    retag(copy, title="Retagged")
    trim(src, lib / "song-edit.flac", start=5.0, duration=30.0)
    make_tone(lib / "unrelated.flac", seconds=60.0, freq=1700)

    conn = store.connect(tmp_path / name)
    scan.scan(conn, [lib])
    fingerprint.fingerprint_pending(conn)
    run_id = matchrun.run_match(conn)
    group.build_groups(conn, run_id)
    for row in conn.execute("SELECT id FROM dup_group WHERE run_id = ?",
                            (run_id,)):
        rank.rank_group(conn, row["id"])
    return conn, lib, run_id


def test_retagged_copy_is_tier_0_and_the_transcode_is_tier_1(tmp_path):
    conn, lib, run_id = _full_run(tmp_path)
    tiers = sorted(r["tier"] for r in conn.execute(
        "SELECT tier FROM dup_group WHERE run_id = ?", (run_id,)))
    assert 0 in tiers
    assert 1 in tiers


def test_the_edit_never_authorizes_quarantine(tmp_path):
    conn, lib, run_id = _full_run(tmp_path)
    for g in conn.execute(
        "SELECT id FROM dup_group WHERE run_id = ? AND tier = 2", (run_id,)
    ):
        keepers = conn.execute(
            "SELECT count(*) c FROM group_member "
            "WHERE group_id = ? AND is_keeper = 1", (g["id"],)).fetchone()["c"]
        assert keepers == 0


def test_full_cycle_applies_and_undoes(tmp_path):
    conn, lib, run_id = _full_run(tmp_path)
    approve.commit_tier(conn, run_id, 0)
    approve.commit_tier(conn, run_id, 1)
    before = sorted(p.name for p in lib.iterdir() if p.is_file())
    applied = quarantine.apply_run(conn, run_id, [lib])
    assert applied["moved"] >= 1
    quarantine.undo_run(conn, run_id)
    after = sorted(p.name for p in lib.iterdir() if p.is_file())
    assert before == after


def test_two_runs_produce_identical_reports(tmp_path):
    # "Reproducible" means: the same scanned state matched twice gives the
    # same report. It does not mean two *separately created* real files'
    # raw mtimes will coincide -- they never will, by definition, and that
    # is not a claim this tool makes or needs. _full_run(tmp_path) called
    # twice against the same tmp_path regenerates (overwrites) the same
    # file paths at two different real wall-clock moments, which changes
    # nothing about the matching *algorithm* but does change the real
    # mtime baked into rank_score's tie-break tuple -- that was confirmed
    # by inspection: is_keeper, tier assignment and every evidence number
    # were already identical between two such calls; only the raw mtime
    # float in rank_score differed. So this scans once and runs match,
    # group and rank twice against that one fixed, unchanging state.
    conn, lib, _ = _full_run(tmp_path)

    run_a = matchrun.run_match(conn)
    group.build_groups(conn, run_a)
    for row in conn.execute("SELECT id FROM dup_group WHERE run_id = ?",
                            (run_a,)):
        rank.rank_group(conn, row["id"])

    run_b = matchrun.run_match(conn)
    group.build_groups(conn, run_b)
    for row in conn.execute("SELECT id FROM dup_group WHERE run_id = ?",
                            (run_b,)):
        rank.rank_group(conn, row["id"])

    def normalize(data):
        for g in data["groups"]:
            g.pop("group_id", None)
            for m in g["members"]:
                m.pop("track_id", None)
        data.pop("run_id", None)
        return json.dumps(data, sort_keys=True)

    assert normalize(report.report_data(conn, run_a)) == \
           normalize(report.report_data(conn, run_b))
