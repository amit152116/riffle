"""Review list of likely same-recording pairs that fall short of tier 1.

Tier 1 needs 85% coverage on *both* files, so a copy with a long intro or
outro lands in tier 2 with the studio version. Most tier-2 pairs are not
that: 636 of 698 in one real library were a single song inside an hour-long
compilation, which must never be treated as a duplicate. So this is a report
for a person to review; it never creates a group or authorizes quarantine.
"""
import json

from typer.testing import CliRunner

from riffle import report, store
from riffle.cli import app


def _conn(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    conn.execute("INSERT INTO match_run (id, status) VALUES (1, 'complete')")
    return conn


def _pair(conn, a, b, cov_a, cov_b, err=2.0, span=200.0, tier=2):
    for cid in (a, b):
        conn.execute("INSERT OR IGNORE INTO audio_content "
                     "(id, audio_hash, hash_method) VALUES (?,?, 'streamhash')",
                     (cid, f"h{cid}"))
        conn.execute("INSERT OR IGNORE INTO track (id, path, audio_content_id, "
                     "present) VALUES (?,?,?,1)", (cid, f"/lib/{cid}.mp3", cid))
    conn.execute(
        "INSERT INTO pair (run_id, a_content_id, b_content_id, coverage_a, "
        " coverage_b, mean_bit_error, matched_span_seconds, tier, "
        " verified_direct) VALUES (1,?,?,?,?,?,?,?,0)",
        (a, b, cov_a, cov_b, err, span, tier))


def test_lists_a_tier_2_pair_covered_60_to_85_percent_on_both_sides(tmp_path):
    conn = _conn(tmp_path)
    _pair(conn, 1, 2, 0.70, 0.90)
    rows = report.probable_pairs(conn, 1)
    assert [(r["a_content_id"], r["b_content_id"]) for r in rows] == [(1, 2)]
    assert rows[0]["a_paths"] == ["/lib/1.mp3"]
    assert rows[0]["b_paths"] == ["/lib/2.mp3"]


def test_a_song_inside_a_compilation_is_not_listed(tmp_path):
    conn = _conn(tmp_path)
    _pair(conn, 1, 2, 0.98, 0.05)  # one side barely covered
    assert report.probable_pairs(conn, 1) == []


def test_poor_alignment_or_a_short_overlap_is_not_listed(tmp_path):
    conn = _conn(tmp_path)
    _pair(conn, 1, 2, 0.8, 0.8, err=4.5)
    _pair(conn, 3, 4, 0.8, 0.8, span=60.0)
    assert report.probable_pairs(conn, 1) == []


def test_thresholds_are_inclusive(tmp_path):
    conn = _conn(tmp_path)
    _pair(conn, 1, 2, 0.60, 0.60, err=4.0, span=90.0)
    assert len(report.probable_pairs(conn, 1)) == 1


def test_tier_1_pairs_are_already_grouped_so_not_listed(tmp_path):
    conn = _conn(tmp_path)
    _pair(conn, 1, 2, 0.95, 0.95, tier=1)
    assert report.probable_pairs(conn, 1) == []


def test_only_the_requested_run_is_listed(tmp_path):
    conn = _conn(tmp_path)
    conn.execute("INSERT INTO match_run (id, status) VALUES (2, 'complete')")
    _pair(conn, 1, 2, 0.7, 0.7)
    conn.execute("UPDATE pair SET run_id = 2")
    assert report.probable_pairs(conn, 1) == []
    assert len(report.probable_pairs(conn, 2)) == 1


def test_a_quarantined_copy_is_shown_at_its_quarantine_path(tmp_path):
    conn = _conn(tmp_path)
    _pair(conn, 1, 2, 0.7, 0.7)
    conn.execute("INSERT INTO run_track (run_id, track_id, path, size, mtime, "
                 "audio_hash, hash_method) VALUES (1,1,'/lib/1.mp3',1,1,'h1',"
                 "'streamhash')")
    conn.execute("INSERT INTO dup_group (id, run_id, tier) VALUES (1,1,1)")
    conn.execute("INSERT INTO quarantine_log (run_id, track_id, group_id, "
                 "src_path, dst_path, state) VALUES "
                 "(1,1,1,'/lib/1.mp3','/lib/.riffle-quarantine/1.mp3','moved')")
    row = report.probable_pairs(conn, 1)[0]
    assert row["a_paths"] == ["/lib/.riffle-quarantine/1.mp3"]
    assert row["a_quarantined"] is True
    assert row["b_quarantined"] is False


def test_render_text_shows_coverage_and_paths(tmp_path):
    conn = _conn(tmp_path)
    _pair(conn, 1, 2, 0.70, 0.90, err=1.5, span=180.0)
    text = report.render_probable(report.probable_pairs(conn, 1))
    assert "/lib/1.mp3" in text and "/lib/2.mp3" in text
    assert "0.70" in text and "0.90" in text
    assert "review only" in text.lower()


def test_render_text_with_nothing_to_review():
    assert "No probable duplicates" in report.render_probable([])


def test_probable_command_prints_json(tmp_path):
    conn = _conn(tmp_path)
    _pair(conn, 1, 2, 0.7, 0.9)
    conn.close()
    result = CliRunner().invoke(
        app, ["--db", str(tmp_path / "db.sqlite"), "probable",
              "--run", "1", "--as-json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data[0]["a_content_id"] == 1


def _run_probable(tmp_path, *extra):
    result = CliRunner().invoke(
        app, ["--db", str(tmp_path / "db.sqlite"), "probable", "--run", "1",
              "--as-json", *extra])
    assert result.exit_code == 0, result.stdout
    return json.loads(result.stdout)


def test_min_coverage_option_looks_further_down(tmp_path):
    conn = _conn(tmp_path)
    _pair(conn, 1, 2, 0.50, 0.55)
    conn.close()
    assert _run_probable(tmp_path) == []
    listed = _run_probable(tmp_path, "--min-coverage", "0.4")
    assert [(r["a_content_id"], r["b_content_id"]) for r in listed] == [(1, 2)]


def test_max_bit_error_option(tmp_path):
    conn = _conn(tmp_path)
    _pair(conn, 1, 2, 0.8, 0.8, err=5.0)
    conn.close()
    assert _run_probable(tmp_path) == []
    assert len(_run_probable(tmp_path, "--max-bit-error", "5.5")) == 1


def test_min_span_option(tmp_path):
    conn = _conn(tmp_path)
    _pair(conn, 1, 2, 0.8, 0.8, span=60.0)
    conn.close()
    assert _run_probable(tmp_path) == []
    assert len(_run_probable(tmp_path, "--min-span", "45")) == 1


def test_options_default_to_the_documented_thresholds(tmp_path):
    # A pair exactly on each default boundary is listed with no options.
    conn = _conn(tmp_path)
    _pair(conn, 1, 2, 0.60, 0.60, err=4.0, span=90.0)
    conn.close()
    assert len(_run_probable(tmp_path)) == 1
