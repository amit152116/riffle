import os

from audiolib import scan, store
from tests.fixtures import make_tone, retag


def _db(tmp_path):
    return store.connect(tmp_path / "db.sqlite")


def test_scan_records_tracks_and_content(tmp_path):
    lib = tmp_path / "lib"
    make_tone(lib / "a.flac", seconds=3.0, freq=440)
    make_tone(lib / "b.flac", seconds=3.0, freq=1200)
    conn = _db(tmp_path)
    scan.scan(conn, [lib])
    assert conn.execute("SELECT count(*) c FROM track").fetchone()["c"] == 2
    assert conn.execute(
        "SELECT count(*) c FROM audio_content").fetchone()["c"] == 2


def test_identical_audio_shares_one_content_row(tmp_path):
    lib = tmp_path / "lib"
    a = make_tone(lib / "a.flac", seconds=3.0)
    b = lib / "b.flac"
    b.write_bytes(a.read_bytes())
    retag(b, title="Other")
    conn = _db(tmp_path)
    scan.scan(conn, [lib])
    assert conn.execute("SELECT count(*) c FROM track").fetchone()["c"] == 2
    assert conn.execute(
        "SELECT count(*) c FROM audio_content").fetchone()["c"] == 1


def test_rescan_is_incremental(tmp_path):
    lib = tmp_path / "lib"
    make_tone(lib / "a.flac", seconds=3.0)
    conn = _db(tmp_path)
    scan.scan(conn, [lib])
    first = conn.execute("SELECT scanned_at FROM track").fetchone()["scanned_at"]
    scan.scan(conn, [lib])
    # Unchanged file keeps its content row; no duplicate track row appears.
    assert conn.execute("SELECT count(*) c FROM track").fetchone()["c"] == 1
    assert first is not None


def test_hardlinks_are_recorded_once(tmp_path):
    lib = tmp_path / "lib"
    a = make_tone(lib / "a.flac", seconds=3.0)
    os.link(a, lib / "b.flac")
    conn = _db(tmp_path)
    scan.scan(conn, [lib])
    assert conn.execute("SELECT count(*) c FROM track").fetchone()["c"] == 1


def test_missing_file_is_marked_absent(tmp_path):
    lib = tmp_path / "lib"
    a = make_tone(lib / "a.flac", seconds=3.0)
    make_tone(lib / "b.flac", seconds=3.0, freq=1200)
    conn = _db(tmp_path)
    scan.scan(conn, [lib])
    a.unlink()
    scan.scan(conn, [lib])
    row = conn.execute(
        "SELECT present, absent_reason FROM track WHERE path LIKE '%a.flac'"
    ).fetchone()
    assert row["present"] == 0
    assert row["absent_reason"] == "missing"


def test_scanning_one_root_does_not_mark_another_absent(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    make_tone(one / "a.flac", seconds=3.0)
    make_tone(two / "b.flac", seconds=3.0, freq=1200)
    conn = _db(tmp_path)
    scan.scan(conn, [one, two])
    scan.scan(conn, [one])
    row = conn.execute(
        "SELECT present FROM track WHERE path LIKE '%b.flac'").fetchone()
    assert row["present"] == 1


def test_quarantine_directory_is_skipped(tmp_path):
    lib = tmp_path / "lib"
    make_tone(lib / "a.flac", seconds=3.0)
    make_tone(lib / scan.QUARANTINE_DIRNAME / "old.flac", seconds=3.0, freq=900)
    conn = _db(tmp_path)
    scan.scan(conn, [lib])
    assert conn.execute("SELECT count(*) c FROM track").fetchone()["c"] == 1


def test_symlink_loop_terminates(tmp_path):
    # Review Focus 3: a symlink pointing at an ancestor must not loop forever.
    lib = tmp_path / "lib"
    make_tone(lib / "a.flac", seconds=2.0)
    (lib / "sub").mkdir()
    os.symlink(lib, lib / "sub" / "back")
    found = list(scan.walk_audio_files([lib]))
    assert len(found) == 1


def test_symlink_to_scanned_file_is_not_a_second_track(tmp_path):
    lib = tmp_path / "lib"
    a = make_tone(lib / "a.flac", seconds=2.0)
    os.symlink(a, lib / "link.flac")
    conn = _db(tmp_path)
    scan.scan(conn, [lib])
    assert conn.execute("SELECT count(*) c FROM track").fetchone()["c"] == 1


def test_unreadable_file_is_recorded_as_an_error(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "broken.flac").write_bytes(b"nope")
    conn = _db(tmp_path)
    scan.scan(conn, [lib])
    row = conn.execute("SELECT stage, attempts FROM ingest_error").fetchone()
    assert row["stage"] == "hash"
    assert row["attempts"] == 1


def test_unchanged_error_is_not_retried(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "broken.flac").write_bytes(b"nope")
    conn = _db(tmp_path)
    scan.scan(conn, [lib])
    scan.scan(conn, [lib])
    assert conn.execute(
        "SELECT attempts FROM ingest_error").fetchone()["attempts"] == 1


def test_changed_error_is_retried(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    broken = lib / "broken.flac"
    broken.write_bytes(b"nope")
    conn = _db(tmp_path)
    scan.scan(conn, [lib])
    broken.write_bytes(b"still not audio but different")
    scan.scan(conn, [lib])
    assert conn.execute(
        "SELECT attempts FROM ingest_error").fetchone()["attempts"] == 2


def test_retry_errors_flag_forces_retry(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "broken.flac").write_bytes(b"nope")
    conn = _db(tmp_path)
    scan.scan(conn, [lib])
    scan.scan(conn, [lib], retry_errors=True)
    assert conn.execute(
        "SELECT attempts FROM ingest_error").fetchone()["attempts"] == 2
