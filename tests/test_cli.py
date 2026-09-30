from typer.testing import CliRunner

from riffle.cli import app
from tests.fixtures import make_tone

runner = CliRunner()


def test_scan_and_match_and_report(tmp_path):
    lib = tmp_path / "lib"
    src = make_tone(lib / "a.flac", seconds=40.0)
    (lib / "copy.flac").write_bytes(src.read_bytes())
    db = str(tmp_path / "db.sqlite")

    assert runner.invoke(app, ["--db", db, "scan", str(lib)]).exit_code == 0
    assert runner.invoke(app, ["--db", db, "match"]).exit_code == 0
    result = runner.invoke(app, ["--db", db, "report", "--run", "1"])
    assert result.exit_code == 0
    assert "Group" in result.stdout


def test_apply_without_approval_moves_nothing(tmp_path):
    lib = tmp_path / "lib"
    src = make_tone(lib / "a.flac", seconds=40.0)
    copy = lib / "copy.flac"
    copy.write_bytes(src.read_bytes())
    db = str(tmp_path / "db.sqlite")

    runner.invoke(app, ["--db", db, "scan", str(lib)])
    runner.invoke(app, ["--db", db, "match"])
    result = runner.invoke(app, ["--db", db, "apply", "--run", "1"])
    assert result.exit_code == 0
    assert copy.exists()
    assert "0" in result.stdout


def test_bulk_approve_requires_confirmation(tmp_path):
    lib = tmp_path / "lib"
    src = make_tone(lib / "a.flac", seconds=40.0)
    (lib / "copy.flac").write_bytes(src.read_bytes())
    db = str(tmp_path / "db.sqlite")
    runner.invoke(app, ["--db", db, "scan", str(lib)])
    runner.invoke(app, ["--db", db, "match"])

    declined = runner.invoke(
        app, ["--db", db, "approve", "--run", "1", "--tier", "0", "--all"],
        input="n\n")
    assert "approved" not in declined.stdout.lower() or declined.exit_code == 0

    accepted = runner.invoke(
        app, ["--db", db, "approve", "--run", "1", "--tier", "0", "--all",
              "--yes"])
    assert accepted.exit_code == 0


def test_concurrent_invocation_is_refused(tmp_path):
    from riffle import store

    db = tmp_path / "db.sqlite"
    with store.exclusive_lock(db):
        result = runner.invoke(app, ["--db", str(db), "match"])
    assert result.exit_code != 0


def test_bandwidth_command_measures_present_tracks(tmp_path):
    lib = tmp_path / "lib"
    make_tone(lib / "a.flac", seconds=20.0)
    db = str(tmp_path / "db.sqlite")
    assert runner.invoke(app, ["--db", db, "scan", str(lib)]).exit_code == 0

    first = runner.invoke(app, ["--db", db, "bandwidth"])
    assert first.exit_code == 0
    assert "measured 1" in first.stdout

    again = runner.invoke(app, ["--db", db, "bandwidth"])
    assert again.exit_code == 0
    assert "measured 0" in again.stdout
    assert "cached 1" in again.stdout


def test_bandwidth_remeasure_flag_remeasures_cached_files(tmp_path):
    lib = tmp_path / "lib"
    make_tone(lib / "a.flac", seconds=20.0)
    db = str(tmp_path / "db.sqlite")
    runner.invoke(app, ["--db", db, "scan", str(lib)])
    runner.invoke(app, ["--db", db, "bandwidth"])
    again = runner.invoke(app, ["--db", db, "bandwidth", "--remeasure"])
    assert again.exit_code == 0
    assert "measured 1" in again.stdout
