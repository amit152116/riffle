from typer.testing import CliRunner

from audiolib.cli import app
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
    from audiolib import store

    db = tmp_path / "db.sqlite"
    with store.exclusive_lock(db):
        result = runner.invoke(app, ["--db", str(db), "match"])
    assert result.exit_code != 0
