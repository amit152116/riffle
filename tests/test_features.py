"""Tests for audio feature extraction."""
import json
from pathlib import Path

import numpy as np

from tests.fixtures import make_tone, make_silence


def test_extract_file_normal_tone(tmp_path):
    p = make_tone(tmp_path / "song.flac", seconds=10.0, volume=0.5)
    from riffle import features
    result = features.extract_file(p)
    assert isinstance(result["bpm"], float)
    assert result["bpm"] > 0
    assert result["bpm_confidence"] >= 0.0
    assert result["key_name"] in [
        "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"
    ]
    assert result["scale"] in ("major", "minor")
    assert result["key_strength"] >= 0.0
    assert isinstance(result["loudness_lufs"], float)
    assert result["danceability"] >= 0.0  # Essentia's DFA-based value ranges 0 to ~3, not [0,1]
    assert 0.0 <= result["energy"] <= 1.0
    assert isinstance(result["spectral_centroid"], float)
    assert isinstance(result["onset_rate"], float)
    assert isinstance(result["dynamic_complexity"], float)
    assert isinstance(result["dissonance"], float)
    assert isinstance(result["zcr"], float)
    mfcc = result["mfcc_mean"]
    assert isinstance(mfcc, np.ndarray)
    assert mfcc.shape == (13,)
    assert result["extractor_version"] is not None
    assert result["config_hash"] is not None


def test_extract_file_silence(tmp_path):
    p = make_silence(tmp_path / "silent.flac", seconds=10.0)
    from riffle import features
    result = features.extract_file(p)
    assert result["energy"] < 0.01
    assert result["loudness_lufs"] < -40.0


def test_bpm_octave_correction(tmp_path):
    from riffle import features
    assert features._correct_bpm(45.0) == 90.0
    assert features._correct_bpm(220.0) == 110.0
    assert features._correct_bpm(120.0) == 120.0


def test_energy_is_rms_not_mean_square():
    """Review finding I1: spec says 'RMS power normalized to 0-1', but the
    code was computing mean-square (no sqrt), which understates energy for
    any real track and makes the Calm/Energetic cluster labels meaningless."""
    from riffle import features
    n_samples = 1000
    raw_energy = n_samples * 0.25  # mean-square power of 0.25
    energy = features._compute_energy(raw_energy, n_samples)
    assert energy == 0.5  # sqrt(0.25), not 0.25 itself


def test_extraction_config_hash_deterministic():
    from riffle import features
    h1 = features.extraction_config_hash()
    h2 = features.extraction_config_hash()
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_essentia_import_error(monkeypatch):
    """Review Focus #1: clear message when essentia not installed."""
    import importlib
    import sys
    monkeypatch.setitem(sys.modules, "essentia", None)
    monkeypatch.setitem(sys.modules, "essentia.standard", None)
    if "riffle.features" in sys.modules:
        del sys.modules["riffle.features"]
    from riffle import features
    import pytest
    with pytest.raises((ImportError, ModuleNotFoundError)):
        features.extract_file("/dummy/path.flac")


from riffle import store


def _db_with_track(conn, tmp_path, audio_path, cid=1):
    st = audio_path.stat()
    conn.execute(
        "INSERT INTO audio_content (id, audio_hash, hash_method, duration, "
        "codec, sample_rate, channels) "
        "VALUES (?, ?, 'streamhash', 10.0, 'flac', 44100, 1)",
        (cid, f"h{cid}"))
    conn.execute(
        "INSERT INTO track (id, path, size, mtime, audio_content_id, "
        "bitrate, present) VALUES (?, ?, ?, ?, ?, 1000000, 1)",
        (cid, str(audio_path), st.st_size, st.st_mtime, cid))


def test_feature_scan_stores_results(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    p = make_tone(tmp_path / "song.flac", seconds=10.0, volume=0.5)
    _db_with_track(conn, tmp_path, p)
    from riffle import features
    result = features.feature_scan(conn)
    assert result["analyzed"] == 1
    assert result["failed"] == 0
    row = conn.execute("SELECT * FROM audio_features WHERE audio_content_id = 1").fetchone()
    assert row is not None
    assert row["bpm"] > 0
    assert row["config_hash"] is not None


def test_feature_scan_is_idempotent(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    p = make_tone(tmp_path / "song.flac", seconds=10.0, volume=0.5)
    _db_with_track(conn, tmp_path, p)
    from riffle import features
    features.feature_scan(conn)
    result = features.feature_scan(conn)
    assert result["analyzed"] == 0
    assert result["cached"] == 1


def test_feature_scan_skips_missing(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    conn.execute(
        "INSERT INTO audio_content (id, audio_hash, hash_method) "
        "VALUES (1, 'h1', 'streamhash')")
    conn.execute(
        "INSERT INTO track (id, path, size, mtime, audio_content_id, present) "
        "VALUES (1, '/nonexistent/file.mp3', 100, 1.0, 1, 1)")
    from riffle import features
    result = features.feature_scan(conn)
    assert result["analyzed"] == 0
    assert result["failed"] == 1


def test_feature_scan_limit(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    for i in range(3):
        p = make_tone(tmp_path / f"song{i}.flac", seconds=5.0, volume=0.5, freq=440 + i * 100)
        _db_with_track(conn, tmp_path, p, cid=i + 1)
    from riffle import features
    result = features.feature_scan(conn, limit=2)
    assert result["analyzed"] == 2


def test_render_features(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    p = make_tone(tmp_path / "song.flac", seconds=10.0, volume=0.5)
    _db_with_track(conn, tmp_path, p)
    from riffle import features
    features.feature_scan(conn)
    text = features.render_features(conn)
    assert "BPM" in text
    assert "Key" in text


def test_feature_scan_raises_clearly_when_essentia_missing(tmp_path, monkeypatch):
    """Review Focus #1 (re-check): a whole-scan-level ImportError must not be
    swallowed as a per-track 'failed' count -- it must surface to the caller."""
    conn = store.connect(tmp_path / "db.sqlite")
    p = make_tone(tmp_path / "song.flac", seconds=5.0, volume=0.5)
    _db_with_track(conn, tmp_path, p)
    from riffle import features

    def _boom():
        raise ImportError("Essentia is not installed. Install it with: pip install essentia")

    monkeypatch.setattr(features, "_check_essentia_available", _boom)
    import pytest
    with pytest.raises(ImportError, match="pip install essentia"):
        features.feature_scan(conn)


def test_cli_features_as_json_emits_valid_json(tmp_path):
    """M6: --as-json must emit machine-readable JSON, not just suppress the
    human-readable report."""
    from typer.testing import CliRunner
    from riffle.cli import app

    lib = tmp_path / "lib"
    p = make_tone(lib / "song.flac", seconds=5.0, volume=0.5)
    db = str(tmp_path / "db.sqlite")
    conn = store.connect(tmp_path / "db.sqlite")
    _db_with_track(conn, tmp_path, p)
    conn.close()

    runner = CliRunner()
    result = runner.invoke(app, ["--db", db, "features", "--as-json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["analyzed"] == 1


def test_cli_features_reports_missing_essentia_cleanly(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from riffle.cli import app
    from riffle import features as features_mod

    def _boom(conn, limit=None):
        raise ImportError("Essentia is not installed. Install it with: pip install essentia")

    monkeypatch.setattr(features_mod, "feature_scan", _boom)
    runner = CliRunner()
    db = str(tmp_path / "db.sqlite")
    from riffle import store
    store.connect(tmp_path / "db.sqlite")
    result = runner.invoke(app, ["--db", db, "features"])
    assert result.exit_code == 1
    assert "essentia" in result.output.lower()
    assert "Traceback" not in result.stdout
