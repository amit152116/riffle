"""Tests for audio quality analysis."""
from pathlib import Path

from riffle import quality, store
from tests.fixtures import make_tone, make_silence


def _db_with_track(conn, tmp_path, audio_path, cid=1):
    """Insert a track and audio_content row for a real audio file."""
    st = audio_path.stat()
    conn.execute(
        "INSERT INTO audio_content (id, audio_hash, hash_method, duration, "
        "codec, sample_rate, channels) "
        "VALUES (?, ?, 'streamhash', 10.0, 'flac', 44100, 2)",
        (cid, f"h{cid}"))
    conn.execute(
        "INSERT INTO track (id, path, size, mtime, audio_content_id, "
        "bitrate, present) VALUES (?, ?, ?, ?, ?, 1000000, 1)",
        (cid, str(audio_path), st.st_size, st.st_mtime, cid))
    return cid


def test_analyze_normal_tone(tmp_path):
    p = make_tone(tmp_path / "normal.flac", seconds=5.0, volume=0.5)
    result = quality.analyze_file(p)
    assert result["peak_db"] < 0.0
    assert result["mean_db"] < 0.0
    assert not result["clipping"]
    assert result["silence_sections"] == 0


def test_analyze_clipping_detection(tmp_path):
    p = make_tone(tmp_path / "loud.flac", seconds=5.0, volume=2.0)
    result = quality.analyze_file(p)
    assert result["clipping"]


def test_analyze_silence_detection(tmp_path):
    sil = make_silence(tmp_path / "sil.flac", seconds=8.0)
    result = quality.analyze_file(sil)
    assert result["silence_sections"] >= 1


def test_analyze_low_volume(tmp_path):
    p = make_tone(tmp_path / "quiet.flac", seconds=5.0, volume=0.01)
    result = quality.analyze_file(p)
    assert result["mean_db"] < -30.0
    assert result["low_volume"]


def test_quality_scan_stores_results(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    p = make_tone(tmp_path / "song.flac", seconds=5.0, volume=0.5)
    _db_with_track(conn, tmp_path, p)
    result = quality.quality_scan(conn)
    assert result["analyzed"] == 1
    assert result["issues"] == 0


def test_quality_scan_flags_low_volume(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    p = make_tone(tmp_path / "quiet.flac", seconds=5.0, volume=0.01)
    _db_with_track(conn, tmp_path, p)
    result = quality.quality_scan(conn)
    assert result["analyzed"] == 1
    assert result["issues"] >= 1


def test_quality_scan_skips_missing_tracks(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    conn.execute(
        "INSERT INTO audio_content (id, audio_hash, hash_method) "
        "VALUES (1, 'h1', 'streamhash')")
    conn.execute(
        "INSERT INTO track (id, path, size, mtime, audio_content_id, present) "
        "VALUES (1, '/nonexistent/file.mp3', 100, 1.0, 1, 1)")
    result = quality.quality_scan(conn)
    assert result["analyzed"] == 0
    assert result["failed"] == 1


def test_quality_scan_is_idempotent(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    p = make_tone(tmp_path / "song.flac", seconds=5.0, volume=0.5)
    _db_with_track(conn, tmp_path, p)
    quality.quality_scan(conn)
    result = quality.quality_scan(conn)
    assert result["analyzed"] == 0
    assert result["cached"] == 1


def test_render_quality_produces_text(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    p = make_tone(tmp_path / "song.flac", seconds=5.0, volume=0.5)
    _db_with_track(conn, tmp_path, p)
    quality.quality_scan(conn)
    text = quality.render_quality(conn)
    assert "Quality" in text
