"""Tests for library health report."""
from riffle import health, store


def _populated_db(tmp_path):
    """A DB with realistic data for health reporting — direct SQL inserts."""
    conn = store.connect(tmp_path / "db.sqlite")

    conn.execute(
        "INSERT INTO audio_content (id, audio_hash, hash_method, duration, "
        "codec, sample_rate, channels) VALUES (1, 'h1', 'streamhash', 210.5, "
        "'mp3', 44100, 2)")
    conn.execute(
        "INSERT INTO audio_content (id, audio_hash, hash_method, duration, "
        "codec, sample_rate, channels) VALUES (2, 'h2', 'streamhash', 180.0, "
        "'flac', 44100, 2)")
    conn.execute(
        "INSERT INTO audio_content (id, audio_hash, hash_method, duration, "
        "codec, sample_rate, channels) VALUES (3, 'h3', 'streamhash', 240.0, "
        "'mp3', 48000, 2)")

    conn.execute(
        "INSERT INTO track (id, path, size, mtime, audio_content_id, bitrate, "
        "tag_title, tag_artist, tag_album, tag_genre, present) "
        "VALUES (1, '/music/song1.mp3', 5000000, 1000.0, 1, 320000, "
        "'Song 1', 'Artist 1', 'Album 1', 'Rock', 1)")
    conn.execute(
        "INSERT INTO track (id, path, size, mtime, audio_content_id, bitrate, "
        "tag_title, tag_artist, tag_album, tag_genre, present) "
        "VALUES (2, '/music/song2.flac', 20000000, 1001.0, 2, 1000000, "
        "'Song 2', 'Artist 2', NULL, NULL, 1)")
    conn.execute(
        "INSERT INTO track (id, path, size, mtime, audio_content_id, bitrate, "
        "tag_title, tag_artist, tag_album, tag_genre, present) "
        "VALUES (3, '/music/song3.mp3', 6000000, 1002.0, 3, 256000, "
        "NULL, NULL, NULL, NULL, 1)")
    conn.execute(
        "INSERT INTO track (id, path, size, mtime, audio_content_id, bitrate, "
        "tag_title, tag_artist, tag_album, tag_genre, present) "
        "VALUES (4, '/music/gone.mp3', 4000000, 999.0, 1, 320000, "
        "'Gone', 'Artist 1', 'Album 1', 'Rock', 0)")

    conn.execute(
        "INSERT INTO ingest_error (path, stage, message, attempts) "
        "VALUES ('/music/broken.mp3', 'hash', 'corrupt stream', 1)")

    return conn


def test_health_counts_present_tracks(tmp_path):
    conn = _populated_db(tmp_path)
    data = health.health_report(conn)
    assert data["tracks"] == 3


def test_health_counts_unique_recordings(tmp_path):
    conn = _populated_db(tmp_path)
    data = health.health_report(conn)
    assert data["unique_recordings"] == 3


def test_health_codec_distribution(tmp_path):
    conn = _populated_db(tmp_path)
    data = health.health_report(conn)
    assert data["codecs"]["mp3"] == 2
    assert data["codecs"]["flac"] == 1


def test_health_format_distribution(tmp_path):
    conn = _populated_db(tmp_path)
    data = health.health_report(conn)
    assert data["formats"][".mp3"] == 2
    assert data["formats"][".flac"] == 1


def test_health_bitrate_stats(tmp_path):
    conn = _populated_db(tmp_path)
    data = health.health_report(conn)
    assert data["bitrate"]["min"] == 256000
    assert data["bitrate"]["max"] == 1000000


def test_health_tag_completeness(tmp_path):
    conn = _populated_db(tmp_path)
    data = health.health_report(conn)
    assert data["tags"]["fully_tagged"] == 1
    assert data["tags"]["no_tags"] == 1
    assert data["tags"]["missing_title"] == 1
    assert data["tags"]["missing_artist"] == 1
    assert data["tags"]["missing_album"] == 2
    assert data["tags"]["missing_genre"] == 2


def test_health_ingest_errors(tmp_path):
    conn = _populated_db(tmp_path)
    data = health.health_report(conn)
    assert data["errors"] == 1


def test_health_total_duration(tmp_path):
    conn = _populated_db(tmp_path)
    data = health.health_report(conn)
    assert abs(data["total_duration"] - 630.5) < 0.1


def test_health_total_size(tmp_path):
    conn = _populated_db(tmp_path)
    data = health.health_report(conn)
    assert data["total_size"] == 31000000


def test_health_sample_rates(tmp_path):
    conn = _populated_db(tmp_path)
    data = health.health_report(conn)
    assert data["sample_rates"][44100] == 2
    assert data["sample_rates"][48000] == 1


def test_health_empty_db(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    data = health.health_report(conn)
    assert data["tracks"] == 0
    assert data["unique_recordings"] == 0
    assert data["total_duration"] == 0.0
    assert data["total_size"] == 0


def test_render_health_produces_readable_text(tmp_path):
    conn = _populated_db(tmp_path)
    data = health.health_report(conn)
    text = health.render_health(data)
    assert "Library Health" in text
    assert "3" in text
    assert ".mp3" in text
    assert "Missing title" in text
