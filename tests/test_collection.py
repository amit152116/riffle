"""Tests for collection browsing and statistics."""
import json
import numpy as np

from riffle import store


def _seed_db(conn, n=5):
    """Insert n tracks with audio_features."""
    for i in range(1, n + 1):
        conn.execute(
            "INSERT INTO audio_content (id, audio_hash, hash_method, duration) "
            "VALUES (?, ?, 'streamhash', 200.0)", (i, f"h{i}"))
        conn.execute(
            "INSERT INTO track (id, path, size, mtime, audio_content_id, "
            "tag_artist, tag_title, tag_genre, present) "
            "VALUES (?, ?, 1000, 1.0, ?, ?, ?, ?, 1)",
            (i, f"/music/track{i}.mp3", i,
             f"Artist{chr(65 + (i % 3))}", f"Song {i}",
             "Pop" if i % 2 == 0 else "Rock"))
        mfcc = store.pack_mfcc(np.random.randn(13))
        conn.execute(
            "INSERT INTO audio_features (audio_content_id, bpm, bpm_confidence, "
            "key_name, scale, key_strength, loudness_lufs, danceability, energy, "
            "spectral_centroid, onset_rate, dynamic_complexity, dissonance, zcr, "
            "mfcc_mean, extractor_version, config_hash, analyzed_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (i, 80.0 + i * 20, 0.9, "C" if i % 2 == 0 else "A", "major", 0.8,
             -14.0 + i, 0.5 + i * 0.05, 0.3 + i * 0.1,
             2000.0 + i * 500, 3.0, 5.0, 0.3, 0.05, mfcc,
             "2.1b6", "abc123", "2026-01-01T00:00:00"))


def test_browse_default_sort(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_db(conn)
    from riffle import collection
    rows = collection.browse(conn)
    assert len(rows) == 5
    artists = [r["tag_artist"] for r in rows]
    assert artists == sorted(artists)


def test_browse_filter_bpm_range(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_db(conn)
    from riffle import collection
    rows = collection.browse(conn, bpm_range=(100, 140))
    for r in rows:
        assert 100 <= r["bpm"] < 140


def test_browse_filter_genre(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_db(conn)
    from riffle import collection
    rows = collection.browse(conn, genre="Pop")
    assert len(rows) > 0
    for r in rows:
        assert "Pop" in r["tag_genre"]


def test_browse_limit_offset(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_db(conn)
    from riffle import collection
    page1 = collection.browse(conn, limit=2, offset=0)
    page2 = collection.browse(conn, limit=2, offset=2)
    assert len(page1) == 2
    assert len(page2) == 2
    assert page1[0]["tag_title"] != page2[0]["tag_title"]


def test_stats_bpm_histogram(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_db(conn)
    from riffle import collection
    data = collection.stats(conn)
    assert data["total_tracks"] == 5
    assert data["total_with_features"] == 5
    assert len(data["bpm_histogram"]) > 0
    total_in_buckets = sum(b["count"] for b in data["bpm_histogram"])
    assert total_in_buckets == 5


def test_stats_total_with_metadata_excludes_no_match_sentinels(tmp_path):
    """M7 consumer fix: a musicbrainz_match row for a no_match outcome
    (recording_mbid IS NULL) must not count as 'has metadata'."""
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_db(conn, n=5)
    conn.execute(
        "INSERT INTO musicbrainz_match (audio_content_id, parsed_at) "
        "VALUES (1, '2026-01-01')")
    from riffle import collection
    data = collection.stats(conn)
    assert data["total_with_metadata"] == 0


def test_render_browse(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_db(conn)
    from riffle import collection
    rows = collection.browse(conn, limit=3)
    text = collection.render_browse(rows)
    assert "Artist" in text
    assert "BPM" in text


def test_browse_filter_cluster_returns_all_members(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_db(conn, n=6)
    conn.execute(
        "INSERT INTO cluster_run (id, n_clusters, n_tracks, created_at) "
        "VALUES (1, 2, 6, '2026-01-01')")
    for cid in (1, 2, 3):
        conn.execute(
            "INSERT INTO cluster_assignment (run_id, audio_content_id, cluster_id) "
            "VALUES (1, ?, 0)", (cid,))
    for cid in (4, 5, 6):
        conn.execute(
            "INSERT INTO cluster_assignment (run_id, audio_content_id, cluster_id) "
            "VALUES (1, ?, 1)", (cid,))
    from riffle import collection
    rows = collection.browse(conn, cluster=0, limit=50)
    assert len(rows) == 3


def test_render_stats(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_db(conn)
    from riffle import collection
    data = collection.stats(conn)
    text = collection.render_stats(data)
    assert "Collection" in text
