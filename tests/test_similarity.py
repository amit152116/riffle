"""Tests for similarity metrics and normalization."""
import numpy as np


def test_mfcc_cosine_distance_identical():
    from riffle import similarity
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
    assert similarity.mfcc_cosine_distance(a, a) == 0.0


def test_mfcc_cosine_distance_orthogonal():
    from riffle import similarity
    a = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert similarity.mfcc_cosine_distance(a, b) == 1.0


def test_key_distance_same():
    from riffle import similarity
    assert similarity.key_distance("C", "major", "C", "major") == 0


def test_key_distance_fifth():
    from riffle import similarity
    assert similarity.key_distance("C", "major", "G", "major") == 1


def test_key_distance_tritone():
    from riffle import similarity
    assert similarity.key_distance("C", "major", "F#", "major") == 6


def test_key_distance_relative_minor():
    from riffle import similarity
    assert similarity.key_distance("C", "major", "A", "minor") == 0


def test_normalize_components_ranges():
    from riffle import similarity
    result = similarity.normalize_components(
        mfcc_dist=1.5, bpm_diff=30.0, key_dist=3, energy_diff=0.5)
    assert 0.0 <= result["mfcc"] <= 1.0
    assert 0.0 <= result["bpm"] <= 1.0
    assert 0.0 <= result["key"] <= 1.0
    assert 0.0 <= result["energy"] <= 1.0


def test_normalize_components_max_values():
    from riffle import similarity
    result = similarity.normalize_components(
        mfcc_dist=2.0, bpm_diff=100.0, key_dist=6, energy_diff=1.0)
    assert result["mfcc"] == 1.0
    assert result["bpm"] == 1.0
    assert result["key"] == 1.0
    assert result["energy"] == 1.0


def test_normalize_components_zero_values():
    from riffle import similarity
    result = similarity.normalize_components(
        mfcc_dist=0.0, bpm_diff=0.0, key_dist=0, energy_diff=0.0)
    assert result["mfcc"] == 0.0
    assert result["bpm"] == 0.0
    assert result["key"] == 0.0
    assert result["energy"] == 0.0


def test_combined_score_uses_normalized():
    from riffle import similarity
    fa = {"mfcc_mean": np.ones(13), "bpm": 120.0, "key_name": "C",
          "scale": "major", "energy": 0.5}
    fb = {"mfcc_mean": np.ones(13), "bpm": 120.0, "key_name": "C",
          "scale": "major", "energy": 0.5}
    score = similarity.combined_score(fa, fb)
    assert score == 0.0


def test_combined_score_weights():
    from riffle import similarity
    fa = {"mfcc_mean": np.ones(13), "bpm": 120.0, "key_name": "C",
          "scale": "major", "energy": 0.5}
    fb = {"mfcc_mean": -np.ones(13), "bpm": 120.0, "key_name": "C",
          "scale": "major", "energy": 0.5}
    score = similarity.combined_score(fa, fb)
    assert score > 0.0
    assert score <= 1.0


def test_combined_score_null_key_bpm():
    """Review Focus #3: tracks with NULL key or BPM don't crash scoring."""
    from riffle import similarity
    fa = {"mfcc_mean": np.ones(13), "bpm": None, "key_name": None,
          "scale": None, "energy": 0.5}
    fb = {"mfcc_mean": np.ones(13), "bpm": 120.0, "key_name": "C",
          "scale": "major", "energy": 0.5}
    score = similarity.combined_score(fa, fb)
    assert 0.0 <= score <= 1.0


from riffle import store


def _seed_tracks_with_features(conn, n=10, seed=42):
    rng = np.random.RandomState(seed)
    for i in range(1, n + 1):
        conn.execute(
            "INSERT INTO audio_content (id, audio_hash, hash_method, duration) "
            "VALUES (?, ?, 'streamhash', 200.0)", (i, f"h{i}"))
        conn.execute(
            "INSERT INTO track (id, path, size, mtime, audio_content_id, "
            "tag_artist, present) VALUES (?, ?, 1000, 1.0, ?, ?, 1)",
            (i, f"/music/t{i}.mp3", i, f"Artist{i % 3}"))
        mfcc = store.pack_mfcc(rng.randn(13))
        conn.execute(
            "INSERT INTO audio_features (audio_content_id, bpm, bpm_confidence, "
            "key_name, scale, key_strength, loudness_lufs, danceability, energy, "
            "spectral_centroid, onset_rate, dynamic_complexity, dissonance, zcr, "
            "mfcc_mean, extractor_version, config_hash, analyzed_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (i, 80.0 + rng.rand() * 100, 0.9, "C", "major", 0.8,
             -14.0, 0.5, 0.3 + rng.rand() * 0.5,
             2000.0, 3.0, 5.0, 0.3, 0.05, mfcc,
             "2.1b6", "abc", "2026-01-01"))


def test_build_similarity_stores_top_k(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_tracks_with_features(conn, n=10)
    from riffle import similarity
    result = similarity.build_similarity(conn, top_k=5)
    assert result["new_tracks"] == 10
    assert result["pairs_stored"] > 0
    rows = conn.execute(
        "SELECT count(*) c FROM track_similarity WHERE track_a_id = 1 OR track_b_id = 1"
    ).fetchone()["c"]
    assert rows >= 5


def test_build_similarity_uses_combined_score(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_tracks_with_features(conn, n=10)
    from riffle import similarity
    similarity.build_similarity(conn, top_k=5)
    row = conn.execute(
        "SELECT * FROM track_similarity LIMIT 1"
    ).fetchone()
    assert row["combined_score"] is not None
    assert row["mfcc_norm"] is not None
    assert 0.0 <= row["mfcc_norm"] <= 1.0
    assert 0.0 <= row["combined_score"] <= 1.0


def test_build_similarity_incremental(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_tracks_with_features(conn, n=5)
    from riffle import similarity
    similarity.build_similarity(conn, top_k=3)
    count_before = conn.execute("SELECT count(*) c FROM track_similarity").fetchone()["c"]

    rng = np.random.RandomState(99)
    for i in range(6, 8):
        conn.execute(
            "INSERT INTO audio_content (id, audio_hash, hash_method, duration) "
            "VALUES (?, ?, 'streamhash', 200.0)", (i, f"h{i}"))
        conn.execute(
            "INSERT INTO track (id, path, size, mtime, audio_content_id, present) "
            "VALUES (?, ?, 1000, 1.0, ?, 1)", (i, f"/music/t{i}.mp3", i))
        mfcc = store.pack_mfcc(rng.randn(13))
        conn.execute(
            "INSERT INTO audio_features (audio_content_id, bpm, bpm_confidence, "
            "key_name, scale, key_strength, loudness_lufs, danceability, energy, "
            "spectral_centroid, onset_rate, dynamic_complexity, dissonance, zcr, "
            "mfcc_mean, extractor_version, config_hash, analyzed_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (i, 120.0, 0.9, "G", "major", 0.8, -14.0, 0.6, 0.5,
             3000.0, 5.0, 5.0, 0.3, 0.05, mfcc, "2.1b6", "abc", "2026-01-01"))

    result = similarity.build_similarity(conn, top_k=3)
    assert result["new_tracks"] == 2
    count_after = conn.execute("SELECT count(*) c FROM track_similarity").fetchone()["c"]
    assert count_after > count_before


def test_full_rebuild(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_tracks_with_features(conn, n=10)
    from riffle import similarity
    similarity.build_similarity(conn, top_k=5)
    similarity.full_rebuild(conn, top_k=5)
    rows = conn.execute("SELECT count(*) c FROM track_similarity").fetchone()["c"]
    assert rows > 0
