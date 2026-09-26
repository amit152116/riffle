"""Tests for k-means clustering."""
import numpy as np

from riffle import store


FEATURE_COLS = ["bpm", "energy", "danceability", "loudness_lufs",
                "spectral_centroid", "onset_rate"]


def _seed_features(conn, n=20, seed=42):
    rng = np.random.RandomState(seed)
    for i in range(1, n + 1):
        conn.execute(
            "INSERT INTO audio_content (id, audio_hash, hash_method, duration) "
            "VALUES (?, ?, 'streamhash', 200.0)", (i, f"h{i}"))
        conn.execute(
            "INSERT INTO track (id, path, size, mtime, audio_content_id, "
            "tag_artist, present) VALUES (?, ?, 1000, 1.0, ?, ?, 1)",
            (i, f"/music/t{i}.mp3", i, f"Artist{i}"))
        mfcc = store.pack_mfcc(rng.randn(13))
        conn.execute(
            "INSERT INTO audio_features (audio_content_id, bpm, bpm_confidence, "
            "key_name, scale, key_strength, loudness_lufs, danceability, energy, "
            "spectral_centroid, onset_rate, dynamic_complexity, dissonance, zcr, "
            "mfcc_mean, extractor_version, config_hash, analyzed_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (i, 80.0 + rng.rand() * 100, 0.9, "C", "major", 0.8,
             -20.0 + rng.rand() * 10, rng.rand(), rng.rand(),
             1000 + rng.rand() * 5000, rng.rand() * 10, 5.0, 0.3, 0.05, mfcc,
             "2.1b6", "abc", "2026-01-01"))


def test_cluster_assigns_all_tracks(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_features(conn, n=30)
    from riffle import cluster
    result = cluster.cluster_tracks(conn, n_clusters=3)
    assert result["n_clusters"] == 3
    assigned = conn.execute("SELECT count(*) c FROM cluster_assignment").fetchone()["c"]
    assert assigned == 30


def test_cluster_is_idempotent(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_features(conn, n=20)
    from riffle import cluster
    cluster.cluster_tracks(conn, n_clusters=3)
    rng = np.random.RandomState(99)
    for i in range(21, 23):
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
            (i, 120.0, 0.9, "C", "major", 0.8, -15.0, 0.6, 0.5,
             3000.0, 5.0, 5.0, 0.3, 0.05, mfcc, "2.1b6", "abc", "2026-01-01"))
    result = cluster.cluster_tracks(conn, n_clusters=3)
    assert result["n_clusters"] == 3
    assigned = conn.execute("SELECT count(*) c FROM cluster_assignment").fetchone()["c"]
    assert assigned == 22


def test_cluster_small_library(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_features(conn, n=10)
    from riffle import cluster
    result = cluster.cluster_tracks(conn)
    assert result["n_clusters"] == 1
    assigned = conn.execute(
        "SELECT count(*) c FROM cluster_assignment WHERE cluster_id = 0"
    ).fetchone()["c"]
    assert assigned == 10


def test_cluster_missing_features_skipped(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_features(conn, n=20)
    conn.execute(
        "INSERT INTO audio_content (id, audio_hash, hash_method) "
        "VALUES (99, 'h99', 'streamhash')")
    conn.execute(
        "INSERT INTO track (id, path, size, mtime, audio_content_id, present) "
        "VALUES (99, '/music/nofeatures.mp3', 100, 1.0, 99, 1)")
    from riffle import cluster
    result = cluster.cluster_tracks(conn, n_clusters=3)
    assert result["n_clusters"] == 3
    no_assignment = conn.execute(
        "SELECT count(*) c FROM cluster_assignment WHERE audio_content_id = 99"
    ).fetchone()["c"]
    assert no_assignment == 0


def test_cluster_scaler_stored(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_features(conn, n=20)
    from riffle import cluster
    cluster.cluster_tracks(conn, n_clusters=3)
    row = conn.execute(
        "SELECT scaler_params FROM cluster_run ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row["scaler_params"] is not None
    assert len(row["scaler_params"]) > 0


def test_cluster_identical_features(tmp_path):
    """Review Focus #5: all tracks have identical features — degenerate case."""
    conn = store.connect(tmp_path / "db.sqlite")
    for i in range(1, 21):
        conn.execute(
            "INSERT INTO audio_content (id, audio_hash, hash_method, duration) "
            "VALUES (?, ?, 'streamhash', 200.0)", (i, f"h{i}"))
        conn.execute(
            "INSERT INTO track (id, path, size, mtime, audio_content_id, present) "
            "VALUES (?, ?, 1000, 1.0, ?, 1)", (i, f"/music/t{i}.mp3", i))
        mfcc = store.pack_mfcc(np.zeros(13))
        conn.execute(
            "INSERT INTO audio_features (audio_content_id, bpm, bpm_confidence, "
            "key_name, scale, key_strength, loudness_lufs, danceability, energy, "
            "spectral_centroid, onset_rate, dynamic_complexity, dissonance, zcr, "
            "mfcc_mean, extractor_version, config_hash, analyzed_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (i, 120.0, 0.9, "C", "major", 0.8, -14.0, 0.5, 0.5,
             3000.0, 5.0, 5.0, 0.3, 0.05, mfcc, "2.1b6", "abc", "2026-01-01"))
    from riffle import cluster
    result = cluster.cluster_tracks(conn, n_clusters=3)
    assert result["n_clusters"] in (1, 3)
    assigned = conn.execute("SELECT count(*) c FROM cluster_assignment").fetchone()["c"]
    assert assigned == 20


def test_cluster_rebuild_stabilizes_ids(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_features(conn, n=30)
    from riffle import cluster
    result1 = cluster.cluster_tracks(conn, n_clusters=3)
    run1 = result1["run_id"]
    assignments1 = {r[0]: r[1] for r in conn.execute(
        "SELECT audio_content_id, cluster_id FROM cluster_assignment WHERE run_id = ?",
        (run1,)
    ).fetchall()}

    result2 = cluster.rebuild_clusters(conn, n_clusters=3)
    run2 = result2["run_id"]
    assert run2 != run1
    assignments2 = {r[0]: r[1] for r in conn.execute(
        "SELECT audio_content_id, cluster_id FROM cluster_assignment WHERE run_id = ?",
        (run2,)
    ).fetchall()}

    same = sum(1 for cid in assignments1 if assignments1[cid] == assignments2.get(cid))
    assert same >= len(assignments1) * 0.7


def test_cluster_label_from_tracks(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_features(conn, n=20)
    from riffle import cluster
    result = cluster.cluster_tracks(conn, n_clusters=3)
    run_id = result["run_id"]
    label = cluster.label_cluster(conn, 0, run_id)
    assert isinstance(label, str)
    assert len(label) > 0
    valid_words = ["Slow", "Mid-tempo", "Upbeat", "Calm", "Moderate", "Energetic",
                   "Major", "Minor", "Bright", "Warm"]
    assert any(w in label for w in valid_words)


def test_cluster_drift_detection(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_features(conn, n=20)
    from riffle import cluster
    cluster.cluster_tracks(conn, n_clusters=3)
    n_at_build = conn.execute(
        "SELECT n_tracks FROM cluster_run ORDER BY id DESC LIMIT 1"
    ).fetchone()["n_tracks"]
    assert n_at_build == 20


def test_render_clusters(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_features(conn, n=20)
    from riffle import cluster
    cluster.cluster_tracks(conn, n_clusters=3)
    text = cluster.render_clusters(conn)
    assert "Cluster" in text
