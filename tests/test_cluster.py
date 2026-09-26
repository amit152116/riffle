"""Tests for k-means clustering."""
import json

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


def test_cluster_n_clusters_exceeds_tracks_raises_clear_error(tmp_path):
    """M7: `riffle cluster --n 20` on a library with only 16 tracks (with
    features) must raise OUR OWN clear error before ever reaching sklearn,
    not sklearn's raw 'n_samples=16 should be >= n_clusters=20' message."""
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_features(conn, n=16)
    from riffle import cluster
    import pytest
    with pytest.raises(ValueError, match="only 16 tracks"):
        cluster.cluster_tracks(conn, n_clusters=20)


def test_cli_cluster_bad_n_reports_cleanly(tmp_path):
    from typer.testing import CliRunner
    from riffle.cli import app

    db = str(tmp_path / "db.sqlite")
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_features(conn, n=16)
    conn.close()

    runner = CliRunner()
    result = runner.invoke(app, ["--db", db, "cluster", "--n", "20"])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "only 16 tracks" in result.output


def test_cluster_excludes_quarantined_content(tmp_path):
    """M4: a track that's been quarantined (present=0) must not be pulled
    into clustering just because its audio_features row still exists."""
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_features(conn, n=20)
    conn.execute("UPDATE track SET present = 0, absent_reason = 'quarantined' WHERE id = 1")
    from riffle import cluster
    result = cluster.cluster_tracks(conn, n_clusters=3)
    no_assignment = conn.execute(
        "SELECT count(*) c FROM cluster_assignment WHERE audio_content_id = 1"
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


def test_cluster_grows_past_single_cluster_threshold(tmp_path):
    """Review finding I3: a library that starts under 15 tracks (single
    cluster 'All') and later grows past 15 must re-cluster properly on the
    next plain `cluster_tracks()` call, not stay stuck at n_clusters=1."""
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_features(conn, n=10)
    from riffle import cluster
    first = cluster.cluster_tracks(conn)
    assert first["n_clusters"] == 1

    rng = np.random.RandomState(7)
    for i in range(11, 21):
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
            (i, 80.0 + rng.rand() * 100, 0.9, "C", "major", 0.8,
             -20.0 + rng.rand() * 10, rng.rand(), rng.rand(),
             1000 + rng.rand() * 5000, rng.rand() * 10, 5.0, 0.3, 0.05, mfcc,
             "2.1b6", "abc", "2026-01-01"))

    second = cluster.cluster_tracks(conn, n_clusters=3)
    assert second["n_clusters"] == 3
    sizes = conn.execute(
        "SELECT count(DISTINCT cluster_id) c FROM cluster_assignment "
        "WHERE run_id = ?", (second["run_id"],)
    ).fetchone()["c"]
    assert sizes == 3


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


def test_library_terciles_uses_percentiles(tmp_path):
    """Library-relative calibration: the BPM/energy label cutoffs should be
    computed from this library's own distribution, not a fixed number."""
    conn = store.connect(tmp_path / "db.sqlite")
    for i, energy in enumerate([0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24,
                                0.26, 0.28], start=1):
        conn.execute(
            "INSERT INTO audio_content (id, audio_hash, hash_method) "
            "VALUES (?, ?, 'streamhash')", (i, f"h{i}"))
        conn.execute(
            "INSERT INTO track (id, path, size, mtime, audio_content_id, present) "
            "VALUES (?, ?, 1000, 1.0, ?, 1)", (i, f"/music/t{i}.mp3", i))
        conn.execute(
            "INSERT INTO audio_features (audio_content_id, energy, bpm) "
            "VALUES (?, ?, 120.0)", (i, energy))
    from riffle import cluster
    lo, hi = cluster._library_terciles(conn, "energy", fallback=(0.3, 0.7))
    # this whole library sits under the old fixed 0.3 cutoff -- the
    # library-relative terciles must NOT be (0.3, 0.7), they must reflect
    # the actual 0.10-0.28 spread
    assert 0.15 <= lo <= 0.19
    assert 0.21 <= hi <= 0.25


def test_library_terciles_falls_back_with_too_few_tracks(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    for i, energy in enumerate([0.1, 0.2], start=1):
        conn.execute(
            "INSERT INTO audio_content (id, audio_hash, hash_method) "
            "VALUES (?, ?, 'streamhash')", (i, f"h{i}"))
        conn.execute(
            "INSERT INTO track (id, path, size, mtime, audio_content_id, present) "
            "VALUES (?, ?, 1000, 1.0, ?, 1)", (i, f"/music/t{i}.mp3", i))
        conn.execute(
            "INSERT INTO audio_features (audio_content_id, energy, bpm) "
            "VALUES (?, ?, 120.0)", (i, energy))
    from riffle import cluster
    lo, hi = cluster._library_terciles(conn, "energy", fallback=(0.3, 0.7))
    assert (lo, hi) == (0.3, 0.7)


def test_label_cluster_energy_is_library_relative(tmp_path):
    """Review real-music finding: a library where every track's RMS energy
    sits under 0.3 (typical for mastered music) must still get differentiated
    Calm/Moderate/Energetic labels, not have every cluster read 'Calm'."""
    conn = store.connect(tmp_path / "db.sqlite")
    rng = np.random.RandomState(1)
    # cluster 0: low energy within this library's own range (0.10-0.14)
    for i in range(1, 6):
        conn.execute(
            "INSERT INTO audio_content (id, audio_hash, hash_method) "
            "VALUES (?, ?, 'streamhash')", (i, f"h{i}"))
        conn.execute(
            "INSERT INTO track (id, path, size, mtime, audio_content_id, present) "
            "VALUES (?, ?, 1000, 1.0, ?, 1)", (i, f"/music/t{i}.mp3", i))
        mfcc = store.pack_mfcc(rng.randn(13))
        conn.execute(
            "INSERT INTO audio_features (audio_content_id, bpm, energy, mfcc_mean) "
            "VALUES (?, 120.0, ?, ?)", (i, 0.10 + i * 0.01, mfcc))
    # cluster 1: highest energy within this library's own range (0.26-0.30)
    for i in range(6, 11):
        conn.execute(
            "INSERT INTO audio_content (id, audio_hash, hash_method) "
            "VALUES (?, ?, 'streamhash')", (i, f"h{i}"))
        conn.execute(
            "INSERT INTO track (id, path, size, mtime, audio_content_id, present) "
            "VALUES (?, ?, 1000, 1.0, ?, 1)", (i, f"/music/t{i}.mp3", i))
        mfcc = store.pack_mfcc(rng.randn(13))
        conn.execute(
            "INSERT INTO audio_features (audio_content_id, bpm, energy, mfcc_mean) "
            "VALUES (?, 120.0, ?, ?)", (i, 0.26 + (i - 6) * 0.01, mfcc))

    from riffle import cluster
    conn.execute(
        "INSERT INTO cluster_run (id, n_clusters, n_tracks, created_at) "
        "VALUES (1, 2, 10, '2026-01-01')")
    for i in range(1, 6):
        conn.execute(
            "INSERT INTO cluster_assignment (run_id, audio_content_id, cluster_id) "
            "VALUES (1, ?, 0)", (i,))
    for i in range(6, 11):
        conn.execute(
            "INSERT INTO cluster_assignment (run_id, audio_content_id, cluster_id) "
            "VALUES (1, ?, 1)", (i,))

    label_low = cluster.label_cluster(conn, 0, 1)
    label_high = cluster.label_cluster(conn, 1, 1)
    # with a fixed 0.3 cutoff BOTH would read "Calm" -- library-relative
    # terciles must differentiate them
    assert label_low != label_high


def test_cluster_drift_detection(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_features(conn, n=20)
    from riffle import cluster
    cluster.cluster_tracks(conn, n_clusters=3)
    n_at_build = conn.execute(
        "SELECT n_tracks FROM cluster_run ORDER BY id DESC LIMIT 1"
    ).fetchone()["n_tracks"]
    assert n_at_build == 20


def test_rebuild_clusters_is_atomic_on_failure(tmp_path, monkeypatch):
    """Review finding I5: rebuild_clusters does a full re-cluster plus a
    two-pass ID relabel as many separate autocommit statements. If something
    raises partway through, a crash/interrupt must not leave the database
    half-migrated (e.g. sentinel cluster_id values like -1003, or a new
    cluster_run with no matching centroids)."""
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_features(conn, n=30)
    from riffle import cluster
    first = cluster.cluster_tracks(conn, n_clusters=3)
    runs_before = conn.execute("SELECT count(*) c FROM cluster_run").fetchone()["c"]
    centroids_before = conn.execute("SELECT count(*) c FROM cluster_centroid").fetchone()["c"]
    assignments_before = conn.execute("SELECT count(*) c FROM cluster_assignment").fetchone()["c"]

    real_label_cluster = cluster.label_cluster
    call_count = {"n": 0}

    def _boom(conn, cluster_id, run_id):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated crash mid-relabel")
        return real_label_cluster(conn, cluster_id, run_id)

    monkeypatch.setattr(cluster, "label_cluster", _boom)
    import pytest
    with pytest.raises(RuntimeError, match="simulated crash"):
        cluster.rebuild_clusters(conn, n_clusters=3)

    runs_after = conn.execute("SELECT count(*) c FROM cluster_run").fetchone()["c"]
    centroids_after = conn.execute("SELECT count(*) c FROM cluster_centroid").fetchone()["c"]
    assignments_after = conn.execute("SELECT count(*) c FROM cluster_assignment").fetchone()["c"]
    sentinel_rows = conn.execute(
        "SELECT count(*) c FROM cluster_centroid WHERE cluster_id <= -1000"
    ).fetchone()["c"]

    assert runs_after == runs_before
    assert centroids_after == centroids_before
    assert assignments_after == assignments_before
    assert sentinel_rows == 0


def test_render_clusters(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_features(conn, n=20)
    from riffle import cluster
    cluster.cluster_tracks(conn, n_clusters=3)
    text = cluster.render_clusters(conn)
    assert "Cluster" in text


def test_cli_cluster_as_json_emits_valid_json(tmp_path):
    """M6: --as-json must emit machine-readable JSON, not just suppress the
    human-readable report."""
    from typer.testing import CliRunner
    from riffle.cli import app

    db = str(tmp_path / "db.sqlite")
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_features(conn, n=20)
    conn.close()

    runner = CliRunner()
    result = runner.invoke(app, ["--db", db, "cluster", "--n", "3", "--as-json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["n_clusters"] == 3


def test_render_clusters_excludes_quarantined_examples(tmp_path):
    """M4: a track quarantined AFTER its cluster_run assignment must not
    show up as an example track in the rendered summary."""
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_features(conn, n=20)
    from riffle import cluster
    cluster.cluster_tracks(conn, n_clusters=3)
    conn.execute("UPDATE track SET tag_artist = 'Unmistakable Artist' WHERE id = 1")
    conn.execute("UPDATE track SET present = 0, absent_reason = 'quarantined' WHERE id = 1")
    text = cluster.render_clusters(conn)
    assert "Unmistakable Artist" not in text
