"""Tests for similarity metrics and normalization."""
import json

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
    """Review Focus #3 / M2: tracks with NULL key or BPM don't crash scoring,
    and the missing component is OMITTED with remaining weights renormalized
    -- not treated as a literal 0 (maximum bpm penalty) or 'C major' (a
    fabricated key match). With identical MFCC and energy and only bpm/key
    missing, the score must be exactly 0.0, not some intermediate value."""
    from riffle import similarity
    fa = {"mfcc_mean": np.ones(13), "bpm": None, "key_name": None,
          "scale": None, "energy": 0.5}
    fb = {"mfcc_mean": np.ones(13), "bpm": 120.0, "key_name": "C",
          "scale": "major", "energy": 0.5}
    score = similarity.combined_score(fa, fb)
    assert score == 0.0


def test_combined_score_zero_bpm_treated_as_missing():
    """Essentia writes bpm=0.0 for beatless/silent tracks, not NULL. That
    must be treated as missing too, not as a literal maximum BPM penalty."""
    from riffle import similarity
    fa = {"mfcc_mean": np.ones(13), "bpm": 0.0, "key_name": "C",
          "scale": "major", "energy": 0.5}
    fb = {"mfcc_mean": np.ones(13), "bpm": 120.0, "key_name": "C",
          "scale": "major", "energy": 0.5}
    score = similarity.combined_score(fa, fb)
    assert score == 0.0


def test_score_components_reports_omitted_norms_as_none():
    from riffle import similarity
    fa = {"mfcc_mean": np.ones(13), "bpm": None, "key_name": None,
          "scale": None, "energy": 0.5}
    fb = {"mfcc_mean": np.ones(13), "bpm": 120.0, "key_name": "C",
          "scale": "major", "energy": 0.5}
    comp = similarity.score_components(fa, fb)
    assert comp["bpm_norm"] is None
    assert comp["key_norm"] is None
    assert comp["mfcc_norm"] is not None
    assert comp["energy_norm"] is not None
    assert comp["combined_score"] == 0.0


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
        "SELECT count(*) c FROM track_similarity WHERE track_id = 1"
    ).fetchone()["c"]
    assert rows == 5


def test_build_similarity_stores_config_hash(tmp_path):
    """Spec rule 5: similarity artifacts must be tied to a versioned
    similarity configuration, so a future weight change can be detected."""
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_tracks_with_features(conn, n=10)
    from riffle import similarity
    similarity.build_similarity(conn, top_k=5)
    row = conn.execute("SELECT config_hash FROM track_similarity LIMIT 1").fetchone()
    assert row["config_hash"] == similarity.similarity_config_hash()
    assert len(row["config_hash"]) == 64


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


def _add_track(conn, cid, seed, bpm=120.0, key="C"):
    rng = np.random.RandomState(seed)
    conn.execute(
        "INSERT INTO audio_content (id, audio_hash, hash_method, duration) "
        "VALUES (?, ?, 'streamhash', 200.0)", (cid, f"h{cid}"))
    conn.execute(
        "INSERT INTO track (id, path, size, mtime, audio_content_id, present) "
        "VALUES (?, ?, 1000, 1.0, ?, 1)", (cid, f"/music/t{cid}.mp3", cid))
    mfcc = store.pack_mfcc(rng.randn(13))
    conn.execute(
        "INSERT INTO audio_features (audio_content_id, bpm, bpm_confidence, "
        "key_name, scale, key_strength, loudness_lufs, danceability, energy, "
        "spectral_centroid, onset_rate, dynamic_complexity, dissonance, zcr, "
        "mfcc_mean, extractor_version, config_hash, analyzed_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (cid, bpm, 0.9, key, "major", 0.8, -14.0, 0.5, 0.5,
         3000.0, 5.0, 5.0, 0.3, 0.05, mfcc, "2.1b6", "abc", "2026-01-01"))


def test_build_similarity_fills_up_below_top_k(tmp_path):
    """M3 fill-up: a track indexed while the library was smaller than top_k
    must pick up newly added tracks until it reaches top_k neighbors, not
    stay capped at whatever was available on the first build."""
    conn = store.connect(tmp_path / "db.sqlite")
    for cid in (1, 2, 3):
        _add_track(conn, cid, seed=cid)
    from riffle import similarity
    similarity.build_similarity(conn, top_k=5)
    count_before = conn.execute(
        "SELECT count(*) c FROM track_similarity WHERE track_id = 1"
    ).fetchone()["c"]
    assert count_before == 2  # only 2 other tracks existed

    for cid in (4, 5, 6):
        _add_track(conn, cid, seed=cid + 100)
    similarity.build_similarity(conn, top_k=5)
    count_after = conn.execute(
        "SELECT count(*) c FROM track_similarity WHERE track_id = 1"
    ).fetchone()["c"]
    assert count_after == 5  # now fills up to top_k


def test_build_similarity_evicts_worst_on_better_match(tmp_path):
    """M3 eviction: when a new track scores better than an existing track's
    current worst neighbor, and that existing track already has top_k rows,
    the old worst must be REMOVED, not just have a better one added on top
    (directed schema: this must not touch any OTHER track's own rows)."""
    conn = store.connect(tmp_path / "db.sqlite")
    for cid in range(1, 6):
        _add_track(conn, cid, seed=cid)
    from riffle import similarity
    similarity.build_similarity(conn, top_k=3)

    track1_before = {r["neighbor_id"] for r in conn.execute(
        "SELECT neighbor_id FROM track_similarity WHERE track_id = 1"
    ).fetchall()}
    assert len(track1_before) == 3

    other_track_rows_before = conn.execute(
        "SELECT count(*) c FROM track_similarity WHERE track_id = 2"
    ).fetchone()["c"]

    # add a track with mfcc identical to track 1 -- guaranteed best match
    conn.execute(
        "INSERT INTO audio_content (id, audio_hash, hash_method, duration) "
        "VALUES (99, 'h99', 'streamhash', 200.0)")
    conn.execute(
        "INSERT INTO track (id, path, size, mtime, audio_content_id, present) "
        "VALUES (99, '/music/t99.mp3', 1000, 1.0, 99, 1)")
    track1_mfcc = store.unpack_mfcc(conn.execute(
        "SELECT mfcc_mean FROM audio_features af "
        "JOIN track t ON t.audio_content_id = af.audio_content_id "
        "WHERE t.id = 1"
    ).fetchone()["mfcc_mean"])
    conn.execute(
        "INSERT INTO audio_features (audio_content_id, bpm, bpm_confidence, "
        "key_name, scale, key_strength, loudness_lufs, danceability, energy, "
        "spectral_centroid, onset_rate, dynamic_complexity, dissonance, zcr, "
        "mfcc_mean, extractor_version, config_hash, analyzed_at) "
        "VALUES (99,120.0,0.9,'C','major',0.8,-14.0,0.5,0.5,"
        "3000.0,5.0,5.0,0.3,0.05,?,'2.1b6','abc','2026-01-01')",
        (store.pack_mfcc(track1_mfcc),))

    similarity.build_similarity(conn, top_k=3)

    track1_after = {r["neighbor_id"] for r in conn.execute(
        "SELECT neighbor_id FROM track_similarity WHERE track_id = 1"
    ).fetchall()}
    assert len(track1_after) == 3  # still capped at top_k, not 4
    assert 99 in track1_after  # the perfect match got in

    other_track_rows_after = conn.execute(
        "SELECT count(*) c FROM track_similarity WHERE track_id = 2"
    ).fetchone()["c"]
    assert other_track_rows_after == other_track_rows_before  # untouched


def test_full_rebuild(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_tracks_with_features(conn, n=10)
    from riffle import similarity
    similarity.build_similarity(conn, top_k=5)
    similarity.full_rebuild(conn, top_k=5)
    rows = conn.execute("SELECT count(*) c FROM track_similarity").fetchone()["c"]
    assert rows > 0


def test_full_rebuild_is_atomic_on_failure(tmp_path, monkeypatch):
    """Review finding I5: full_rebuild deletes the whole index then rebuilds
    it as separate autocommit statements. If build_similarity raises partway
    through, a crash/interrupt must not leave the index empty or half-built."""
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_tracks_with_features(conn, n=10)
    from riffle import similarity
    similarity.build_similarity(conn, top_k=5)
    rows_before = conn.execute("SELECT count(*) c FROM track_similarity").fetchone()["c"]
    assert rows_before > 0

    def _boom(conn, top_k=20):
        raise RuntimeError("simulated crash mid-rebuild")

    monkeypatch.setattr(similarity, "build_similarity", _boom)
    import pytest
    with pytest.raises(RuntimeError, match="simulated crash"):
        similarity.full_rebuild(conn, top_k=5)

    rows_after = conn.execute("SELECT count(*) c FROM track_similarity").fetchone()["c"]
    assert rows_after == rows_before


def test_cli_build_index_as_json_emits_valid_json(tmp_path):
    """M6: --as-json must emit machine-readable JSON, not just suppress the
    human-readable report."""
    from typer.testing import CliRunner
    from riffle.cli import app

    db = str(tmp_path / "db.sqlite")
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_tracks_with_features(conn, n=10)
    conn.close()

    runner = CliRunner()
    result = runner.invoke(app, ["--db", db, "build-index", "--as-json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["new_tracks"] == 10
