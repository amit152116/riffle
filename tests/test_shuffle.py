"""Tests for smart shuffle and track similarity lookup."""
import json
import numpy as np

from riffle import store


def _seed_library(conn, n=15, seed=42):
    """Create n tracks with features and a pre-built similarity index."""
    rng = np.random.RandomState(seed)
    for i in range(1, n + 1):
        conn.execute(
            "INSERT INTO audio_content (id, audio_hash, hash_method, duration) "
            "VALUES (?, ?, 'streamhash', 200.0)", (i, f"h{i}"))
        artist = f"Artist{chr(65 + (i % 4))}"
        conn.execute(
            "INSERT INTO track (id, path, size, mtime, audio_content_id, "
            "tag_artist, tag_title, tag_genre, present) "
            "VALUES (?, ?, 1000, 1.0, ?, ?, ?, ?, 1)",
            (i, f"/music/{artist}/track{i}.mp3", i, artist, f"Song {i}",
             "Pop" if i % 2 == 0 else "Rock"))
        mfcc = store.pack_mfcc(rng.randn(13))
        conn.execute(
            "INSERT INTO audio_features (audio_content_id, bpm, bpm_confidence, "
            "key_name, scale, key_strength, loudness_lufs, danceability, energy, "
            "spectral_centroid, onset_rate, dynamic_complexity, dissonance, zcr, "
            "mfcc_mean, extractor_version, config_hash, analyzed_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (i, 80.0 + rng.rand() * 100, 0.9,
             ["C", "G", "D", "A", "E"][i % 5], "major", 0.8,
             -14.0, 0.5, 0.3 + rng.rand() * 0.5,
             2000.0, 3.0, 5.0, 0.3, 0.05, mfcc,
             "2.1b6", "abc", "2026-01-01"))

    from riffle import similarity
    similarity.build_similarity(conn, top_k=10)


def test_find_similar_returns_ordered(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_library(conn)
    from riffle import shuffle
    rows = shuffle.find_similar(conn, 1, n=5)
    assert len(rows) == 5
    scores = [r["combined_score"] for r in rows]
    assert scores == sorted(scores)


def test_resolve_track_substring(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_library(conn)
    from riffle import shuffle
    tid = shuffle.resolve_track(conn, "track1.mp3")
    assert tid == 1


def test_resolve_track_ambiguous(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_library(conn)
    from riffle import shuffle
    import pytest
    with pytest.raises(ValueError, match="[Mm]ultiple|[Aa]mbiguous"):
        shuffle.resolve_track(conn, "track")


def test_resolve_track_not_found(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_library(conn)
    from riffle import shuffle
    import pytest
    with pytest.raises(ValueError, match="[Nn]o track|[Nn]ot found"):
        shuffle.resolve_track(conn, "nonexistent_xyz")


def test_smart_shuffle_excludes_same_audio_content_siblings(tmp_path):
    """Review finding I4: two present tracks sharing one audio_content_id
    (byte-identical files, before dedup approval) must never both appear,
    even without an approved dup_group."""
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_library(conn, n=15)
    # track 16 is a second path pointing at the SAME audio_content_id as track 1
    conn.execute(
        "INSERT INTO track (id, path, size, mtime, audio_content_id, "
        "tag_artist, tag_title, tag_genre, present) "
        "VALUES (16, '/music/other/copy_of_1.mp3', 1000, 1.0, 1, 'ArtistA', "
        "'Song 1 copy', 'Rock', 1)")
    from riffle import similarity
    similarity.build_similarity(conn, top_k=10)
    from riffle import shuffle
    playlist = shuffle.smart_shuffle(conn, seed_track_id=1, n=15)
    track_ids = [p["track_id"] for p in playlist]
    assert 1 in track_ids
    assert 16 not in track_ids


def test_smart_shuffle_excludes_duplicates(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_library(conn, n=15)
    conn.execute(
        "INSERT INTO match_run (id, created_at, status) VALUES (1, '2026-01-01', 'complete')")
    conn.execute(
        "INSERT INTO dup_group (id, run_id, tier, decision) VALUES (1, 1, 0, 'approved')")
    conn.execute("INSERT INTO group_content (group_id, audio_content_id) VALUES (1, 1)")
    conn.execute("INSERT INTO group_content (group_id, audio_content_id) VALUES (1, 2)")
    from riffle import shuffle
    playlist = shuffle.smart_shuffle(conn, seed_track_id=1, n=10)
    track_ids = [p["track_id"] for p in playlist]
    assert 1 in track_ids
    assert 2 not in track_ids


def test_smart_shuffle_respects_genre_filter(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_library(conn, n=15)
    from riffle import shuffle
    playlist = shuffle.smart_shuffle(conn, n=5, genre="Pop")
    for p in playlist:
        assert "Pop" in (p.get("tag_genre") or "")


def test_smart_shuffle_artist_diversity(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_library(conn, n=15)
    from riffle import shuffle
    playlist = shuffle.smart_shuffle(conn, n=10)
    back_to_back = 0
    for i in range(1, len(playlist)):
        if playlist[i]["tag_artist"] == playlist[i - 1]["tag_artist"]:
            back_to_back += 1
    assert back_to_back <= len(playlist) // 3


def test_smart_shuffle_partial_when_few_tracks(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_library(conn, n=5)
    from riffle import shuffle
    playlist = shuffle.smart_shuffle(conn, n=20)
    assert 1 <= len(playlist) <= 5


def test_cli_similar_ambiguous_track_reports_cleanly(tmp_path):
    from typer.testing import CliRunner
    from riffle.cli import app
    from riffle import store as store_mod

    conn = store_mod.connect(tmp_path / "db.sqlite")
    _seed_library(conn, n=15)
    conn.close()

    runner = CliRunner()
    result = runner.invoke(app, ["--db", str(tmp_path / "db.sqlite"), "similar", "track"])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "ambiguous" in result.output.lower() or "multiple" in result.output.lower()


def test_cli_shuffle_seed_not_found_reports_cleanly(tmp_path):
    from typer.testing import CliRunner
    from riffle.cli import app
    from riffle import store as store_mod

    conn = store_mod.connect(tmp_path / "db.sqlite")
    _seed_library(conn, n=15)
    conn.close()

    runner = CliRunner()
    result = runner.invoke(
        app, ["--db", str(tmp_path / "db.sqlite"), "shuffle", "--seed", "nonexistent_xyz"])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "no track" in result.output.lower() or "not found" in result.output.lower()
