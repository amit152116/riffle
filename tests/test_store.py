import sqlite3

import numpy as np
import pytest

from riffle import store


def test_connect_creates_schema(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    names = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"audio_content", "fingerprint", "track", "scan_run",
            "ingest_error", "match_run", "run_track", "pair",
            "dup_group", "group_content", "group_member",
            "acoustid_cache", "quarantine_log",
            "schema_version"} <= names


def test_connect_is_idempotent(tmp_path):
    p = tmp_path / "db.sqlite"
    store.connect(p).close()
    conn = store.connect(p)
    v = conn.execute("SELECT version FROM schema_version").fetchone()["version"]
    assert v == store.SCHEMA_VERSION


def test_wal_enabled(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_hash_method_and_hash_together_are_unique(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    conn.execute("INSERT INTO audio_content (audio_hash, hash_method) "
                 "VALUES ('abc', 'streamhash')")
    # The same hex under the other method is a different identity, not a clash.
    conn.execute("INSERT INTO audio_content (audio_hash, hash_method) "
                 "VALUES ('abc', 'whole_file')")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO audio_content (audio_hash, hash_method) "
                     "VALUES ('abc', 'streamhash')")


def test_pair_rejects_unordered_ids(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    conn.execute("INSERT INTO match_run (id, status) VALUES (1, 'running')")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO pair (run_id, a_content_id, b_content_id) "
            "VALUES (1, 5, 2)")


def test_fingerprint_roundtrip():
    arr = np.array([1, 2, 4294967295, 0], dtype=np.uint32)
    blob = store.pack_fingerprint(arr)
    assert len(blob) == 4 * len(arr)
    back = store.unpack_fingerprint(blob, len(arr))
    assert back.dtype == np.uint32
    assert np.array_equal(back, arr)


def test_unpack_rejects_length_mismatch():
    blob = store.pack_fingerprint(np.array([1, 2], dtype=np.uint32))
    with pytest.raises(ValueError):
        store.unpack_fingerprint(blob, 3)


def test_exclusive_lock_blocks_second_holder(tmp_path):
    p = tmp_path / "db.sqlite"
    with store.exclusive_lock(p):
        with pytest.raises(store.LockError):
            with store.exclusive_lock(p):
                pass


def test_migration_3_creates_audio_features(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    assert "audio_features" in tables
    cols = [r[1] for r in conn.execute("PRAGMA table_info(audio_features)")]
    assert "mfcc_mean" in cols
    assert "config_hash" in cols


def test_mfcc_pack_unpack_roundtrip():
    original = np.random.randn(13).astype(np.float64)
    blob = store.pack_mfcc(original)
    assert len(blob) == 13 * 8
    recovered = store.unpack_mfcc(blob)
    np.testing.assert_array_almost_equal(original, recovered)


def test_migration_4_creates_musicbrainz_match(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    assert "musicbrainz_match" in tables
    cols = [r[1] for r in conn.execute("PRAGMA table_info(musicbrainz_match)")]
    assert "artists_json" in cols
    assert "recording_mbid" in cols


def test_migration_4_adds_audio_content_id_to_cache(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    cols = [r[1] for r in conn.execute("PRAGMA table_info(acoustid_cache)")]
    assert "audio_content_id" in cols


def test_migration_5_creates_cluster_and_similarity_tables(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    assert "cluster_run" in tables
    assert "cluster_centroid" in tables
    assert "cluster_assignment" in tables
    assert "track_similarity" in tables
    sim_cols = [r[1] for r in conn.execute("PRAGMA table_info(track_similarity)")]
    assert "mfcc_norm" in sim_cols
    assert "combined_score" in sim_cols


def test_migration_6_makes_track_similarity_directed(tmp_path):
    """track_similarity was originally undirected (track_a_id < track_b_id,
    used to store one row per pair). Top-K is conceptually directed per
    track, so Migration 6 recreates it as (track_id, neighbor_id) with each
    track owning its own top-K rows, independent of any other track's."""
    conn = store.connect(tmp_path / "db.sqlite")
    cols = [r[1] for r in conn.execute("PRAGMA table_info(track_similarity)")]
    assert "track_id" in cols
    assert "neighbor_id" in cols
    assert "track_a_id" not in cols
    assert "config_hash" in cols
    for tid in (1, 2, 3):
        conn.execute(
            "INSERT INTO track (id, path, size, mtime, present) "
            "VALUES (?, ?, 100, 1.0, 1)", (tid, f"/music/t{tid}.mp3"))
    # a track can own top-K rows independent of its neighbor's own rows
    conn.execute(
        "INSERT INTO track_similarity (track_id, neighbor_id, mfcc_norm, "
        "combined_score) VALUES (1, 2, 0.1, 0.1)")
    conn.execute(
        "INSERT INTO track_similarity (track_id, neighbor_id, mfcc_norm, "
        "combined_score) VALUES (2, 1, 0.1, 0.1)")
    conn.execute(
        "INSERT INTO track_similarity (track_id, neighbor_id, mfcc_norm, "
        "combined_score) VALUES (2, 3, 0.2, 0.2)")
    rows = conn.execute(
        "SELECT count(*) c FROM track_similarity WHERE track_id = 2"
    ).fetchone()["c"]
    assert rows == 2


def test_migration_7_runs_as_a_callable(tmp_path):
    """Migration 7 must be a callable, not a SQL string. executescript()
    cannot safely run inside the transaction this migration needs --
    verified empirically: it force-commits any pending transaction before
    running its own statements, with no transaction of its own afterward."""
    assert callable(store._MIGRATIONS[6])


def test_migration_7_bumps_schema_version(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    assert conn.execute(
        "SELECT version FROM schema_version"
    ).fetchone()["version"] == 7


def test_migration_7_rolls_back_atomically_on_failure(tmp_path, monkeypatch):
    """If any step inside _migration_7 raises, the whole migration --
    including the schema_version bump -- must roll back together. A retry
    must see the exact same pre-migration state, not a half-applied one."""
    def _boom(conn):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(store, "_backfill_mb_artists", _boom)

    db_path = tmp_path / "db.sqlite"
    with pytest.raises(RuntimeError, match="simulated failure"):
        store.connect(db_path)

    # A raw connection (not store.connect(), which would retry migration 7
    # and hit the same monkeypatched failure again).
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    version = conn.execute("SELECT version FROM schema_version").fetchone()["version"]
    assert version == 6
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "mb_artist" not in tables


def _connect_at_v6(db_path):
    """A raw v6 database, without running Migration 7 -- for testing
    Migration 7 against realistic pre-existing data, not just an empty
    freshly-created schema."""
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    for sql in store._MIGRATIONS[:6]:
        conn.executescript(sql)
    conn.execute("INSERT INTO schema_version (version) VALUES (6)")
    return conn


def test_migration_7_creates_mb_artist_tables(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "mb_artist" in tables
    assert "mb_recording_artist" in tables


def test_migration_7_backfills_artists_from_existing_json(tmp_path):
    db_path = tmp_path / "db.sqlite"
    conn = _connect_at_v6(db_path)
    conn.execute(
        "INSERT INTO audio_content (id, audio_hash, hash_method) "
        "VALUES (1, 'h1', 'streamhash')")
    conn.execute(
        "INSERT INTO musicbrainz_match (audio_content_id, artists_json) "
        "VALUES (1, ?)",
        ('[{"name": "A.R. Rahman", "mbid": "mbid-1"}, '
         '{"name": "Chinmayi", "mbid": "mbid-2"}]',))
    conn.close()

    conn = store.connect(db_path)
    artists = {r["mbid"]: r["name"] for r in conn.execute("SELECT * FROM mb_artist")}
    assert artists == {"mbid-1": "A.R. Rahman", "mbid-2": "Chinmayi"}
    links = conn.execute(
        "SELECT position, artist_mbid FROM mb_recording_artist ORDER BY position"
    ).fetchall()
    assert [(r["position"], r["artist_mbid"]) for r in links] == [
        (0, "mbid-1"), (1, "mbid-2")]
