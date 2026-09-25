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
