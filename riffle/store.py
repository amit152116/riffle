"""SQLite schema, migrations, locking, and fingerprint serialization."""
from __future__ import annotations

import contextlib
import fcntl
import json
import sqlite3
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 7


class LockError(Exception):
    """Another riffle process holds the database lock."""


_MIGRATION_1 = """
CREATE TABLE schema_version (version INTEGER NOT NULL);

CREATE TABLE audio_content (
    id            INTEGER PRIMARY KEY,
    audio_hash    TEXT NOT NULL,
    hash_method   TEXT NOT NULL
                  CHECK (hash_method IN ('streamhash','whole_file')),
    duration      REAL,
    codec         TEXT,
    sample_rate   INTEGER,
    channels      INTEGER,
    first_seen_at TEXT,
    UNIQUE (hash_method, audio_hash)
);

CREATE TABLE fingerprint (
    id               INTEGER PRIMARY KEY,
    audio_content_id INTEGER NOT NULL REFERENCES audio_content(id),
    analyzer         TEXT NOT NULL,
    analyzer_version TEXT NOT NULL,
    config_hash      TEXT NOT NULL,
    purpose          TEXT NOT NULL
                     CHECK (purpose IN ('canonical','acoustid_lookup')),
    algorithm        INTEGER,
    fp_raw           BLOB NOT NULL,
    fp_length        INTEGER NOT NULL,
    computed_at      TEXT,
    UNIQUE (audio_content_id, analyzer, analyzer_version,
            config_hash, purpose, algorithm)
);

CREATE TABLE scan_run (
    id           INTEGER PRIMARY KEY,
    started_at   TEXT,
    completed_at TEXT,
    roots        TEXT,
    status       TEXT NOT NULL
                 CHECK (status IN ('running','complete','failed'))
);

CREATE TABLE track (
    id                INTEGER PRIMARY KEY,
    path              TEXT NOT NULL UNIQUE,
    size              INTEGER,
    mtime             REAL,
    dev               INTEGER,
    inode             INTEGER,
    nlink             INTEGER,
    audio_content_id  INTEGER REFERENCES audio_content(id),
    bitrate           INTEGER,
    tag_title         TEXT,
    tag_artist        TEXT,
    tag_album         TEXT,
    tag_genre         TEXT,
    tag_completeness  INTEGER,
    last_seen_scan_id INTEGER REFERENCES scan_run(id),
    present           INTEGER NOT NULL DEFAULT 1,
    absent_reason     TEXT CHECK (absent_reason IN ('missing','quarantined')),
    scanned_at        TEXT
);
CREATE INDEX track_content ON track(audio_content_id);
CREATE INDEX track_present ON track(present);
CREATE INDEX track_inode ON track(dev, inode);

CREATE TABLE ingest_error (
    path            TEXT PRIMARY KEY,
    dev             INTEGER,
    inode           INTEGER,
    size            INTEGER,
    mtime           REAL,
    stage           TEXT NOT NULL,
    message         TEXT,
    attempts        INTEGER NOT NULL DEFAULT 1,
    last_attempt_at TEXT,
    resolved_at     TEXT
);

CREATE TABLE match_run (
    id                 INTEGER PRIMARY KEY,
    created_at         TEXT,
    fingerprint_config TEXT,
    match_config       TEXT,
    software_version   TEXT,
    status             TEXT NOT NULL
                       CHECK (status IN ('running','complete','failed'))
);

CREATE TABLE run_track (
    run_id      INTEGER NOT NULL REFERENCES match_run(id),
    track_id    INTEGER NOT NULL,
    path        TEXT NOT NULL,
    size        INTEGER,
    mtime       REAL,
    audio_hash  TEXT,
    hash_method TEXT,
    PRIMARY KEY (run_id, track_id)
);

CREATE TABLE pair (
    run_id               INTEGER NOT NULL REFERENCES match_run(id),
    a_content_id         INTEGER NOT NULL,
    b_content_id         INTEGER NOT NULL,
    best_offset          INTEGER,
    peak_votes           INTEGER,
    peak_vote_ratio      REAL,
    matched_span_items   INTEGER,
    matched_span_seconds REAL,
    coverage_a           REAL,
    coverage_b           REAL,
    mean_bit_error       REAL,
    segment_count        INTEGER,
    tier                 INTEGER,
    verified_direct      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (run_id, a_content_id, b_content_id),
    CHECK (a_content_id < b_content_id)
);

CREATE TABLE dup_group (
    id              INTEGER PRIMARY KEY,
    run_id          INTEGER NOT NULL REFERENCES match_run(id),
    tier            INTEGER NOT NULL,
    formed_by_chain INTEGER NOT NULL DEFAULT 0,
    decision        TEXT NOT NULL DEFAULT 'proposed'
                    CHECK (decision IN
                           ('proposed','approved','rejected','applied')),
    decided_at      TEXT
);

CREATE TABLE group_content (
    group_id         INTEGER NOT NULL REFERENCES dup_group(id),
    audio_content_id INTEGER NOT NULL REFERENCES audio_content(id),
    PRIMARY KEY (group_id, audio_content_id)
);

CREATE TABLE group_member (
    group_id         INTEGER NOT NULL REFERENCES dup_group(id),
    track_id         INTEGER NOT NULL REFERENCES track(id),
    audio_content_id INTEGER NOT NULL,
    is_keeper        INTEGER NOT NULL DEFAULT 0,
    rank_score       TEXT,
    PRIMARY KEY (group_id, track_id)
);

CREATE TABLE acoustid_cache (
    lookup_key    TEXT PRIMARY KEY,
    response_json TEXT,
    fetched_at    TEXT
);

CREATE TABLE quarantine_log (
    id       INTEGER PRIMARY KEY,
    run_id   INTEGER NOT NULL,
    track_id INTEGER NOT NULL,
    group_id INTEGER NOT NULL,
    src_path TEXT NOT NULL,
    dst_path TEXT NOT NULL,
    src_hash TEXT,
    dst_hash TEXT,
    moved_at TEXT,
    state    TEXT NOT NULL CHECK (state IN ('moved','failed','undone'))
);
"""

_MIGRATION_2 = """
CREATE TABLE quality_flag (
    id               INTEGER PRIMARY KEY,
    audio_content_id INTEGER NOT NULL REFERENCES audio_content(id),
    track_id         INTEGER NOT NULL REFERENCES track(id),
    peak_db          REAL,
    mean_db          REAL,
    clipping         INTEGER NOT NULL DEFAULT 0,
    low_volume       INTEGER NOT NULL DEFAULT 0,
    silence_sections INTEGER NOT NULL DEFAULT 0,
    analyzed_at      TEXT,
    UNIQUE (audio_content_id)
);
"""

_MIGRATION_3 = """
CREATE TABLE audio_features (
    id                 INTEGER PRIMARY KEY,
    audio_content_id   INTEGER NOT NULL REFERENCES audio_content(id),
    bpm                REAL,
    bpm_confidence     REAL,
    key_name           TEXT,
    scale              TEXT,
    key_strength       REAL,
    loudness_lufs      REAL,
    danceability       REAL,
    energy             REAL,
    spectral_centroid  REAL,
    onset_rate         REAL,
    dynamic_complexity REAL,
    dissonance         REAL,
    zcr                REAL,
    mfcc_mean          BLOB,
    extractor_version  TEXT,
    config_hash        TEXT,
    analyzed_at        TEXT,
    UNIQUE (audio_content_id)
);
"""

_MIGRATION_4 = """
CREATE TABLE musicbrainz_match (
    id                 INTEGER PRIMARY KEY,
    audio_content_id   INTEGER NOT NULL REFERENCES audio_content(id),
    acoustid_id        TEXT,
    recording_mbid     TEXT,
    recording_title    TEXT,
    artists_json       TEXT,
    release_title      TEXT,
    release_mbid       TEXT,
    score              REAL,
    parsed_at          TEXT,
    UNIQUE (audio_content_id)
);

ALTER TABLE acoustid_cache ADD COLUMN audio_content_id INTEGER REFERENCES audio_content(id);
"""

_MIGRATION_5 = """
CREATE TABLE cluster_run (
    id                  INTEGER PRIMARY KEY,
    n_clusters          INTEGER NOT NULL,
    n_tracks            INTEGER NOT NULL,
    scaler_params       BLOB,
    created_at          TEXT
);

CREATE TABLE cluster_centroid (
    id                  INTEGER PRIMARY KEY,
    run_id              INTEGER NOT NULL REFERENCES cluster_run(id),
    cluster_id          INTEGER NOT NULL,
    label               TEXT,
    centroid            BLOB,
    n_tracks            INTEGER,
    UNIQUE (run_id, cluster_id)
);

CREATE TABLE cluster_assignment (
    run_id              INTEGER NOT NULL REFERENCES cluster_run(id),
    audio_content_id    INTEGER NOT NULL REFERENCES audio_content(id),
    cluster_id          INTEGER NOT NULL,
    distance_to_centroid REAL,
    PRIMARY KEY (run_id, audio_content_id)
);

CREATE TABLE track_similarity (
    track_a_id       INTEGER NOT NULL REFERENCES track(id),
    track_b_id       INTEGER NOT NULL REFERENCES track(id),
    mfcc_norm        REAL NOT NULL,
    bpm_norm         REAL,
    key_norm         REAL,
    energy_norm      REAL,
    combined_score   REAL NOT NULL,
    PRIMARY KEY (track_a_id, track_b_id),
    CHECK (track_a_id < track_b_id)
);
CREATE INDEX sim_track_a ON track_similarity(track_a_id, combined_score);
CREATE INDEX sim_track_b ON track_similarity(track_b_id, combined_score);
"""

_MIGRATION_6 = """
DROP INDEX IF EXISTS sim_track_a;
DROP INDEX IF EXISTS sim_track_b;
DROP TABLE IF EXISTS track_similarity;

CREATE TABLE track_similarity (
    track_id         INTEGER NOT NULL REFERENCES track(id),
    neighbor_id      INTEGER NOT NULL REFERENCES track(id),
    mfcc_norm        REAL NOT NULL,
    bpm_norm         REAL,
    key_norm         REAL,
    energy_norm      REAL,
    combined_score   REAL NOT NULL,
    config_hash      TEXT,
    PRIMARY KEY (track_id, neighbor_id)
);
CREATE INDEX sim_track_score ON track_similarity(track_id, combined_score);
"""

_MIGRATION_7_STATEMENTS: list[str] = [
    "CREATE TABLE mb_artist (mbid TEXT PRIMARY KEY, name TEXT NOT NULL)",
    "CREATE TABLE mb_recording_artist ("
    "match_id INTEGER NOT NULL REFERENCES musicbrainz_match(id), "
    "position INTEGER NOT NULL, "
    "artist_mbid TEXT NOT NULL REFERENCES mb_artist(mbid), "
    "PRIMARY KEY (match_id, position))",
    "CREATE INDEX recording_artist_mbid ON mb_recording_artist(artist_mbid)",
]


def _backfill_mb_artists(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT id, artists_json FROM musicbrainz_match "
        "WHERE artists_json IS NOT NULL"
    ).fetchall()
    for row in rows:
        artists = json.loads(row["artists_json"])
        for position, artist in enumerate(artists):
            conn.execute(
                "INSERT OR IGNORE INTO mb_artist (mbid, name) VALUES (?, ?)",
                (artist["mbid"], artist["name"]))
            conn.execute(
                "INSERT INTO mb_recording_artist (match_id, position, artist_mbid) "
                "VALUES (?, ?, ?)",
                (row["id"], position, artist["mbid"]))


def _migration_7(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys=OFF")
    try:
        conn.execute("BEGIN")
        for stmt in _MIGRATION_7_STATEMENTS:
            conn.execute(stmt)
        _backfill_mb_artists(conn)
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError(
                f"Migration 7 left {len(violations)} dangling reference(s): {violations}")
        conn.execute("DELETE FROM schema_version")
        conn.execute("INSERT INTO schema_version (version) VALUES (7)")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.execute("PRAGMA foreign_keys=ON")


_MIGRATIONS = [_MIGRATION_1, _MIGRATION_2, _MIGRATION_3, _MIGRATION_4, _MIGRATION_5,
               _MIGRATION_6, _migration_7]


def connect(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    have = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name='schema_version'"
    ).fetchone()
    current = 0
    if have:
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        current = row["version"] if row else 0

    for i, step in enumerate(_MIGRATIONS, start=1):
        if current < i:
            if callable(step):
                step(conn)  # owns its own transaction AND schema_version update
            else:
                conn.executescript(step)
                conn.execute("DELETE FROM schema_version")
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (i,))
    return conn


@contextlib.contextmanager
def exclusive_lock(db_path: Path):
    lock_path = Path(str(db_path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = lock_path.open("w")
    try:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise LockError(
                f"another riffle process holds {lock_path}"
            ) from exc
        yield
    finally:
        fh.close()


def pack_fingerprint(arr: np.ndarray) -> bytes:
    return np.ascontiguousarray(arr, dtype="<u4").tobytes()


def unpack_fingerprint(blob: bytes, fp_length: int) -> np.ndarray:
    if len(blob) != fp_length * 4:
        raise ValueError(f"blob is {len(blob)} bytes, expected {fp_length * 4}")
    return np.frombuffer(blob, dtype="<u4").astype(np.uint32)


def pack_mfcc(arr: np.ndarray) -> bytes:
    return np.ascontiguousarray(arr, dtype=np.float64).tobytes()


def unpack_mfcc(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float64).copy()
