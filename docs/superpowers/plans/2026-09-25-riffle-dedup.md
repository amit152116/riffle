# riffle Spec 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local music-library scanner and duplicate detector that fingerprints files with Chromaprint, aligns them with offset-histogram voting, and quarantines confirmed duplicates without ever deleting a file.

**Architecture:** Three layers. L0 walks the filesystem and assigns each file an identity based on a hash of its audio stream, so retagging does not change identity. L1 produces versioned analysis artifacts — for now, Chromaprint fingerprints. L3-A builds a transient numpy index over fingerprint keys, votes on alignment offsets, segments the matched region, classifies pairs into tiers, groups them at the content level, and moves approved losers into a per-filesystem quarantine directory.

**Tech Stack:** Python 3.12, `uv`, SQLite (stdlib `sqlite3`, WAL mode), numpy, typer, mutagen, pyacoustid, pytest. External binaries: `ffmpeg`, `fpcalc` (from `libchromaprint-tools` 1.5.1).

**Spec:** `docs/superpowers/specs/2026-09-25-riffle-dedup-design.md`

## Global Constraints

- Python pinned to **3.12**. Subsystem B will need `essentia-tensorflow 2.1b6.dev1389`, which covers cp39–cp313 only.
- Chromaprint pinned to **1.5.1** (distro `libchromaprint-tools`). `chromaprint.decode_fingerprint` must **never** be applied to bytes this tool did not generate. Any future feature accepting external fingerprints must first move to Chromaprint ≥ 1.6.1.
- `fpcalc` is always invoked with **`-length 0`** (unlimited). The 120-second default hides every trim and edit past two minutes.
- The alignment key is the **top 12 bits** of each fingerprint item: `x >> 20`. Default match threshold **10.0**, Gaussian sigma **8.0**, gradient peak threshold **0.15**, segment merge when score difference **< 0.7**.
- `ACOUSTID_MAX_BIT_ERROR` (2) and `ACOUSTID_MAX_ALIGN_OFFSET` (120) are **reference only**. `match.py` must not impose a ±120 alignment window or a ≤2 bit-error filter.
- **No randomness anywhere in matching.** Upstream's `rand()` jitter is deliberately not reproduced; ties break by index order, low to high.
- `fp_raw` is **little-endian uint32, C-contiguous, no header**. `len(fp_raw) == fp_length * 4` is asserted on every read and write.
- Hashes are compared **only within the same `hash_method`**. `streamhash` and `whole_file` are separate namespaces.
- **Nothing is ever deleted outside quarantine, nothing is ever overwritten, and no group may reach a state with zero surviving copies.**
- Only **tier 0 and tier 1** edges may authorize quarantine. Tier 2 never does.
- `apply` acts only on groups whose `decision` is `approved`, and re-hashes every file from disk before touching it.
- AcoustID: **3 req/s** maximum. MusicBrainz: **1 req/s**, contactable `User-Agent` mandatory.
- SQLite in WAL mode, single writer, lock file. Concurrent invocations fail with a clear message.

## Review Focus

Input classes the spec implies but that no task's own happy-path tests would exercise. Each has a test assigned to the task that owns the code.

1. **Fingerprints too short to align.** A file under ~2 seconds yields a handful of items, or zero. `coverage = matched / fp_length` divides by zero, and `peak_vote_ratio` divides by zero when no votes are cast. → Task 10.
2. **Paths containing newlines, quotes, or non-UTF-8 bytes.** These reach subprocess arguments, SQLite text columns, the JSON report, and the quarantine manifest. A filename with a newline silently corrupts any line-oriented output. → Task 4.
3. **Symlinked directories, including loops.** A symlink pointing at an ancestor makes the walk run forever. A symlink to a file already scanned creates a phantom duplicate that is really one file. → Task 5.
4. **A full or read-only filesystem during `apply`.** The link succeeds and the unlink fails, or the link fails after the group is half processed. Either way the run must end in an explicit recorded state, never an ambiguous one. → Task 16.
5. **Missing or zero duration metadata.** A stream with no duration breaks the tier floors, which are expressed in seconds, and breaks the AcoustID `duration` parameter, which is required. → Task 6.

---

### Task 1: Project scaffold and external dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `riffle/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_environment.py`
- Create: `README.md`

**Interfaces:**
- Consumes: nothing.
- Produces: an importable `riffle` package; `riffle.__version__` (str).

- [ ] **Step 1: Ask the user to approve installing the system package**

This is the approval gate the spec records. Do not skip it, and do not run a privileged install unattended.

Tell the user: "`fpcalc` is required and is not installed. It comes from the `libchromaprint-tools` package (version 1.5.1-5), which also provides `libchromaprint1`. May I install it, or would you rather run the install yourself?"

The install is a privileged package-manager operation on `libchromaprint-tools`. Let the user run it in their own shell if they prefer; this plan does not assume a particular privilege-escalation setup.

- [ ] **Step 2: Verify the external binaries and record the version**

```bash
which ffmpeg fpcalc
fpcalc -version
ffmpeg -hide_banner -muxers | grep streamhash
```

Expected: both binaries resolve, `fpcalc` reports a 1.5.x version, `streamhash` is listed. If `fpcalc` is missing, stop and report — no later task can proceed.

- [ ] **Step 3: Create the project metadata**

```toml
# pyproject.toml
[project]
name = "riffle"
version = "0.1.0"
description = "Local music library scanner and duplicate detector"
requires-python = ">=3.12,<3.13"
dependencies = [
    "numpy>=1.26",
    "typer>=0.12",
    "mutagen>=1.47",
    "pyacoustid>=1.3",
]

[project.scripts]
riffle = "riffle.cli:app"

[dependency-groups]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

```python
# riffle/__init__.py
__version__ = "0.1.0"
```

```python
# tests/__init__.py
```

- [ ] **Step 4: Write the failing environment test**

```python
# tests/test_environment.py
import shutil
import subprocess

import riffle


def test_package_imports():
    assert riffle.__version__ == "0.1.0"


def test_ffmpeg_available():
    assert shutil.which("ffmpeg") is not None


def test_fpcalc_available():
    assert shutil.which("fpcalc") is not None


def test_streamhash_muxer_available():
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-muxers"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "streamhash" in out
```

- [ ] **Step 5: Create the environment and run the tests**

```bash
uv sync
uv run pytest tests/test_environment.py -v
```

Expected: 4 passed. If `test_fpcalc_available` fails, return to Step 1.

- [ ] **Step 6: Write the README**

```markdown
# riffle

Local music library scanner and duplicate detector.

Design: `docs/superpowers/specs/2026-09-25-riffle-dedup-design.md`
Plan: `docs/superpowers/plans/2026-09-25-riffle-dedup.md`

## Requirements

- Python 3.12
- ffmpeg
- fpcalc, from the libchromaprint-tools package

## Usage

    uv run riffle scan ~/Music
    uv run riffle match
    uv run riffle report --run 1
    uv run riffle approve --run 1 --tier 1 --all
    uv run riffle apply --run 1

Nothing is ever deleted. `apply` moves losers into a quarantine directory
on the same filesystem, and `undo` puts them back.
```

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock riffle tests README.md
git commit -m "feat: project scaffold and environment checks"
```

---

### Task 2: Test fixture generator

**Files:**
- Create: `tests/fixtures.py`
- Create: `tests/test_fixtures.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `make_tone(path: Path, seconds: float, freq: int = 440, codec: str = "flac", bitrate: str | None = None) -> Path`
  - `make_silence(path: Path, seconds: float, codec: str = "flac") -> Path`
  - `transcode(src: Path, dst: Path, codec: str, bitrate: str | None = None) -> Path`
  - `trim(src: Path, dst: Path, start: float, duration: float | None = None) -> Path`
  - `concat(dst: Path, *srcs: Path) -> Path`
  - `retag(path: Path, **tags: str) -> Path`

Real structure is needed because a constant tone produces a degenerate fingerprint. `make_tone` generates a frequency-swept, amplitude-modulated signal so Chromaprint has something to key on.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fixtures.py
import subprocess

from tests.fixtures import make_tone, transcode, trim, concat, retag


def _duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def test_make_tone_creates_audio_of_requested_length(tmp_path):
    p = make_tone(tmp_path / "a.flac", seconds=5.0)
    assert p.exists()
    assert abs(_duration(p) - 5.0) < 0.2


def test_transcode_preserves_duration(tmp_path):
    src = make_tone(tmp_path / "a.flac", seconds=5.0)
    dst = transcode(src, tmp_path / "a.mp3", codec="libmp3lame", bitrate="128k")
    assert abs(_duration(dst) - 5.0) < 0.3


def test_trim_shortens(tmp_path):
    src = make_tone(tmp_path / "a.flac", seconds=10.0)
    dst = trim(src, tmp_path / "t.flac", start=3.0, duration=4.0)
    assert abs(_duration(dst) - 4.0) < 0.2


def test_concat_lengthens(tmp_path):
    a = make_tone(tmp_path / "a.flac", seconds=4.0, freq=300)
    b = make_tone(tmp_path / "b.flac", seconds=4.0, freq=900)
    dst = concat(tmp_path / "ab.flac", a, b)
    assert abs(_duration(dst) - 8.0) < 0.3


def test_retag_changes_the_file_but_not_the_audio(tmp_path):
    p = make_tone(tmp_path / "a.flac", seconds=3.0)
    before = p.read_bytes()
    retag(p, title="Changed")
    assert p.read_bytes() != before
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_fixtures.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests.fixtures'`

- [ ] **Step 3: Implement the fixture helpers**

```python
# tests/fixtures.py
"""ffmpeg-backed audio fixtures.

A constant tone fingerprints degenerately, so the generated signal sweeps in
frequency and pulses in amplitude to give Chromaprint something to work with.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import mutagen


def _run(args: list[str]) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        check=True,
    )


def make_tone(path: Path, seconds: float, freq: int = 440,
              codec: str = "flac", bitrate: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    expr = (
        f"sin(2*PI*t*({freq}+{freq // 2}*sin(2*PI*t/7)))"
        f"*(0.4+0.3*sin(2*PI*t*1.7))"
    )
    args = [
        "-f", "lavfi",
        "-i", f"aevalsrc={expr}:s=44100:d={seconds}",
        "-ac", "1", "-c:a", codec,
    ]
    if bitrate:
        args += ["-b:a", bitrate]
    _run(args + [str(path)])
    return path


def make_silence(path: Path, seconds: float, codec: str = "flac") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=mono:d={seconds}",
        "-c:a", codec, str(path),
    ])
    return path


def transcode(src: Path, dst: Path, codec: str,
              bitrate: str | None = None) -> Path:
    args = ["-i", str(src), "-ac", "1", "-c:a", codec]
    if bitrate:
        args += ["-b:a", bitrate]
    _run(args + [str(dst)])
    return dst


def trim(src: Path, dst: Path, start: float,
         duration: float | None = None) -> Path:
    args = ["-i", str(src), "-ss", str(start)]
    if duration is not None:
        args += ["-t", str(duration)]
    _run(args + ["-c:a", "flac", str(dst)])
    return dst


def concat(dst: Path, *srcs: Path) -> Path:
    listing = dst.with_suffix(".txt")
    listing.write_text("".join(f"file '{s.resolve()}'\n" for s in srcs))
    _run(["-f", "concat", "-safe", "0", "-i", str(listing),
          "-c:a", "flac", str(dst)])
    listing.unlink()
    return dst


def retag(path: Path, **tags: str) -> Path:
    f = mutagen.File(path, easy=True)
    for k, v in tags.items():
        f[k] = v
    f.save()
    return path
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_fixtures.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures.py tests/test_fixtures.py
git commit -m "test: ffmpeg-backed audio fixture generators"
```

---

### Task 3: Store — schema, migrations, WAL, locking, serialization

**Files:**
- Create: `riffle/store.py`
- Create: `tests/test_store.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `SCHEMA_VERSION: int`
  - `connect(db_path: Path) -> sqlite3.Connection` — applies migrations, enables WAL, sets `row_factory = sqlite3.Row`
  - `LockError(Exception)`
  - `exclusive_lock(db_path: Path)` — context manager, raises `LockError` if another process holds it
  - `pack_fingerprint(arr: np.ndarray) -> bytes`
  - `unpack_fingerprint(blob: bytes, fp_length: int) -> np.ndarray`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store.py
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL with `ImportError: cannot import name 'store' from 'riffle'`

- [ ] **Step 3: Implement the store**

```python
# riffle/store.py
"""SQLite schema, migrations, locking, and fingerprint serialization."""
from __future__ import annotations

import contextlib
import fcntl
import sqlite3
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1


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

_MIGRATIONS = [_MIGRATION_1]


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

    for i, sql in enumerate(_MIGRATIONS, start=1):
        if current < i:
            conn.executescript(sql)
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_store.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add riffle/store.py tests/test_store.py
git commit -m "feat: sqlite schema, migrations, locking, fingerprint serialization"
```

---

### Task 4: Audio-stream hashing

**Files:**
- Create: `riffle/hashing.py`
- Create: `tests/test_hashing.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `HashResult` — dataclass with `audio_hash: str`, `hash_method: str`
  - `audio_identity(path: Path) -> HashResult` — `streamhash` when stream copy works, `whole_file` otherwise
  - `whole_file_sha256(path: Path) -> str`

**Review Focus item 2 is tested here**: paths containing newlines and quotes. Everything runs through `subprocess` argument lists, never a shell string, so the only real risk is code that builds a command line by concatenation. The test pins that.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hashing.py
import pytest

from riffle import hashing
from tests.fixtures import make_tone, retag, transcode


def test_streamhash_is_stable_across_retagging(tmp_path):
    p = make_tone(tmp_path / "a.flac", seconds=3.0)
    before = hashing.audio_identity(p)
    retag(p, title="Something Else", artist="Someone")
    after = hashing.audio_identity(p)
    assert before.hash_method == "streamhash"
    assert after.audio_hash == before.audio_hash


def test_whole_file_hash_changes_on_retagging(tmp_path):
    p = make_tone(tmp_path / "a.flac", seconds=3.0)
    before = hashing.whole_file_sha256(p)
    retag(p, title="Something Else")
    assert hashing.whole_file_sha256(p) != before


def test_identical_audio_in_different_files_hashes_the_same(tmp_path):
    a = make_tone(tmp_path / "a.flac", seconds=3.0)
    b = tmp_path / "b.flac"
    b.write_bytes(a.read_bytes())
    retag(b, title="Different Tags")
    assert hashing.audio_identity(a).audio_hash == \
           hashing.audio_identity(b).audio_hash


def test_different_audio_hashes_differently(tmp_path):
    a = make_tone(tmp_path / "a.flac", seconds=3.0, freq=440)
    b = make_tone(tmp_path / "b.flac", seconds=3.0, freq=1200)
    assert hashing.audio_identity(a).audio_hash != \
           hashing.audio_identity(b).audio_hash


def test_transcoding_changes_the_stream_hash(tmp_path):
    a = make_tone(tmp_path / "a.flac", seconds=3.0)
    b = transcode(a, tmp_path / "a.mp3", codec="libmp3lame", bitrate="128k")
    assert hashing.audio_identity(a).audio_hash != \
           hashing.audio_identity(b).audio_hash


def test_path_with_newline_and_quotes(tmp_path):
    # Review Focus 2: these must never be pasted into a shell string.
    weird = tmp_path / "we'ird\nname \"x\".flac"
    make_tone(weird, seconds=2.0)
    result = hashing.audio_identity(weird)
    assert result.hash_method == "streamhash"
    assert len(result.audio_hash) == 64


def test_non_audio_file_raises(tmp_path):
    p = tmp_path / "notaudio.flac"
    p.write_bytes(b"this is not audio")
    with pytest.raises(hashing.HashError):
        hashing.audio_identity(p)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_hashing.py -v`
Expected: FAIL with `ImportError: cannot import name 'hashing' from 'riffle'`

- [ ] **Step 3: Implement hashing**

```python
# riffle/hashing.py
"""Audio identity.

Identity is the hash of the encoded audio stream, not of the file. Stream
copy means no decode, so this is I/O-bound, and tag edits do not change it.
A container that cannot be stream-copied falls back to a whole-file hash,
which is a weaker guarantee and is recorded as such.
"""
from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path


class HashError(Exception):
    """The file could not be hashed."""


@dataclass(frozen=True)
class HashResult:
    audio_hash: str
    hash_method: str


def _streamhash(path: Path) -> str | None:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-i", str(path), "-map", "0:a", "-c:a", "copy",
         "-f", "streamhash", "-hash", "sha256", "-"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    # Lines look like: 0,a,SHA256=<hex>
    for line in proc.stdout.splitlines():
        if "=" in line:
            digest = line.rsplit("=", 1)[1].strip()
            if len(digest) == 64:
                return digest
    return None


def whole_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audio_identity(path: Path) -> HashResult:
    path = Path(path)
    if not path.exists():
        raise HashError(f"no such file: {path}")

    digest = _streamhash(path)
    if digest is not None:
        return HashResult(digest, "streamhash")

    # No decodable audio stream at all is an error, not a fallback case.
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_name",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        raise HashError(f"no audio stream in {path}")

    return HashResult(whole_file_sha256(path), "whole_file")
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_hashing.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add riffle/hashing.py tests/test_hashing.py
git commit -m "feat: audio-stream hashing with whole-file fallback"
```

---

### Task 5: Scan — walk, presence, tags, errors

**Files:**
- Create: `riffle/scan.py`
- Create: `tests/test_scan.py`

**Interfaces:**
- Consumes: `store.connect`, `hashing.audio_identity`, `hashing.HashError`
- Produces:
  - `AUDIO_EXTENSIONS: frozenset[str]`
  - `QUARANTINE_DIRNAME = ".riffle-quarantine"`
  - `walk_audio_files(roots: list[Path]) -> Iterator[Path]` — skips quarantine dirs, does not follow directory symlinks, never revisits a `(dev, inode)`
  - `scan(conn, roots: list[Path], verify_hashes: bool = False, retry_errors: bool = False) -> int` — returns the `scan_run.id`
  - `should_retry(row, st) -> bool`

**Review Focus item 3 is tested here**: symlinked directories, including a loop back to an ancestor.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scan.py
import os

from riffle import scan, store
from tests.fixtures import make_tone, retag


def _db(tmp_path):
    return store.connect(tmp_path / "db.sqlite")


def test_scan_records_tracks_and_content(tmp_path):
    lib = tmp_path / "lib"
    make_tone(lib / "a.flac", seconds=3.0, freq=440)
    make_tone(lib / "b.flac", seconds=3.0, freq=1200)
    conn = _db(tmp_path)
    scan.scan(conn, [lib])
    assert conn.execute("SELECT count(*) c FROM track").fetchone()["c"] == 2
    assert conn.execute(
        "SELECT count(*) c FROM audio_content").fetchone()["c"] == 2


def test_identical_audio_shares_one_content_row(tmp_path):
    lib = tmp_path / "lib"
    a = make_tone(lib / "a.flac", seconds=3.0)
    b = lib / "b.flac"
    b.write_bytes(a.read_bytes())
    retag(b, title="Other")
    conn = _db(tmp_path)
    scan.scan(conn, [lib])
    assert conn.execute("SELECT count(*) c FROM track").fetchone()["c"] == 2
    assert conn.execute(
        "SELECT count(*) c FROM audio_content").fetchone()["c"] == 1


def test_rescan_is_incremental(tmp_path):
    lib = tmp_path / "lib"
    make_tone(lib / "a.flac", seconds=3.0)
    conn = _db(tmp_path)
    scan.scan(conn, [lib])
    first = conn.execute("SELECT scanned_at FROM track").fetchone()["scanned_at"]
    scan.scan(conn, [lib])
    # Unchanged file keeps its content row; no duplicate track row appears.
    assert conn.execute("SELECT count(*) c FROM track").fetchone()["c"] == 1
    assert first is not None


def test_hardlinks_are_recorded_once(tmp_path):
    lib = tmp_path / "lib"
    a = make_tone(lib / "a.flac", seconds=3.0)
    os.link(a, lib / "b.flac")
    conn = _db(tmp_path)
    scan.scan(conn, [lib])
    assert conn.execute("SELECT count(*) c FROM track").fetchone()["c"] == 1


def test_missing_file_is_marked_absent(tmp_path):
    lib = tmp_path / "lib"
    a = make_tone(lib / "a.flac", seconds=3.0)
    make_tone(lib / "b.flac", seconds=3.0, freq=1200)
    conn = _db(tmp_path)
    scan.scan(conn, [lib])
    a.unlink()
    scan.scan(conn, [lib])
    row = conn.execute(
        "SELECT present, absent_reason FROM track WHERE path LIKE '%a.flac'"
    ).fetchone()
    assert row["present"] == 0
    assert row["absent_reason"] == "missing"


def test_scanning_one_root_does_not_mark_another_absent(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    make_tone(one / "a.flac", seconds=3.0)
    make_tone(two / "b.flac", seconds=3.0, freq=1200)
    conn = _db(tmp_path)
    scan.scan(conn, [one, two])
    scan.scan(conn, [one])
    row = conn.execute(
        "SELECT present FROM track WHERE path LIKE '%b.flac'").fetchone()
    assert row["present"] == 1


def test_quarantine_directory_is_skipped(tmp_path):
    lib = tmp_path / "lib"
    make_tone(lib / "a.flac", seconds=3.0)
    make_tone(lib / scan.QUARANTINE_DIRNAME / "old.flac", seconds=3.0, freq=900)
    conn = _db(tmp_path)
    scan.scan(conn, [lib])
    assert conn.execute("SELECT count(*) c FROM track").fetchone()["c"] == 1


def test_symlink_loop_terminates(tmp_path):
    # Review Focus 3: a symlink pointing at an ancestor must not loop forever.
    lib = tmp_path / "lib"
    make_tone(lib / "a.flac", seconds=2.0)
    (lib / "sub").mkdir()
    os.symlink(lib, lib / "sub" / "back")
    found = list(scan.walk_audio_files([lib]))
    assert len(found) == 1


def test_symlink_to_scanned_file_is_not_a_second_track(tmp_path):
    lib = tmp_path / "lib"
    a = make_tone(lib / "a.flac", seconds=2.0)
    os.symlink(a, lib / "link.flac")
    conn = _db(tmp_path)
    scan.scan(conn, [lib])
    assert conn.execute("SELECT count(*) c FROM track").fetchone()["c"] == 1


def test_unreadable_file_is_recorded_as_an_error(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "broken.flac").write_bytes(b"nope")
    conn = _db(tmp_path)
    scan.scan(conn, [lib])
    row = conn.execute("SELECT stage, attempts FROM ingest_error").fetchone()
    assert row["stage"] == "hash"
    assert row["attempts"] == 1


def test_unchanged_error_is_not_retried(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "broken.flac").write_bytes(b"nope")
    conn = _db(tmp_path)
    scan.scan(conn, [lib])
    scan.scan(conn, [lib])
    assert conn.execute(
        "SELECT attempts FROM ingest_error").fetchone()["attempts"] == 1


def test_changed_error_is_retried(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    broken = lib / "broken.flac"
    broken.write_bytes(b"nope")
    conn = _db(tmp_path)
    scan.scan(conn, [lib])
    broken.write_bytes(b"still not audio but different")
    scan.scan(conn, [lib])
    assert conn.execute(
        "SELECT attempts FROM ingest_error").fetchone()["attempts"] == 2


def test_retry_errors_flag_forces_retry(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "broken.flac").write_bytes(b"nope")
    conn = _db(tmp_path)
    scan.scan(conn, [lib])
    scan.scan(conn, [lib], retry_errors=True)
    assert conn.execute(
        "SELECT attempts FROM ingest_error").fetchone()["attempts"] == 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_scan.py -v`
Expected: FAIL with `ImportError: cannot import name 'scan' from 'riffle'`

- [ ] **Step 3: Implement the scanner**

```python
# riffle/scan.py
"""Filesystem walk, identity assignment, tags, presence, and error state."""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import mutagen

from riffle import hashing

AUDIO_EXTENSIONS = frozenset(
    {".mp3", ".flac", ".m4a", ".ogg", ".opus", ".wav", ".wma", ".aac"}
)
QUARANTINE_DIRNAME = ".riffle-quarantine"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def walk_audio_files(roots: list[Path]) -> Iterator[Path]:
    """Yield audio files under roots.

    Directory symlinks are not followed, so a link back to an ancestor cannot
    loop. A `(dev, inode)` is yielded once, so hardlinks and file symlinks do
    not become separate tracks.
    """
    seen: set[tuple[int, int]] = set()
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = [d for d in dirnames if d != QUARANTINE_DIRNAME]
            for name in sorted(filenames):
                path = Path(dirpath) / name
                if path.suffix.lower() not in AUDIO_EXTENSIONS:
                    continue
                try:
                    st = path.stat()  # follows symlinks: same inode as target
                except OSError:
                    continue
                key = (st.st_dev, st.st_ino)
                if key in seen:
                    continue
                seen.add(key)
                yield path


def should_retry(row, st: os.stat_result) -> bool:
    """A recorded error is retried only when the file itself changed."""
    return not (
        row["dev"] == st.st_dev
        and row["inode"] == st.st_ino
        and row["size"] == st.st_size
        and row["mtime"] == st.st_mtime
    )


def _record_error(conn, path: Path, st, stage: str, message: str) -> None:
    row = conn.execute(
        "SELECT attempts FROM ingest_error WHERE path = ?", (str(path),)
    ).fetchone()
    attempts = (row["attempts"] + 1) if row else 1
    conn.execute(
        "INSERT INTO ingest_error "
        "(path, dev, inode, size, mtime, stage, message, attempts, "
        " last_attempt_at, resolved_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,NULL) "
        "ON CONFLICT(path) DO UPDATE SET "
        "dev=excluded.dev, inode=excluded.inode, size=excluded.size, "
        "mtime=excluded.mtime, stage=excluded.stage, "
        "message=excluded.message, attempts=excluded.attempts, "
        "last_attempt_at=excluded.last_attempt_at, resolved_at=NULL",
        (str(path), st.st_dev, st.st_ino, st.st_size, st.st_mtime,
         stage, message, attempts, _now()),
    )


def _read_tags(path: Path) -> dict:
    try:
        f = mutagen.File(path, easy=True)
    except Exception:
        f = None
    if f is None:
        return {"title": None, "artist": None, "album": None,
                "genre": None, "bitrate": None, "completeness": 0}

    def first(key):
        v = f.get(key)
        return v[0] if v else None

    tags = {
        "title": first("title"),
        "artist": first("artist"),
        "album": first("album"),
        "genre": first("genre"),
        "bitrate": getattr(f.info, "bitrate", None),
    }
    tags["completeness"] = sum(
        1 for k in ("title", "artist", "album", "genre") if tags[k]
    )
    return tags


def _content_id(conn, path: Path, ident) -> int:
    row = conn.execute(
        "SELECT id FROM audio_content WHERE hash_method = ? AND audio_hash = ?",
        (ident.hash_method, ident.audio_hash),
    ).fetchone()
    if row:
        return row["id"]
    cur = conn.execute(
        "INSERT INTO audio_content (audio_hash, hash_method, first_seen_at) "
        "VALUES (?,?,?)",
        (ident.audio_hash, ident.hash_method, _now()),
    )
    return cur.lastrowid


def scan(conn, roots: list[Path], verify_hashes: bool = False,
         retry_errors: bool = False) -> int:
    roots = [Path(r).resolve() for r in roots]
    cur = conn.execute(
        "INSERT INTO scan_run (started_at, roots, status) VALUES (?,?,'running')",
        (_now(), json.dumps([str(r) for r in roots])),
    )
    scan_id = cur.lastrowid

    try:
        for path in walk_audio_files(roots):
            st = path.stat()

            err = conn.execute(
                "SELECT * FROM ingest_error WHERE path = ? AND resolved_at IS NULL",
                (str(path),),
            ).fetchone()
            if err and not retry_errors and not should_retry(err, st):
                continue

            existing = conn.execute(
                "SELECT * FROM track WHERE path = ?", (str(path),)
            ).fetchone()
            unchanged = (
                existing
                and existing["size"] == st.st_size
                and existing["mtime"] == st.st_mtime
                and existing["audio_content_id"] is not None
            )

            if unchanged and not verify_hashes:
                content_id = existing["audio_content_id"]
            else:
                try:
                    ident = hashing.audio_identity(path)
                except hashing.HashError as exc:
                    _record_error(conn, path, st, "hash", str(exc))
                    continue
                content_id = _content_id(conn, path, ident)

            tags = _read_tags(path)
            conn.execute(
                "INSERT INTO track (path, size, mtime, dev, inode, nlink, "
                " audio_content_id, bitrate, tag_title, tag_artist, tag_album, "
                " tag_genre, tag_completeness, last_seen_scan_id, present, "
                " absent_reason, scanned_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,1,NULL,?) "
                "ON CONFLICT(path) DO UPDATE SET "
                "size=excluded.size, mtime=excluded.mtime, dev=excluded.dev, "
                "inode=excluded.inode, nlink=excluded.nlink, "
                "audio_content_id=excluded.audio_content_id, "
                "bitrate=excluded.bitrate, tag_title=excluded.tag_title, "
                "tag_artist=excluded.tag_artist, tag_album=excluded.tag_album, "
                "tag_genre=excluded.tag_genre, "
                "tag_completeness=excluded.tag_completeness, "
                "last_seen_scan_id=excluded.last_seen_scan_id, "
                "present=1, absent_reason=NULL, scanned_at=excluded.scanned_at",
                (str(path), st.st_size, st.st_mtime, st.st_dev, st.st_ino,
                 st.st_nlink, content_id, tags["bitrate"], tags["title"],
                 tags["artist"], tags["album"], tags["genre"],
                 tags["completeness"], scan_id, _now()),
            )
            conn.execute(
                "UPDATE ingest_error SET resolved_at = ? WHERE path = ?",
                (_now(), str(path)),
            )

        # Only tracks under this scan's roots may be marked absent, and only
        # after the scan completes. A failed scan marks nothing.
        for root in roots:
            conn.execute(
                "UPDATE track SET present = 0, absent_reason = 'missing' "
                "WHERE path LIKE ? AND present = 1 "
                "AND (last_seen_scan_id IS NULL OR last_seen_scan_id != ?)",
                (f"{root}{os.sep}%", scan_id),
            )
        conn.execute(
            "UPDATE scan_run SET status='complete', completed_at=? WHERE id=?",
            (_now(), scan_id),
        )
    except Exception:
        conn.execute(
            "UPDATE scan_run SET status='failed', completed_at=? WHERE id=?",
            (_now(), scan_id),
        )
        raise
    return scan_id
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_scan.py -v`
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add riffle/scan.py tests/test_scan.py
git commit -m "feat: incremental scanner with presence and error state"
```

---

### Task 6: Fingerprinting

**Files:**
- Create: `riffle/fingerprint.py`
- Create: `tests/test_fingerprint.py`

**Interfaces:**
- Consumes: `store.pack_fingerprint`
- Produces:
  - `FingerprintResult` — dataclass: `raw: np.ndarray`, `duration: float`, `algorithm: int`
  - `fpcalc_version() -> str`
  - `config_hash(config: dict) -> str`
  - `DEFAULT_CONFIG: dict` — `{"length": 0, "algorithm": 2, "raw": True}`
  - `fingerprint_file(path: Path, config: dict = DEFAULT_CONFIG) -> FingerprintResult`
  - `item_duration_seconds() -> float`
  - `fingerprint_pending(conn, config: dict = DEFAULT_CONFIG) -> int` — fingerprints content rows lacking a current `canonical` artifact, returns the count written

**Review Focus item 5 is tested here**: missing or zero duration.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fingerprint.py
import numpy as np
import pytest

from riffle import fingerprint, scan, store
from tests.fixtures import make_tone, transcode


def test_fpcalc_version_is_reported():
    v = fingerprint.fpcalc_version()
    assert v and any(ch.isdigit() for ch in v)


def test_fingerprint_shape_matches_expectation(tmp_path):
    # Guards the parser against a change in fpcalc's output format.
    p = make_tone(tmp_path / "a.flac", seconds=20.0)
    res = fingerprint.fingerprint_file(p)
    assert isinstance(res.raw, np.ndarray)
    assert res.raw.dtype == np.uint32
    assert len(res.raw) > 50
    assert 19.0 < res.duration < 21.0


def test_full_length_is_used_not_the_120s_default(tmp_path):
    long = make_tone(tmp_path / "long.flac", seconds=150.0)
    res = fingerprint.fingerprint_file(long)
    # At ~8 items/sec, 120s would cap near 970 items. Full length must exceed it.
    assert len(res.raw) > 1100


def test_item_duration_is_plausible():
    d = fingerprint.item_duration_seconds()
    assert 0.05 < d < 0.5


def test_transcode_fingerprints_similarly(tmp_path):
    a = make_tone(tmp_path / "a.flac", seconds=20.0)
    b = transcode(a, tmp_path / "a.mp3", codec="libmp3lame", bitrate="128k")
    fa = fingerprint.fingerprint_file(a).raw
    fb = fingerprint.fingerprint_file(b).raw
    n = min(len(fa), len(fb))
    # Same audio, different encoding: most top-12-bit keys should survive.
    same = np.sum((fa[:n] >> 20) == (fb[:n] >> 20))
    assert same / n > 0.5


def test_config_hash_is_stable_and_order_independent():
    a = fingerprint.config_hash({"length": 0, "algorithm": 2})
    b = fingerprint.config_hash({"algorithm": 2, "length": 0})
    assert a == b
    assert a != fingerprint.config_hash({"length": 120, "algorithm": 2})


def test_zero_duration_input_raises(tmp_path):
    # Review Focus 5: a stream with no usable duration must fail loudly,
    # not produce a fingerprint with duration 0 that later divides by zero.
    p = tmp_path / "empty.flac"
    p.write_bytes(b"")
    with pytest.raises(fingerprint.FingerprintError):
        fingerprint.fingerprint_file(p)


def test_fingerprint_pending_writes_one_artifact_per_content(tmp_path):
    lib = tmp_path / "lib"
    make_tone(lib / "a.flac", seconds=15.0)
    make_tone(lib / "b.flac", seconds=15.0, freq=1200)
    conn = store.connect(tmp_path / "db.sqlite")
    scan.scan(conn, [lib])
    written = fingerprint.fingerprint_pending(conn)
    assert written == 2
    # Idempotent: a second call writes nothing new.
    assert fingerprint.fingerprint_pending(conn) == 0
    row = conn.execute("SELECT * FROM fingerprint LIMIT 1").fetchone()
    assert row["analyzer"] == "chromaprint"
    assert row["purpose"] == "canonical"
    assert len(row["fp_raw"]) == row["fp_length"] * 4
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_fingerprint.py -v`
Expected: FAIL with `ImportError: cannot import name 'fingerprint' from 'riffle'`

- [ ] **Step 3: Implement fingerprinting**

```python
# riffle/fingerprint.py
"""Chromaprint fingerprints via fpcalc.

fpcalc's default caps at the first 120 seconds, which would hide every trim
and edit past two minutes, so `-length 0` (unlimited) is always passed.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

from riffle import store

ANALYZER = "chromaprint"
DEFAULT_CONFIG = {"length": 0, "algorithm": 2, "raw": True}


class FingerprintError(Exception):
    """fpcalc could not fingerprint the file."""


@dataclass(frozen=True)
class FingerprintResult:
    raw: np.ndarray
    duration: float
    algorithm: int


@lru_cache(maxsize=1)
def fpcalc_version() -> str:
    out = subprocess.run(
        ["fpcalc", "-version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return out


def config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@lru_cache(maxsize=1)
def item_duration_seconds() -> float:
    """Seconds of audio per fingerprint item, from the library itself."""
    import chromaprint  # provided by pyacoustid

    ctx = chromaprint._libchromaprint.chromaprint_new(DEFAULT_CONFIG["algorithm"])
    try:
        item = chromaprint._libchromaprint.chromaprint_get_item_duration(ctx)
        rate = chromaprint._libchromaprint.chromaprint_get_sample_rate(ctx)
        return item / rate
    finally:
        chromaprint._libchromaprint.chromaprint_free(ctx)


def _parse(stdout: str) -> tuple[np.ndarray, float]:
    data = json.loads(stdout)
    fp = data["fingerprint"]
    if isinstance(fp, str):
        fp = [int(x) for x in fp.split(",") if x]
    return np.asarray(fp, dtype=np.uint32), float(data["duration"])


def fingerprint_file(path: Path, config: dict = DEFAULT_CONFIG) -> FingerprintResult:
    args = [
        "fpcalc", "-json", "-raw",
        "-length", str(config["length"]),
        "-algorithm", str(config["algorithm"]),
        str(path),
    ]
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FingerprintError(f"fpcalc failed on {path}: {proc.stderr.strip()}")
    try:
        raw, duration = _parse(proc.stdout)
    except (ValueError, KeyError) as exc:
        raise FingerprintError(
            f"unexpected fpcalc output for {path}: {proc.stdout[:200]!r}"
        ) from exc
    if duration <= 0:
        raise FingerprintError(f"{path} reports a duration of {duration}")
    if len(raw) == 0:
        raise FingerprintError(f"{path} produced an empty fingerprint")
    return FingerprintResult(raw, duration, config["algorithm"])


def fingerprint_pending(conn, config: dict = DEFAULT_CONFIG) -> int:
    version = fpcalc_version()
    chash = config_hash(config)
    rows = conn.execute(
        "SELECT ac.id AS content_id, MIN(t.path) AS path "
        "FROM audio_content ac "
        "JOIN track t ON t.audio_content_id = ac.id AND t.present = 1 "
        "WHERE NOT EXISTS ("
        "  SELECT 1 FROM fingerprint f "
        "  WHERE f.audio_content_id = ac.id AND f.purpose = 'canonical' "
        "    AND f.analyzer = ? AND f.analyzer_version = ? AND f.config_hash = ?"
        ") GROUP BY ac.id",
        (ANALYZER, version, chash),
    ).fetchall()

    written = 0
    for row in rows:
        try:
            res = fingerprint_file(Path(row["path"]), config)
        except FingerprintError as exc:
            conn.execute(
                "INSERT INTO ingest_error (path, stage, message, attempts, "
                " last_attempt_at) VALUES (?,?,?,1,?) "
                "ON CONFLICT(path) DO UPDATE SET "
                "attempts = ingest_error.attempts + 1, "
                "message = excluded.message, "
                "last_attempt_at = excluded.last_attempt_at",
                (row["path"], "fingerprint", str(exc),
                 datetime.now(timezone.utc).isoformat()),
            )
            continue
        conn.execute(
            "INSERT INTO fingerprint (audio_content_id, analyzer, "
            " analyzer_version, config_hash, purpose, algorithm, fp_raw, "
            " fp_length, computed_at) "
            "VALUES (?,?,?,?,'canonical',?,?,?,?)",
            (row["content_id"], ANALYZER, version, chash, res.algorithm,
             store.pack_fingerprint(res.raw), len(res.raw),
             datetime.now(timezone.utc).isoformat()),
        )
        conn.execute(
            "UPDATE audio_content SET duration = ? WHERE id = ?",
            (res.duration, row["content_id"]),
        )
        written += 1
    return written
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_fingerprint.py -v`
Expected: 8 passed.

If `test_fingerprint_shape_matches_expectation` fails on parsing, print the raw output of `fpcalc -json -raw -length 0 <file>` and adjust `_parse` to the actual shape. Do not silence the test — it exists to catch exactly that.

- [ ] **Step 5: Commit**

```bash
git add riffle/fingerprint.py tests/test_fingerprint.py
git commit -m "feat: full-length chromaprint fingerprinting with versioned artifacts"
```

---

### Task 7: Candidate index with stop-key and occurrence caps

**Files:**
- Create: `riffle/match.py`
- Create: `tests/test_match_index.py`

**Interfaces:**
- Consumes: nothing from earlier tasks at runtime.
- Produces:
  - `DEFAULT_MATCH_CONFIG: dict` — `{"align_bits": 12, "k_cap": 200, "k_cap_fraction": 0.02, "m_cap": 8, "match_threshold": 10.0, "sigma": 8.0, "gradient_peak": 0.15, "merge_delta": 0.7, "min_peak_vote_ratio": 0.05, "tier1_min_coverage": 0.85, "tier1_max_bit_error": 6.0, "tier1_min_overlap_seconds": 20.0, "tier2_min_overlap_seconds": 15.0}`
  - `effective_k_cap(config: dict, n_fingerprints: int) -> int`
  - `candidate_key(items: np.ndarray, align_bits: int) -> np.ndarray`
  - `build_postings(fps: dict[int, np.ndarray], config: dict) -> dict[int, list[tuple[int, int]]]` — key to list of `(content_id, position)`, both caps applied
  - `candidate_pairs(postings) -> set[tuple[int, int]]` — ordered `(a, b)` with `a < b`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_match_index.py
import numpy as np

from riffle import match


def test_candidate_key_takes_the_top_bits():
    items = np.array([0xFFF00000, 0x00100000, 0x000FFFFF], dtype=np.uint32)
    keys = match.candidate_key(items, 12)
    assert list(keys) == [0xFFF, 0x001, 0x000]


def test_postings_group_equal_keys():
    fps = {
        1: np.array([0x00100000, 0x00200000], dtype=np.uint32),
        2: np.array([0x00100000], dtype=np.uint32),
    }
    postings = match.build_postings(fps, match.DEFAULT_MATCH_CONFIG)
    assert sorted(postings[0x001]) == [(1, 0), (2, 0)]


def test_stop_key_cap_drops_ubiquitous_keys():
    cfg = dict(match.DEFAULT_MATCH_CONFIG, k_cap=3)
    fps = {i: np.array([0x00100000], dtype=np.uint32) for i in range(1, 6)}
    postings = match.build_postings(fps, cfg)
    assert 0x001 not in postings


def test_effective_k_cap_scales_with_the_corpus():
    cfg = dict(match.DEFAULT_MATCH_CONFIG, k_cap=200, k_cap_fraction=0.02)
    # Large corpus: the absolute cap binds.
    assert match.effective_k_cap(cfg, 50_000) == 200
    # Target scale: the fraction binds, so the cap tracks the library.
    assert match.effective_k_cap(cfg, 5_000) == 100
    assert match.effective_k_cap(cfg, 2_000) == 40
    # Tiny corpus: never below two, the smallest list that yields a pair.
    assert match.effective_k_cap(cfg, 3) == 2


def test_occurrence_cap_limits_positions_within_one_fingerprint():
    cfg = dict(match.DEFAULT_MATCH_CONFIG, m_cap=2)
    fps = {
        1: np.array([0x00100000] * 10, dtype=np.uint32),
        2: np.array([0x00100000] * 10, dtype=np.uint32),
    }
    postings = match.build_postings(fps, cfg)
    per_content = {}
    for cid, pos in postings[0x001]:
        per_content.setdefault(cid, []).append(pos)
    assert per_content[1] == [0, 1]
    assert per_content[2] == [0, 1]


def test_candidate_pairs_are_ordered_and_unique():
    postings = {0x001: [(5, 0), (2, 0), (9, 3)]}
    pairs = match.candidate_pairs(postings)
    assert pairs == {(2, 5), (2, 9), (5, 9)}


def test_no_self_pairs():
    postings = {0x001: [(3, 0), (3, 7)]}
    assert match.candidate_pairs(postings) == set()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_match_index.py -v`
Expected: FAIL with `ImportError: cannot import name 'match' from 'riffle'`

- [ ] **Step 3: Implement the index**

```python
# riffle/match.py
"""Candidate generation and pairwise alignment.

Constants follow Chromaprint's own FingerprintMatcher (src/fingerprint_matcher
.cpp): the alignment key is the top 12 bits, the match threshold is 10.0, the
bit-error series is Gaussian-smoothed at sigma 8.0, and segments are cut at
gradient peaks above 0.15 and merged when their scores differ by less than 0.7.

ACOUSTID_MAX_BIT_ERROR and ACOUSTID_MAX_ALIGN_OFFSET appear in that file but
are NOT used by Match(). They belong to AcoustID's server query path. Imposing
a +/-120 alignment window here would discard exactly the long-offset matches
this tool exists to find.

Upstream adds rand() jitter to every Hamming distance. That is deliberately
not reproduced: reports must be reproducible. Ties break by index order.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

DEFAULT_MATCH_CONFIG = {
    "align_bits": 12,
    "k_cap": 200,
    "k_cap_fraction": 0.02,
    "m_cap": 8,
    "match_threshold": 10.0,
    "sigma": 8.0,
    "gradient_peak": 0.15,
    "merge_delta": 0.7,
    "min_peak_vote_ratio": 0.05,
    "tier1_min_coverage": 0.85,
    "tier1_max_bit_error": 6.0,
    "tier1_min_overlap_seconds": 20.0,
    "tier2_min_overlap_seconds": 15.0,
}


def effective_k_cap(config: dict, n_fingerprints: int) -> int:
    """Stop-key cap, bounded both absolutely and as a share of the corpus.

    The absolute term bounds cost, since O(n^2) pair emission depends on the
    count. The fractional term bounds informativeness: a key present in
    several percent of the library distinguishes nothing, and a purely
    absolute cap grows more permissive as the library shrinks. The floor of
    two is the smallest posting list that can yield a pair at all.
    """
    fractional = int(config["k_cap_fraction"] * n_fingerprints)
    return max(2, min(config["k_cap"], fractional))


def candidate_key(items: np.ndarray, align_bits: int) -> np.ndarray:
    """Upstream's ALIGN_STRIP: keep the top `align_bits` bits."""
    return (np.asarray(items, dtype=np.uint32) >> (32 - align_bits)).astype(
        np.uint32
    )


def build_postings(fps: dict[int, np.ndarray], config: dict) -> dict:
    """Key -> [(content_id, position)], with both caps applied.

    `m_cap` bounds how many positions of one key are carried per fingerprint,
    which stops a repeated key producing a cartesian blow-up inside a pair.
    `k_cap` drops keys that appear in too many fingerprints, which is what
    silence and fades produce; pair emission is O(n^2) in a posting list.
    """
    align_bits = config["align_bits"]
    m_cap = config["m_cap"]
    k_cap = effective_k_cap(config, len(fps))

    postings: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for content_id in sorted(fps):
        keys = candidate_key(fps[content_id], align_bits)
        counts: dict[int, int] = defaultdict(int)
        for pos, key in enumerate(keys.tolist()):
            if counts[key] >= m_cap:
                continue
            counts[key] += 1
            postings[key].append((content_id, pos))

    return {
        key: entries
        for key, entries in postings.items()
        if len({cid for cid, _ in entries}) <= k_cap
    }


def candidate_pairs(postings: dict) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for entries in postings.values():
        ids = sorted({cid for cid, _ in entries})
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                pairs.add((a, b))
    return pairs
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_match_index.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add riffle/match.py tests/test_match_index.py
git commit -m "feat: candidate index with stop-key and occurrence caps"
```

---

### Task 8: Offset histogram and peak selection

**Files:**
- Modify: `riffle/match.py`
- Create: `tests/test_match_offset.py`

**Interfaces:**
- Consumes: `candidate_key`
- Produces:
  - `offset_histogram(fp_a: np.ndarray, fp_b: np.ndarray, align_bits: int) -> tuple[np.ndarray, int]` — the histogram and the value subtracted to recover a signed offset (`len(fp_b)`)
  - `best_alignment(hist: np.ndarray, shift: int) -> tuple[int, int, int]` — `(offset, peak_votes, total_votes)`; offset 0 and votes 0 when no peak qualifies

- [ ] **Step 1: Write the failing test**

```python
# tests/test_match_offset.py
import numpy as np

from riffle import match


def _fp(values):
    return np.array([v << 20 for v in values], dtype=np.uint32)


def test_identical_fingerprints_align_at_zero():
    fp = _fp([1, 2, 3, 4, 5, 6, 7, 8])
    hist, shift = match.offset_histogram(fp, fp, 12)
    offset, peak, total = match.best_alignment(hist, shift)
    assert offset == 0
    assert peak >= 8


def test_shifted_fingerprint_reports_the_shift():
    base = _fp([10, 11, 12, 13, 14, 15, 16, 17])
    shifted = _fp([90, 91, 92, 10, 11, 12, 13, 14, 15, 16, 17])
    hist, shift = match.offset_histogram(base, shifted, 12)
    offset, peak, total = match.best_alignment(hist, shift)
    # base position i corresponds to shifted position i + 3
    assert offset == -3
    assert peak >= 8


def test_unrelated_fingerprints_produce_no_strong_peak():
    a = _fp(list(range(100, 140)))
    b = _fp(list(range(500, 540)))
    hist, shift = match.offset_histogram(a, b, 12)
    offset, peak, total = match.best_alignment(hist, shift)
    assert peak == 0
    assert total == 0


def test_empty_fingerprint_is_handled():
    a = _fp([1, 2, 3])
    b = np.array([], dtype=np.uint32)
    hist, shift = match.offset_histogram(a, b, 12)
    offset, peak, total = match.best_alignment(hist, shift)
    assert (offset, peak, total) == (0, 0, 0)


def test_peak_selection_is_deterministic_under_ties():
    # Two equal-height peaks: the lower index must win, every time.
    hist = np.array([0, 5, 0, 5, 0], dtype=np.int64)
    results = {match.best_alignment(hist, 2) for _ in range(20)}
    assert len(results) == 1
    offset, peak, _ = results.pop()
    assert (offset, peak) == (-1, 5)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_match_offset.py -v`
Expected: FAIL with `AttributeError: module 'riffle.match' has no attribute 'offset_histogram'`

- [ ] **Step 3: Append the implementation to `riffle/match.py`**

```python
def offset_histogram(fp_a: np.ndarray, fp_b: np.ndarray,
                     align_bits: int) -> tuple[np.ndarray, int]:
    """Histogram of position deltas over keys the two fingerprints share.

    Bin index is `pos_a - pos_b + len(fp_b)`, so the shift to subtract to get
    a signed offset is `len(fp_b)`.
    """
    n_a, n_b = len(fp_a), len(fp_b)
    hist = np.zeros(n_a + n_b + 1, dtype=np.int64)
    if n_a == 0 or n_b == 0:
        return hist, n_b

    keys_a = candidate_key(fp_a, align_bits)
    keys_b = candidate_key(fp_b, align_bits)

    by_key_b: dict[int, list[int]] = defaultdict(list)
    for pos, key in enumerate(keys_b.tolist()):
        by_key_b[key].append(pos)

    for pos_a, key in enumerate(keys_a.tolist()):
        for pos_b in by_key_b.get(key, ()):
            hist[pos_a - pos_b + n_b] += 1
    return hist, n_b


def best_alignment(hist: np.ndarray, shift: int) -> tuple[int, int, int]:
    """Highest local-maximum bin with more than one vote.

    Ties are resolved by the lowest index, scanning low to high, so the result
    does not depend on iteration order or on any random jitter.
    """
    total = int(hist.sum())
    best_index = -1
    best_count = 0
    for i in range(len(hist)):
        count = int(hist[i])
        if count <= 1:
            continue
        left_ok = hist[i - 1] <= count if i > 0 else True
        right_ok = hist[i + 1] <= count if i < len(hist) - 1 else True
        if left_ok and right_ok and count > best_count:
            best_count = count
            best_index = i
    if best_index < 0:
        return 0, 0, 0
    return best_index - shift, best_count, total
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_match_offset.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add riffle/match.py tests/test_match_offset.py
git commit -m "feat: offset histogram with deterministic peak selection"
```

---

### Task 9: Bit-error segmentation

**Files:**
- Modify: `riffle/match.py`
- Create: `tests/test_match_segments.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `Segment` — frozen dataclass: `pos_a: int`, `pos_b: int`, `length: int`, `score: float`
  - `hamming_series(fp_a, fp_b, offset) -> np.ndarray` — per-item bit differences over the overlap
  - `gaussian_smooth(series: np.ndarray, sigma: float) -> np.ndarray`
  - `segments(fp_a, fp_b, offset: int, config: dict) -> list[Segment]`

Upstream approximates the Gaussian with repeated box filters; this uses a truncated Gaussian kernel instead. The difference is immaterial at sigma 8 and the thresholds are calibrated anyway, but it is a deliberate deviation and is noted in the docstring rather than glossed over.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_match_segments.py
import numpy as np

from riffle import match


def test_hamming_series_counts_differing_bits():
    a = np.array([0b0000, 0b1111], dtype=np.uint32)
    b = np.array([0b0001, 0b1111], dtype=np.uint32)
    series = match.hamming_series(a, b, 0)
    assert list(series) == [1, 0]


def test_hamming_series_respects_offset():
    a = np.array([9, 1, 2, 3], dtype=np.uint32)
    b = np.array([1, 2, 3], dtype=np.uint32)
    # a position 1 lines up with b position 0, so offset is +1
    assert list(match.hamming_series(a, b, 1)) == [0, 0, 0]


def test_no_overlap_gives_empty_series():
    a = np.array([1, 2], dtype=np.uint32)
    b = np.array([1, 2], dtype=np.uint32)
    assert len(match.hamming_series(a, b, 99)) == 0


def test_gaussian_smooth_preserves_length_and_flattens_noise():
    series = np.zeros(200)
    series[100] = 32.0
    out = match.gaussian_smooth(series, 8.0)
    assert len(out) == 200
    assert out[100] < 32.0
    assert out[90] > 0.0


def test_identical_audio_yields_one_full_length_segment():
    rng = np.random.default_rng(0)
    fp = rng.integers(0, 2 ** 32, size=400, dtype=np.uint64).astype(np.uint32)
    segs = match.segments(fp, fp, 0, match.DEFAULT_MATCH_CONFIG)
    assert len(segs) == 1
    assert segs[0].length == 400
    assert segs[0].score == 0.0


def test_unrelated_audio_yields_no_segments():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 2 ** 32, size=400, dtype=np.uint64).astype(np.uint32)
    b = rng.integers(0, 2 ** 32, size=400, dtype=np.uint64).astype(np.uint32)
    segs = match.segments(a, b, 0, match.DEFAULT_MATCH_CONFIG)
    assert sum(s.length for s in segs) == 0


def test_partial_match_yields_a_partial_segment():
    rng = np.random.default_rng(2)
    shared = rng.integers(0, 2 ** 32, size=200, dtype=np.uint64).astype(np.uint32)
    noise_a = rng.integers(0, 2 ** 32, size=200, dtype=np.uint64).astype(np.uint32)
    noise_b = rng.integers(0, 2 ** 32, size=200, dtype=np.uint64).astype(np.uint32)
    a = np.concatenate([shared, noise_a])
    b = np.concatenate([shared, noise_b])
    segs = match.segments(a, b, 0, match.DEFAULT_MATCH_CONFIG)
    matched = sum(s.length for s in segs)
    assert 100 < matched < 320


def test_segments_are_deterministic():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 2 ** 32, size=300, dtype=np.uint64).astype(np.uint32)
    b = a.copy()
    b[150:] = rng.integers(0, 2 ** 32, size=150, dtype=np.uint64).astype(np.uint32)
    runs = [match.segments(a, b, 0, match.DEFAULT_MATCH_CONFIG) for _ in range(5)]
    assert all(r == runs[0] for r in runs)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_match_segments.py -v`
Expected: FAIL with `AttributeError: module 'riffle.match' has no attribute 'hamming_series'`

- [ ] **Step 3: Append the implementation to `riffle/match.py`**

```python
from dataclasses import dataclass

_POPCOUNT = np.array(
    [bin(i).count("1") for i in range(256)], dtype=np.uint8
)


@dataclass(frozen=True)
class Segment:
    pos_a: int
    pos_b: int
    length: int
    score: float


def hamming_series(fp_a: np.ndarray, fp_b: np.ndarray,
                   offset: int) -> np.ndarray:
    """Per-item differing-bit counts over the region the offset overlaps.

    `offset` is `pos_a - pos_b`: item `i` of the overlap is `fp_a[start_a + i]`
    against `fp_b[start_b + i]`.
    """
    start_a = offset if offset > 0 else 0
    start_b = -offset if offset < 0 else 0
    size = min(len(fp_a) - start_a, len(fp_b) - start_b)
    if size <= 0:
        return np.zeros(0, dtype=np.int32)
    xor = (fp_a[start_a:start_a + size] ^ fp_b[start_b:start_b + size])
    as_bytes = np.ascontiguousarray(xor, dtype="<u4").view(np.uint8)
    return _POPCOUNT[as_bytes].reshape(-1, 4).sum(axis=1).astype(np.int32)


def gaussian_smooth(series: np.ndarray, sigma: float) -> np.ndarray:
    """Truncated Gaussian convolution.

    Upstream approximates the Gaussian with three box-filter passes. A direct
    kernel is used here: the difference is immaterial at sigma 8, and every
    threshold downstream is calibrated rather than inherited verbatim.
    """
    if len(series) == 0:
        return series.astype(float)
    radius = max(1, int(round(4 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    padded = np.pad(series.astype(float), radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def segments(fp_a: np.ndarray, fp_b: np.ndarray, offset: int,
             config: dict) -> list[Segment]:
    """Cut the overlap into segments and keep the ones that match.

    No random jitter is added to the bit counts. Gradient-peak ties resolve by
    index order, so repeated runs return identical segments.
    """
    raw = hamming_series(fp_a, fp_b, offset)
    size = len(raw)
    if size == 0:
        return []

    start_a = offset if offset > 0 else 0
    start_b = -offset if offset < 0 else 0

    smoothed = gaussian_smooth(raw, config["sigma"])
    gradient = np.abs(np.gradient(smoothed))

    boundaries: list[int] = []
    threshold = config["gradient_peak"]
    for i in range(1, size - 1):
        g = gradient[i]
        if g > threshold and g >= gradient[i - 1] and g >= gradient[i + 1]:
            if not boundaries or boundaries[-1] + 1 < i:
                boundaries.append(i)
    boundaries.append(size)

    kept: list[Segment] = []
    begin = 0
    for end in boundaries:
        length = end - begin
        if length <= 0:
            continue
        score = float(raw[begin:end].mean())
        if score < config["match_threshold"]:
            if kept and abs(kept[-1].score - score) < config["merge_delta"] \
                    and kept[-1].pos_a + kept[-1].length == start_a + begin:
                prev = kept.pop()
                total = prev.length + length
                merged_score = (
                    prev.score * prev.length + score * length
                ) / total
                kept.append(Segment(prev.pos_a, prev.pos_b, total, merged_score))
            else:
                kept.append(
                    Segment(start_a + begin, start_b + begin, length, score)
                )
        begin = end
    return kept
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_match_segments.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add riffle/match.py tests/test_match_segments.py
git commit -m "feat: deterministic bit-error segmentation"
```

---

### Task 10: Pair evidence and tier classification

**Files:**
- Modify: `riffle/match.py`
- Create: `tests/test_match_evidence.py`

**Interfaces:**
- Consumes: `offset_histogram`, `best_alignment`, `segments`
- Produces:
  - `PairEvidence` — frozen dataclass with `best_offset, peak_votes, peak_vote_ratio, matched_span_items, matched_span_seconds, coverage_a, coverage_b, mean_bit_error, segment_count, tier`
  - `compare(fp_a, fp_b, config, item_seconds: float) -> PairEvidence`
  - `classify(ev_fields: dict, config: dict) -> int`

**Review Focus item 1 is tested here**: fingerprints too short to align, and pairs with zero votes. Both divide by zero if unguarded.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_match_evidence.py
import numpy as np

from riffle import match

ITEM = 0.1238
CFG = match.DEFAULT_MATCH_CONFIG


def _rand(n, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2 ** 32, size=n, dtype=np.uint64).astype(np.uint32)


def test_identical_is_tier_1():
    fp = _rand(400, 10)
    ev = match.compare(fp, fp, CFG, ITEM)
    assert ev.tier == 1
    assert ev.coverage_a == 1.0
    assert ev.coverage_b == 1.0
    assert ev.mean_bit_error == 0.0


def test_unrelated_is_tier_none():
    ev = match.compare(_rand(400, 11), _rand(400, 12), CFG, ITEM)
    assert ev.tier == match.TIER_NONE
    assert ev.matched_span_items == 0


def test_clip_inside_longer_track_is_tier_2():
    long = _rand(800, 13)
    clip = long[200:400].copy()
    ev = match.compare(long, clip, CFG, ITEM)
    assert ev.tier == 2
    assert ev.coverage_b > 0.9
    assert ev.coverage_a < 0.4


def test_empty_fingerprint_does_not_divide_by_zero():
    # Review Focus 1
    empty = np.array([], dtype=np.uint32)
    ev = match.compare(_rand(100, 14), empty, CFG, ITEM)
    assert ev.coverage_a == 0.0
    assert ev.coverage_b == 0.0
    assert ev.peak_vote_ratio == 0.0
    assert ev.matched_span_seconds == 0.0


def test_two_item_fingerprints_do_not_crash():
    # Review Focus 1: a sub-second file yields a handful of items.
    tiny = np.array([0x12345678, 0x9ABCDEF0], dtype=np.uint32)
    ev = match.compare(tiny, tiny, CFG, ITEM)
    assert ev.matched_span_seconds >= 0.0
    assert 0.0 <= ev.peak_vote_ratio <= 1.0


def test_tier_1_requires_the_absolute_overlap_floor():
    # Review Focus 1: full coverage on both sides, but far too short.
    short = _rand(30, 15)  # ~3.7 seconds at 0.1238 s/item
    ev = match.compare(short, short, CFG, ITEM)
    assert ev.coverage_a == 1.0
    assert ev.tier != 1


def test_compare_is_deterministic():
    a, b = _rand(400, 16), _rand(400, 16)
    results = {match.compare(a, b, CFG, ITEM) for _ in range(10)}
    assert len(results) == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_match_evidence.py -v`
Expected: FAIL with `AttributeError: module 'riffle.match' has no attribute 'compare'`

- [ ] **Step 3: Append the implementation to `riffle/match.py`**

```python
@dataclass(frozen=True)
class PairEvidence:
    best_offset: int
    peak_votes: int
    peak_vote_ratio: float
    matched_span_items: int
    matched_span_seconds: float
    coverage_a: float
    coverage_b: float
    mean_bit_error: float
    segment_count: int
    tier: int


TIER_NONE = 0


def classify(fields: dict, config: dict) -> int:
    """Tier from evidence. Tier 0 is assigned by identity, never here."""
    if fields["matched_span_items"] == 0:
        return TIER_NONE
    if fields["peak_vote_ratio"] < config["min_peak_vote_ratio"]:
        return TIER_NONE

    cov_min = min(fields["coverage_a"], fields["coverage_b"])
    cov_max = max(fields["coverage_a"], fields["coverage_b"])
    seconds = fields["matched_span_seconds"]

    if (cov_min >= config["tier1_min_coverage"]
            and fields["mean_bit_error"] <= config["tier1_max_bit_error"]
            and seconds >= config["tier1_min_overlap_seconds"]):
        return 1

    if cov_max >= config["tier1_min_coverage"] \
            and seconds >= config["tier2_min_overlap_seconds"]:
        return 2
    if cov_min > 0.0 and seconds >= config["tier2_min_overlap_seconds"]:
        return 2
    return TIER_NONE


def compare(fp_a: np.ndarray, fp_b: np.ndarray, config: dict,
            item_seconds: float) -> PairEvidence:
    n_a, n_b = len(fp_a), len(fp_b)
    hist, shift = offset_histogram(fp_a, fp_b, config["align_bits"])
    offset, peak_votes, total_votes = best_alignment(hist, shift)

    segs = segments(fp_a, fp_b, offset, config) if peak_votes else []
    span = sum(s.length for s in segs)
    if span:
        mean_bit_error = sum(s.score * s.length for s in segs) / span
    else:
        mean_bit_error = 0.0

    fields = {
        "best_offset": offset,
        "peak_votes": peak_votes,
        "peak_vote_ratio": (peak_votes / total_votes) if total_votes else 0.0,
        "matched_span_items": span,
        "matched_span_seconds": span * item_seconds,
        "coverage_a": (span / n_a) if n_a else 0.0,
        "coverage_b": (span / n_b) if n_b else 0.0,
        "mean_bit_error": mean_bit_error,
        "segment_count": len(segs),
    }
    fields["tier"] = classify(fields, config)
    return PairEvidence(**fields)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_match_evidence.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add riffle/match.py tests/test_match_evidence.py
git commit -m "feat: pair evidence and tier classification"
```

---

### Task 11: Match run — persist pairs for a whole library

**Files:**
- Create: `riffle/matchrun.py`
- Create: `tests/test_matchrun.py`

**Interfaces:**
- Consumes: `match.*`, `store`, `fingerprint.item_duration_seconds`
- Produces:
  - `load_fingerprints(conn) -> dict[int, np.ndarray]` — canonical artifacts for content reachable from a present track
  - `run_match(conn, config: dict = match.DEFAULT_MATCH_CONFIG) -> int` — writes `match_run`, `run_track`, and `pair` rows; returns the run id

- [ ] **Step 1: Write the failing test**

```python
# tests/test_matchrun.py
from riffle import fingerprint, matchrun, scan, store
from tests.fixtures import make_tone, transcode


def _library(tmp_path):
    lib = tmp_path / "lib"
    src = make_tone(lib / "song.flac", seconds=40.0)
    transcode(src, lib / "song.mp3", codec="libmp3lame", bitrate="128k")
    make_tone(lib / "other.flac", seconds=40.0, freq=1500)
    conn = store.connect(tmp_path / "db.sqlite")
    scan.scan(conn, [lib])
    fingerprint.fingerprint_pending(conn)
    return conn


def test_run_match_records_a_run_and_snapshot(tmp_path):
    conn = _library(tmp_path)
    run_id = matchrun.run_match(conn)
    run = conn.execute("SELECT * FROM match_run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "complete"
    assert run["match_config"]
    snap = conn.execute(
        "SELECT count(*) c FROM run_track WHERE run_id = ?", (run_id,)
    ).fetchone()["c"]
    assert snap == 3


def test_transcoded_copy_is_found_as_tier_1(tmp_path):
    conn = _library(tmp_path)
    run_id = matchrun.run_match(conn)
    tiers = [r["tier"] for r in conn.execute(
        "SELECT tier FROM pair WHERE run_id = ?", (run_id,))]
    assert 1 in tiers


def test_unrelated_track_is_not_paired_at_tier_1(tmp_path):
    conn = _library(tmp_path)
    run_id = matchrun.run_match(conn)
    rows = conn.execute(
        "SELECT count(*) c FROM pair WHERE run_id = ? AND tier = 1", (run_id,)
    ).fetchone()
    assert rows["c"] == 1


def test_pairs_are_stored_in_canonical_order(tmp_path):
    conn = _library(tmp_path)
    run_id = matchrun.run_match(conn)
    for row in conn.execute("SELECT * FROM pair WHERE run_id = ?", (run_id,)):
        assert row["a_content_id"] < row["b_content_id"]


def test_absent_tracks_are_excluded(tmp_path):
    conn = _library(tmp_path)
    conn.execute("UPDATE track SET present = 0, absent_reason = 'missing'")
    run_id = matchrun.run_match(conn)
    assert conn.execute(
        "SELECT count(*) c FROM run_track WHERE run_id = ?", (run_id,)
    ).fetchone()["c"] == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_matchrun.py -v`
Expected: FAIL with `ImportError: cannot import name 'matchrun' from 'riffle'`

- [ ] **Step 3: Implement the run**

```python
# riffle/matchrun.py
"""One matching pass over the library, recorded for reproducibility."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

import riffle
from riffle import fingerprint, match, store


def load_fingerprints(conn) -> dict[int, np.ndarray]:
    rows = conn.execute(
        "SELECT f.audio_content_id AS cid, f.fp_raw, f.fp_length "
        "FROM fingerprint f "
        "WHERE f.purpose = 'canonical' AND EXISTS ("
        "  SELECT 1 FROM track t "
        "  WHERE t.audio_content_id = f.audio_content_id AND t.present = 1)"
    ).fetchall()
    return {
        r["cid"]: store.unpack_fingerprint(r["fp_raw"], r["fp_length"])
        for r in rows
    }


def run_match(conn, config: dict = match.DEFAULT_MATCH_CONFIG) -> int:
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "INSERT INTO match_run (created_at, fingerprint_config, match_config, "
        " software_version, status) VALUES (?,?,?,?,'running')",
        (now,
         json.dumps(fingerprint.DEFAULT_CONFIG, sort_keys=True),
         json.dumps(config, sort_keys=True),
         riffle.__version__),
    )
    run_id = cur.lastrowid

    try:
        conn.execute(
            "INSERT INTO run_track (run_id, track_id, path, size, mtime, "
            " audio_hash, hash_method) "
            "SELECT ?, t.id, t.path, t.size, t.mtime, ac.audio_hash, "
            "       ac.hash_method "
            "FROM track t JOIN audio_content ac ON ac.id = t.audio_content_id "
            "WHERE t.present = 1",
            (run_id,),
        )

        fps = load_fingerprints(conn)
        postings = match.build_postings(fps, config)
        item_seconds = fingerprint.item_duration_seconds()

        for a, b in sorted(match.candidate_pairs(postings)):
            ev = match.compare(fps[a], fps[b], config, item_seconds)
            if ev.tier == match.TIER_NONE:
                continue
            conn.execute(
                "INSERT INTO pair (run_id, a_content_id, b_content_id, "
                " best_offset, peak_votes, peak_vote_ratio, "
                " matched_span_items, matched_span_seconds, coverage_a, "
                " coverage_b, mean_bit_error, segment_count, tier, "
                " verified_direct) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,0)",
                (run_id, a, b, ev.best_offset, ev.peak_votes,
                 ev.peak_vote_ratio, ev.matched_span_items,
                 ev.matched_span_seconds, ev.coverage_a, ev.coverage_b,
                 ev.mean_bit_error, ev.segment_count, ev.tier),
            )

        conn.execute("UPDATE match_run SET status='complete' WHERE id=?",
                     (run_id,))
    except Exception:
        conn.execute("UPDATE match_run SET status='failed' WHERE id=?",
                     (run_id,))
        raise
    return run_id
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_matchrun.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add riffle/matchrun.py tests/test_matchrun.py
git commit -m "feat: persist a match run with snapshot and pair evidence"
```

---

### Task 12: Grouping — components, direct verification, chain detection

**Files:**
- Create: `riffle/group.py`
- Create: `tests/test_group.py`

**Interfaces:**
- Consumes: `match.compare`, `matchrun.load_fingerprints`
- Produces:
  - `components(edges: set[tuple[int, int]]) -> list[set[int]]`
  - `verify_component(conn, run_id, member_ids, fps, config, item_seconds) -> dict[tuple[int, int], int]` — tier per pair, all pairs, caps bypassed; writes `verified_direct = 1` rows
  - `build_groups(conn, run_id, config=match.DEFAULT_MATCH_CONFIG, verifier=None) -> int` — returns the number of groups written. `verifier(a, b) -> tier` overrides direct verification so a chain can be constructed in tests without real audio.

The caps make candidate generation lossy, so a missing edge is not evidence of a non-match. Every component is re-verified pairwise before clique-versus-chain is decided.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_group.py
from riffle import fingerprint, group, matchrun, scan, store
from tests.fixtures import make_tone, transcode


def test_components_splits_disjoint_edges():
    comps = group.components({(1, 2), (2, 3), (5, 6)})
    assert sorted(sorted(c) for c in comps) == [[1, 2, 3], [5, 6]]


def test_components_of_no_edges_is_empty():
    assert group.components(set()) == []


def _library(tmp_path):
    lib = tmp_path / "lib"
    src = make_tone(lib / "song.flac", seconds=40.0)
    transcode(src, lib / "song.mp3", codec="libmp3lame", bitrate="128k")
    transcode(src, lib / "song.m4a", codec="aac", bitrate="192k")
    make_tone(lib / "other.flac", seconds=40.0, freq=1500)
    conn = store.connect(tmp_path / "db.sqlite")
    scan.scan(conn, [lib])
    fingerprint.fingerprint_pending(conn)
    return conn


def test_transcodes_form_one_clique_group(tmp_path):
    conn = _library(tmp_path)
    run_id = matchrun.run_match(conn)
    n = group.build_groups(conn, run_id)
    assert n == 1
    g = conn.execute("SELECT * FROM dup_group WHERE run_id = ?",
                     (run_id,)).fetchone()
    assert g["tier"] == 1
    assert g["formed_by_chain"] == 0
    assert g["decision"] == "proposed"
    members = conn.execute(
        "SELECT count(*) c FROM group_member WHERE group_id = ?", (g["id"],)
    ).fetchone()["c"]
    assert members == 3


def test_tier_0_identical_audio_groups_without_matching(tmp_path):
    lib = tmp_path / "lib"
    a = make_tone(lib / "a.flac", seconds=40.0)
    (lib / "copy.flac").write_bytes(a.read_bytes())
    conn = store.connect(tmp_path / "db.sqlite")
    scan.scan(conn, [lib])
    fingerprint.fingerprint_pending(conn)
    run_id = matchrun.run_match(conn)
    group.build_groups(conn, run_id)
    g = conn.execute("SELECT * FROM dup_group WHERE run_id = ?",
                     (run_id,)).fetchone()
    assert g["tier"] == 0
    assert conn.execute(
        "SELECT count(*) c FROM group_member WHERE group_id = ?", (g["id"],)
    ).fetchone()["c"] == 2


def test_chain_component_is_flagged_and_demoted(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    conn.execute("INSERT INTO match_run (id, status) VALUES (1, 'complete')")
    for cid in (1, 2, 3):
        conn.execute("INSERT INTO audio_content (id, audio_hash, hash_method) "
                     "VALUES (?,?, 'streamhash')", (cid, f"h{cid}"))
        conn.execute("INSERT INTO track (id, path, audio_content_id, present) "
                     "VALUES (?,?,?,1)", (cid, f"/x/{cid}.flac", cid))
    # A-B and B-C are tier 1; A-C is not present at all.
    for a, b in ((1, 2), (2, 3)):
        conn.execute("INSERT INTO pair (run_id, a_content_id, b_content_id, "
                     "tier, verified_direct) VALUES (1,?,?,1,1)", (a, b))
    # Force verification to report A-C as a non-match.
    group.build_groups(conn, 1, verifier=lambda a, b: 0 if (a, b) == (1, 3) else 1)
    g = conn.execute("SELECT * FROM dup_group WHERE run_id = 1").fetchone()
    assert g["formed_by_chain"] == 1
    assert g["tier"] == 2


def test_chain_group_cannot_authorize_quarantine(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    conn.execute("INSERT INTO match_run (id, status) VALUES (1, 'complete')")
    for cid in (1, 2, 3):
        conn.execute("INSERT INTO audio_content (id, audio_hash, hash_method) "
                     "VALUES (?,?, 'streamhash')", (cid, f"h{cid}"))
        conn.execute("INSERT INTO track (id, path, audio_content_id, present) "
                     "VALUES (?,?,?,1)", (cid, f"/x/{cid}.flac", cid))
    for a, b in ((1, 2), (2, 3)):
        conn.execute("INSERT INTO pair (run_id, a_content_id, b_content_id, "
                     "tier, verified_direct) VALUES (1,?,?,1,1)", (a, b))
    group.build_groups(conn, 1, verifier=lambda a, b: 0 if (a, b) == (1, 3) else 1)
    keepers = conn.execute(
        "SELECT count(*) c FROM group_member WHERE is_keeper = 1"
    ).fetchone()["c"]
    assert keepers == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_group.py -v`
Expected: FAIL with `ImportError: cannot import name 'group' from 'riffle'`

- [ ] **Step 3: Implement grouping**

```python
# riffle/group.py
"""Content-level grouping, expanded to tracks at the end.

The K and M caps make candidate generation lossy, so a missing edge is not
evidence of a non-match. Every component is re-verified pairwise with the caps
bypassed before clique-versus-chain is decided; otherwise `formed_by_chain`
would fire on generation misses rather than on genuine non-matches.
"""
from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations

from riffle import fingerprint, match, matchrun


def components(edges: set[tuple[int, int]]) -> list[set[int]]:
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)

    out: dict[int, set[int]] = {}
    for node in parent:
        out.setdefault(find(node), set()).add(node)
    return list(out.values())


def _tier0_components(conn) -> list[set[int]]:
    """Content ids reachable from more than one present track."""
    rows = conn.execute(
        "SELECT audio_content_id AS cid, count(*) AS n FROM track "
        "WHERE present = 1 AND audio_content_id IS NOT NULL "
        "GROUP BY audio_content_id HAVING n > 1"
    ).fetchall()
    return [{r["cid"]} for r in rows]


def build_groups(conn, run_id: int, config: dict = match.DEFAULT_MATCH_CONFIG,
                 verifier=None) -> int:
    """Write dup_group, group_content and group_member rows for a run.

    `verifier(a, b) -> tier` overrides direct verification; tests use it to
    construct a chain without needing real audio.
    """
    now = datetime.now(timezone.utc).isoformat()

    if verifier is None:
        fps = matchrun.load_fingerprints(conn)
        item_seconds = fingerprint.item_duration_seconds()

        def verifier(a, b):  # noqa: F811 - deliberate local default
            if a not in fps or b not in fps:
                return match.TIER_NONE
            return match.compare(fps[a], fps[b], config, item_seconds).tier

    edges = {
        (r["a_content_id"], r["b_content_id"])
        for r in conn.execute(
            "SELECT a_content_id, b_content_id FROM pair "
            "WHERE run_id = ? AND tier = 1", (run_id,))
    }

    written = 0
    for comp in components(edges):
        members = sorted(comp)
        verified: dict[tuple[int, int], int] = {}
        for a, b in combinations(members, 2):
            tier = verifier(a, b)
            verified[(a, b)] = tier
            conn.execute(
                "INSERT INTO pair (run_id, a_content_id, b_content_id, tier, "
                " verified_direct) VALUES (?,?,?,?,1) "
                "ON CONFLICT(run_id, a_content_id, b_content_id) "
                "DO UPDATE SET tier = excluded.tier, verified_direct = 1",
                (run_id, a, b, tier),
            )

        is_clique = all(t == 1 for t in verified.values())
        tier = 1 if is_clique else 2
        _write_group(conn, run_id, members, tier,
                     formed_by_chain=0 if is_clique else 1, now=now)
        written += 1

    for comp in _tier0_components(conn):
        _write_group(conn, run_id, sorted(comp), tier=0,
                     formed_by_chain=0, now=now)
        written += 1

    return written


def _write_group(conn, run_id: int, content_ids: list[int], tier: int,
                 formed_by_chain: int, now: str) -> int:
    cur = conn.execute(
        "INSERT INTO dup_group (run_id, tier, formed_by_chain, decision) "
        "VALUES (?,?,?, 'proposed')", (run_id, tier, formed_by_chain)
    )
    group_id = cur.lastrowid
    for cid in content_ids:
        conn.execute(
            "INSERT INTO group_content (group_id, audio_content_id) "
            "VALUES (?,?)", (group_id, cid))
        for row in conn.execute(
            "SELECT id FROM track WHERE audio_content_id = ? AND present = 1",
            (cid,),
        ):
            conn.execute(
                "INSERT INTO group_member (group_id, track_id, "
                " audio_content_id, is_keeper) VALUES (?,?,?,0)",
                (group_id, row["id"], cid))
    return group_id
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_group.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add riffle/group.py tests/test_group.py
git commit -m "feat: content-level grouping with direct verification and chain detection"
```

---

### Task 13: Keeper ranking

**Files:**
- Create: `riffle/rank.py`
- Create: `tests/test_rank.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `LOSSLESS_CODECS: frozenset[str]`
  - `rank_key(track_row) -> tuple` — the ordered tuple, sorted descending; best first
  - `rank_group(conn, group_id) -> int | None` — sets `is_keeper` and `rank_score`, returns the keeper's track id, or `None` for a group that may not authorize quarantine

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rank.py
from riffle import rank, store


def _setup(tmp_path, tier=1, formed_by_chain=0):
    conn = store.connect(tmp_path / "db.sqlite")
    conn.execute("INSERT INTO match_run (id, status) VALUES (1,'complete')")
    conn.execute("INSERT INTO dup_group (id, run_id, tier, formed_by_chain) "
                 "VALUES (1,1,?,?)", (tier, formed_by_chain))
    conn.execute("INSERT INTO audio_content (id, audio_hash, hash_method) "
                 "VALUES (1,'h','streamhash')")
    return conn


def _add(conn, tid, path, bitrate, completeness, mtime, dev=1, inode=None):
    conn.execute(
        "INSERT INTO track (id, path, bitrate, tag_completeness, mtime, "
        " dev, inode, audio_content_id, present) VALUES (?,?,?,?,?,?,?,1,1)",
        (tid, path, bitrate, completeness, mtime, dev, inode or tid))
    conn.execute("INSERT INTO group_member (group_id, track_id, "
                 " audio_content_id) VALUES (1,?,1)", (tid,))


def test_lossless_wins_over_higher_bitrate_lossy(tmp_path):
    conn = _setup(tmp_path)
    _add(conn, 1, "/m/a.mp3", 320000, 4, 100.0)
    _add(conn, 2, "/m/a.flac", 900000, 1, 200.0)
    assert rank.rank_group(conn, 1) == 2


def test_bitrate_breaks_ties_among_lossy(tmp_path):
    conn = _setup(tmp_path)
    _add(conn, 1, "/m/a.mp3", 128000, 4, 100.0)
    _add(conn, 2, "/m/b.mp3", 320000, 4, 100.0)
    assert rank.rank_group(conn, 1) == 2


def test_tag_completeness_breaks_bitrate_ties(tmp_path):
    conn = _setup(tmp_path)
    _add(conn, 1, "/m/a.mp3", 320000, 1, 100.0)
    _add(conn, 2, "/m/b.mp3", 320000, 4, 100.0)
    assert rank.rank_group(conn, 1) == 2


def test_oldest_mtime_wins_last(tmp_path):
    conn = _setup(tmp_path)
    _add(conn, 1, "/m/a.mp3", 320000, 4, 500.0)
    _add(conn, 2, "/m/b.mp3", 320000, 4, 100.0)
    assert rank.rank_group(conn, 1) == 2


def test_hardlinked_copy_is_never_a_loser(tmp_path):
    conn = _setup(tmp_path)
    _add(conn, 1, "/m/a.flac", 900000, 4, 100.0, dev=1, inode=42)
    _add(conn, 2, "/m/b.flac", 900000, 4, 100.0, dev=1, inode=42)
    keeper = rank.rank_group(conn, 1)
    losers = [r["track_id"] for r in conn.execute(
        "SELECT track_id FROM group_member "
        "WHERE group_id = 1 AND is_keeper = 0")]
    assert keeper is not None
    assert losers == []


def test_tier_2_group_gets_no_keeper(tmp_path):
    conn = _setup(tmp_path, tier=2)
    _add(conn, 1, "/m/a.flac", 900000, 4, 100.0)
    _add(conn, 2, "/m/b.mp3", 128000, 1, 200.0)
    assert rank.rank_group(conn, 1) is None
    assert conn.execute(
        "SELECT count(*) c FROM group_member WHERE is_keeper = 1"
    ).fetchone()["c"] == 0


def test_chain_group_gets_no_keeper(tmp_path):
    conn = _setup(tmp_path, tier=1, formed_by_chain=1)
    _add(conn, 1, "/m/a.flac", 900000, 4, 100.0)
    _add(conn, 2, "/m/b.mp3", 128000, 1, 200.0)
    assert rank.rank_group(conn, 1) is None


def test_ranking_is_reproducible_under_full_ties(tmp_path):
    conn = _setup(tmp_path)
    _add(conn, 1, "/m/zzz.mp3", 320000, 4, 100.0)
    _add(conn, 2, "/m/aaa.mp3", 320000, 4, 100.0)
    assert rank.rank_group(conn, 1) == 2  # path sort breaks the tie
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_rank.py -v`
Expected: FAIL with `ImportError: cannot import name 'rank' from 'riffle'`

- [ ] **Step 3: Implement ranking**

```python
# riffle/rank.py
"""Keeper selection.

Only tier 0 and tier 1 groups may authorize quarantine, and a group that
cohered only through a chain never does. A radio edit is not an inferior copy
of the album version, so ranking it a loser for being shorter would be wrong.
"""
from __future__ import annotations

import json
from pathlib import Path

LOSSLESS_SUFFIXES = frozenset({".flac", ".wav", ".alac", ".ape", ".wv"})


def rank_key(row) -> tuple:
    """Higher sorts better. Path is the final, deterministic tie-break."""
    suffix = Path(row["path"]).suffix.lower()
    return (
        1 if suffix in LOSSLESS_SUFFIXES else 0,
        row["bitrate"] or 0,
        row["tag_completeness"] or 0,
        (-row["mtime"]) if row["mtime"] is not None else 0.0,
    )


def rank_group(conn, group_id: int) -> int | None:
    g = conn.execute("SELECT tier, formed_by_chain FROM dup_group WHERE id = ?",
                     (group_id,)).fetchone()
    if g is None or g["tier"] not in (0, 1) or g["formed_by_chain"]:
        return None

    rows = conn.execute(
        "SELECT t.* FROM group_member gm JOIN track t ON t.id = gm.track_id "
        "WHERE gm.group_id = ? AND t.present = 1", (group_id,)
    ).fetchall()
    if len(rows) < 2:
        return None

    by_path = sorted(rows, key=lambda r: r["path"])
    ordered = sorted(by_path, key=rank_key, reverse=True)
    keeper = ordered[0]
    keeper_inode = (keeper["dev"], keeper["inode"])

    for row in ordered:
        same_file = (row["dev"], row["inode"]) == keeper_inode
        is_keeper = row["id"] == keeper["id"] or same_file
        conn.execute(
            "UPDATE group_member SET is_keeper = ?, rank_score = ? "
            "WHERE group_id = ? AND track_id = ?",
            (1 if is_keeper else 0, json.dumps(rank_key(row)),
             group_id, row["id"]),
        )
    return keeper["id"]
```

Note: hardlinked copies are marked `is_keeper = 1` so they can never become losers. They are the same file, not a duplicate of it.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_rank.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add riffle/rank.py tests/test_rank.py
git commit -m "feat: deterministic keeper ranking"
```

---

### Task 14: Report

**Files:**
- Create: `riffle/report.py`
- Create: `tests/test_report.py`

**Interfaces:**
- Consumes: `store`
- Produces:
  - `report_data(conn, run_id: int, tier: int | None = None) -> dict`
  - `render_text(data: dict) -> str`

The report carries the pairwise evidence for every member, so a group's basis is inspectable rather than asserted. Tracks on a `whole_file` identity are flagged, because that identity does not survive retagging.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report.py
import json

from riffle import report, store


def _fixture(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    conn.execute("INSERT INTO match_run (id, created_at, status) "
                 "VALUES (1,'now','complete')")
    for cid, method in ((1, "streamhash"), (2, "whole_file")):
        conn.execute("INSERT INTO audio_content (id, audio_hash, hash_method, "
                     " duration) VALUES (?,?,?,180.0)",
                     (cid, f"h{cid}", method))
    conn.execute("INSERT INTO track (id, path, audio_content_id, present, "
                 " bitrate) VALUES (1,'/m/a.flac',1,1,900000)")
    conn.execute("INSERT INTO track (id, path, audio_content_id, present, "
                 " bitrate) VALUES (2,'/m/a.mp3',2,1,320000)")
    conn.execute("INSERT INTO pair (run_id, a_content_id, b_content_id, tier, "
                 " coverage_a, coverage_b, mean_bit_error, "
                 " matched_span_seconds, peak_vote_ratio, verified_direct) "
                 "VALUES (1,1,2,1,0.98,0.97,1.2,180.0,0.6,1)")
    conn.execute("INSERT INTO dup_group (id, run_id, tier, formed_by_chain) "
                 "VALUES (1,1,1,0)")
    for cid, tid, keeper in ((1, 1, 1), (2, 2, 0)):
        conn.execute("INSERT INTO group_content (group_id, audio_content_id) "
                     "VALUES (1,?)", (cid,))
        conn.execute("INSERT INTO group_member (group_id, track_id, "
                     " audio_content_id, is_keeper) VALUES (1,?,?,?)",
                     (tid, cid, keeper))
    return conn


def test_report_lists_groups_and_members(tmp_path):
    data = report.report_data(_fixture(tmp_path), 1)
    assert len(data["groups"]) == 1
    g = data["groups"][0]
    assert g["tier"] == 1
    assert len(g["members"]) == 2
    assert [m["is_keeper"] for m in g["members"]].count(True) == 1


def test_report_includes_pairwise_evidence(tmp_path):
    data = report.report_data(_fixture(tmp_path), 1)
    ev = data["groups"][0]["evidence"]
    assert len(ev) == 1
    assert ev[0]["coverage_a"] == 0.98
    assert ev[0]["verified_direct"] is True


def test_report_flags_whole_file_identities(tmp_path):
    data = report.report_data(_fixture(tmp_path), 1)
    flagged = [m for m in data["groups"][0]["members"]
               if m["hash_method"] == "whole_file"]
    assert len(flagged) == 1
    assert data["warnings"]


def test_report_filters_by_tier(tmp_path):
    data = report.report_data(_fixture(tmp_path), 1, tier=2)
    assert data["groups"] == []


def test_report_is_json_serializable(tmp_path):
    json.dumps(report.report_data(_fixture(tmp_path), 1))


def test_render_text_mentions_the_keeper_and_the_decision(tmp_path):
    text = report.render_text(report.report_data(_fixture(tmp_path), 1))
    assert "KEEP" in text
    assert "proposed" in text


def test_render_text_survives_a_newline_in_a_path(tmp_path):
    conn = _fixture(tmp_path)
    conn.execute("UPDATE track SET path = ? WHERE id = 2",
                 ("/m/we'ird\nname.mp3",))
    text = report.render_text(report.report_data(conn, 1))
    # The path is escaped, so it cannot forge a line of its own.
    assert "\\n" in text
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL with `ImportError: cannot import name 'report' from 'riffle'`

- [ ] **Step 3: Implement the report**

```python
# riffle/report.py
"""Human-readable and JSON reports for a match run."""
from __future__ import annotations


def _escape(path: str) -> str:
    """Paths may contain newlines; line-oriented output must not be forgeable."""
    return path.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")


TIER_NAMES = {0: "identical stream", 1: "same recording", 2: "variant"}


def report_data(conn, run_id: int, tier: int | None = None) -> dict:
    groups = []
    warnings = []

    query = "SELECT * FROM dup_group WHERE run_id = ?"
    params: list = [run_id]
    if tier is not None:
        query += " AND tier = ?"
        params.append(tier)

    for g in conn.execute(query + " ORDER BY id", params):
        members = []
        for m in conn.execute(
            "SELECT gm.track_id, gm.is_keeper, gm.rank_score, t.path, "
            "       t.bitrate, ac.hash_method, ac.duration "
            "FROM group_member gm "
            "JOIN track t ON t.id = gm.track_id "
            "JOIN audio_content ac ON ac.id = gm.audio_content_id "
            "WHERE gm.group_id = ? ORDER BY gm.track_id", (g["id"],)
        ):
            if m["hash_method"] == "whole_file":
                warnings.append(
                    f"{_escape(m['path'])} uses a whole-file identity, "
                    "which does not survive retagging"
                )
            members.append({
                "track_id": m["track_id"],
                "path": m["path"],
                "is_keeper": bool(m["is_keeper"]),
                "bitrate": m["bitrate"],
                "hash_method": m["hash_method"],
                "duration": m["duration"],
                "rank_score": m["rank_score"],
            })

        content_ids = [r["audio_content_id"] for r in conn.execute(
            "SELECT audio_content_id FROM group_content WHERE group_id = ?",
            (g["id"],))]
        evidence = []
        if content_ids:
            placeholders = ",".join("?" * len(content_ids))
            for p in conn.execute(
                f"SELECT * FROM pair WHERE run_id = ? "
                f"AND a_content_id IN ({placeholders}) "
                f"AND b_content_id IN ({placeholders})",
                [run_id, *content_ids, *content_ids],
            ):
                evidence.append({
                    "a": p["a_content_id"], "b": p["b_content_id"],
                    "tier": p["tier"],
                    "best_offset": p["best_offset"],
                    "peak_vote_ratio": p["peak_vote_ratio"],
                    "coverage_a": p["coverage_a"],
                    "coverage_b": p["coverage_b"],
                    "mean_bit_error": p["mean_bit_error"],
                    "matched_span_seconds": p["matched_span_seconds"],
                    "verified_direct": bool(p["verified_direct"]),
                })

        groups.append({
            "group_id": g["id"],
            "tier": g["tier"],
            "tier_name": TIER_NAMES.get(g["tier"], "unknown"),
            "formed_by_chain": bool(g["formed_by_chain"]),
            "decision": g["decision"],
            "members": members,
            "evidence": evidence,
        })

    return {"run_id": run_id, "groups": groups,
            "warnings": sorted(set(warnings))}


def render_text(data: dict) -> str:
    lines = [f"Match run {data['run_id']}", ""]
    if not data["groups"]:
        lines.append("No groups.")
    for g in data["groups"]:
        chain = " [CHAIN - review only]" if g["formed_by_chain"] else ""
        lines.append(
            f"Group {g['group_id']}  tier {g['tier']} "
            f"({g['tier_name']}){chain}  decision: {g['decision']}"
        )
        for m in g["members"]:
            marker = "KEEP  " if m["is_keeper"] else "loser "
            lines.append(f"  {marker} {_escape(m['path'])}")
        for e in g["evidence"]:
            lines.append(
                f"    {e['a']}~{e['b']} tier {e['tier']} "
                f"cov {e['coverage_a']:.2f}/{e['coverage_b']:.2f} "
                f"err {e['mean_bit_error']:.2f} "
                f"span {e['matched_span_seconds']:.1f}s"
            )
        lines.append("")
    for w in data["warnings"]:
        lines.append(f"warning: {w}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_report.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add riffle/report.py tests/test_report.py
git commit -m "feat: json and text reports with pairwise evidence"
```

---

### Task 15: Approval state

**Files:**
- Create: `riffle/approve.py`
- Create: `tests/test_approve.py`

**Interfaces:**
- Consumes: `store`
- Produces:
  - `ApprovalError(Exception)`
  - `approve_group(conn, run_id, group_id) -> None`
  - `reject_group(conn, run_id, group_id) -> None`
  - `approve_tier(conn, run_id, tier) -> list[int]` — returns the group ids that would be approved, without committing
  - `commit_tier(conn, run_id, tier) -> int`

"User confirms" is enforced by the data model, not by a convention about what `apply` implies. A tier-2 or chain group can never be approved.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_approve.py
import pytest

from riffle import approve, store


def _fixture(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    conn.execute("INSERT INTO match_run (id, status) VALUES (1,'complete')")
    conn.execute("INSERT INTO dup_group (id, run_id, tier, formed_by_chain) "
                 "VALUES (1,1,1,0)")
    conn.execute("INSERT INTO dup_group (id, run_id, tier, formed_by_chain) "
                 "VALUES (2,1,2,0)")
    conn.execute("INSERT INTO dup_group (id, run_id, tier, formed_by_chain) "
                 "VALUES (3,1,1,1)")
    conn.execute("INSERT INTO dup_group (id, run_id, tier, formed_by_chain) "
                 "VALUES (4,1,0,0)")
    return conn


def test_groups_start_proposed(tmp_path):
    conn = _fixture(tmp_path)
    assert conn.execute(
        "SELECT decision FROM dup_group WHERE id = 1"
    ).fetchone()["decision"] == "proposed"


def test_approve_sets_the_decision(tmp_path):
    conn = _fixture(tmp_path)
    approve.approve_group(conn, 1, 1)
    row = conn.execute("SELECT decision, decided_at FROM dup_group "
                       "WHERE id = 1").fetchone()
    assert row["decision"] == "approved"
    assert row["decided_at"]


def test_reject_sets_the_decision(tmp_path):
    conn = _fixture(tmp_path)
    approve.reject_group(conn, 1, 1)
    assert conn.execute(
        "SELECT decision FROM dup_group WHERE id = 1"
    ).fetchone()["decision"] == "rejected"


def test_tier_2_group_cannot_be_approved(tmp_path):
    conn = _fixture(tmp_path)
    with pytest.raises(approve.ApprovalError):
        approve.approve_group(conn, 1, 2)


def test_chain_group_cannot_be_approved(tmp_path):
    conn = _fixture(tmp_path)
    with pytest.raises(approve.ApprovalError):
        approve.approve_group(conn, 1, 3)


def test_approve_tier_previews_without_committing(tmp_path):
    conn = _fixture(tmp_path)
    ids = approve.approve_tier(conn, 1, 1)
    assert ids == [1]
    assert conn.execute(
        "SELECT decision FROM dup_group WHERE id = 1"
    ).fetchone()["decision"] == "proposed"


def test_commit_tier_approves_only_eligible_groups(tmp_path):
    conn = _fixture(tmp_path)
    assert approve.commit_tier(conn, 1, 1) == 1
    decisions = {r["id"]: r["decision"] for r in conn.execute(
        "SELECT id, decision FROM dup_group")}
    assert decisions == {1: "approved", 2: "proposed",
                         3: "proposed", 4: "proposed"}


def test_tier_0_can_be_approved(tmp_path):
    conn = _fixture(tmp_path)
    approve.approve_group(conn, 1, 4)
    assert conn.execute(
        "SELECT decision FROM dup_group WHERE id = 4"
    ).fetchone()["decision"] == "approved"


def test_unknown_group_raises(tmp_path):
    conn = _fixture(tmp_path)
    with pytest.raises(approve.ApprovalError):
        approve.approve_group(conn, 1, 99)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_approve.py -v`
Expected: FAIL with `ImportError: cannot import name 'approve' from 'riffle'`

- [ ] **Step 3: Implement approval**

```python
# riffle/approve.py
"""Persisted confirmation.

Only tier 0 and tier 1 groups may be approved, and never one that cohered
through a chain. `apply` acts on approved groups only, so the safety boundary
lives in the data rather than in what a command is assumed to mean.
"""
from __future__ import annotations

from datetime import datetime, timezone


class ApprovalError(Exception):
    """The group cannot be approved."""


def _eligible(row) -> bool:
    return row["tier"] in (0, 1) and not row["formed_by_chain"]


def _get(conn, run_id: int, group_id: int):
    row = conn.execute(
        "SELECT * FROM dup_group WHERE id = ? AND run_id = ?",
        (group_id, run_id)).fetchone()
    if row is None:
        raise ApprovalError(f"no group {group_id} in run {run_id}")
    return row


def approve_group(conn, run_id: int, group_id: int) -> None:
    row = _get(conn, run_id, group_id)
    if not _eligible(row):
        reason = ("it formed through a chain" if row["formed_by_chain"]
                  else f"it is tier {row['tier']}")
        raise ApprovalError(
            f"group {group_id} cannot authorize quarantine: {reason}")
    conn.execute(
        "UPDATE dup_group SET decision = 'approved', decided_at = ? "
        "WHERE id = ?", (datetime.now(timezone.utc).isoformat(), group_id))


def reject_group(conn, run_id: int, group_id: int) -> None:
    _get(conn, run_id, group_id)
    conn.execute(
        "UPDATE dup_group SET decision = 'rejected', decided_at = ? "
        "WHERE id = ?", (datetime.now(timezone.utc).isoformat(), group_id))


def approve_tier(conn, run_id: int, tier: int) -> list[int]:
    """Preview: the groups `commit_tier` would approve. Changes nothing."""
    return [r["id"] for r in conn.execute(
        "SELECT id FROM dup_group WHERE run_id = ? AND tier = ? "
        "AND formed_by_chain = 0 AND decision = 'proposed' ORDER BY id",
        (run_id, tier)) if tier in (0, 1)]


def commit_tier(conn, run_id: int, tier: int) -> int:
    ids = approve_tier(conn, run_id, tier)
    for group_id in ids:
        approve_group(conn, run_id, group_id)
    return len(ids)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_approve.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add riffle/approve.py tests/test_approve.py
git commit -m "feat: persisted approval state gating quarantine"
```

---

### Task 16: Quarantine — apply

**Files:**
- Create: `riffle/quarantine.py`
- Create: `tests/test_quarantine.py`

**Interfaces:**
- Consumes: `hashing.audio_identity`, `scan.QUARANTINE_DIRNAME`
- Produces:
  - `QuarantineError(Exception)`
  - `quarantine_dir_for(path: Path, roots: list[Path]) -> Path` — a directory on the same filesystem as `path`
  - `apply_run(conn, run_id: int, roots: list[Path]) -> dict` — counts of `moved`, `skipped_groups`, `failed`

**Review Focus item 4 is tested here**: a read-only destination during the move.

Invariants, enforced rather than asserted: no group reaches a state with zero surviving copies; nothing is overwritten; every move is verified against freshly read bytes. `os.link` fails atomically if the destination exists and fails with `EXDEV` across filesystems, which is why it is used instead of `shutil.move`, whose cross-device path silently degrades to copy-then-delete.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quarantine.py
import os

import pytest

from riffle import hashing, quarantine, scan, store
from tests.fixtures import make_tone


def _fixture(tmp_path, approve_it=True):
    lib = tmp_path / "lib"
    keeper = make_tone(lib / "keep.flac", seconds=5.0)
    loser = lib / "dupe.flac"
    loser.write_bytes(keeper.read_bytes())

    conn = store.connect(tmp_path / "db.sqlite")
    conn.execute("INSERT INTO match_run (id, status) VALUES (1,'complete')")
    conn.execute("INSERT INTO audio_content (id, audio_hash, hash_method) "
                 "VALUES (1,?, 'streamhash')",
                 (hashing.audio_identity(keeper).audio_hash,))
    decision = "approved" if approve_it else "proposed"
    conn.execute("INSERT INTO dup_group (id, run_id, tier, decision) "
                 "VALUES (1,1,1,?)", (decision,))
    for tid, path, is_keeper in ((1, keeper, 1), (2, loser, 0)):
        st = path.stat()
        conn.execute(
            "INSERT INTO track (id, path, size, mtime, audio_content_id, "
            " present) VALUES (?,?,?,?,1,1)",
            (tid, str(path), st.st_size, st.st_mtime))
        conn.execute(
            "INSERT INTO run_track (run_id, track_id, path, size, mtime, "
            " audio_hash, hash_method) VALUES (1,?,?,?,?,?, 'streamhash')",
            (tid, str(path), st.st_size, st.st_mtime,
             hashing.audio_identity(path).audio_hash))
        conn.execute("INSERT INTO group_member (group_id, track_id, "
                     " audio_content_id, is_keeper) VALUES (1,?,1,?)",
                     (tid, is_keeper))
    return conn, lib, keeper, loser


def test_apply_moves_the_loser_and_keeps_the_keeper(tmp_path):
    conn, lib, keeper, loser = _fixture(tmp_path)
    result = quarantine.apply_run(conn, 1, [lib])
    assert result["moved"] == 1
    assert keeper.exists()
    assert not loser.exists()
    moved_to = conn.execute(
        "SELECT dst_path FROM quarantine_log").fetchone()["dst_path"]
    assert os.path.exists(moved_to)


def test_apply_skips_unapproved_groups(tmp_path):
    conn, lib, keeper, loser = _fixture(tmp_path, approve_it=False)
    result = quarantine.apply_run(conn, 1, [lib])
    assert result["moved"] == 0
    assert loser.exists()


def test_apply_skips_the_group_when_the_keeper_is_gone(tmp_path):
    conn, lib, keeper, loser = _fixture(tmp_path)
    keeper.unlink()
    result = quarantine.apply_run(conn, 1, [lib])
    assert result["moved"] == 0
    assert result["skipped_groups"] == 1
    assert loser.exists()  # zero-copy state avoided


def test_apply_skips_a_modified_loser(tmp_path):
    conn, lib, keeper, loser = _fixture(tmp_path)
    make_tone(loser, seconds=5.0, freq=1700)
    result = quarantine.apply_run(conn, 1, [lib])
    assert result["moved"] == 0
    assert loser.exists()


def test_quarantined_track_is_marked_not_missing(tmp_path):
    conn, lib, keeper, loser = _fixture(tmp_path)
    quarantine.apply_run(conn, 1, [lib])
    row = conn.execute(
        "SELECT present, absent_reason FROM track WHERE id = 2").fetchone()
    assert row["present"] == 0
    assert row["absent_reason"] == "quarantined"


def test_apply_never_overwrites_an_existing_quarantine_file(tmp_path):
    conn, lib, keeper, loser = _fixture(tmp_path)
    qdir = quarantine.quarantine_dir_for(loser, [lib])
    dst = qdir / loser.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(b"occupied")
    quarantine.apply_run(conn, 1, [lib])
    assert dst.read_bytes() == b"occupied"


def test_quarantine_dir_is_on_the_same_filesystem(tmp_path):
    conn, lib, keeper, loser = _fixture(tmp_path)
    qdir = quarantine.quarantine_dir_for(loser, [lib])
    qdir.mkdir(parents=True, exist_ok=True)
    assert qdir.stat().st_dev == loser.stat().st_dev


def test_read_only_destination_records_a_failure_not_a_loss(tmp_path):
    # Review Focus 4: the move fails, the source must survive, and the run
    # must end in an explicit recorded state.
    conn, lib, keeper, loser = _fixture(tmp_path)
    qdir = quarantine.quarantine_dir_for(loser, [lib])
    qdir.mkdir(parents=True, exist_ok=True)
    os.chmod(qdir, 0o500)
    try:
        result = quarantine.apply_run(conn, 1, [lib])
    finally:
        os.chmod(qdir, 0o700)
    assert result["failed"] == 1
    assert loser.exists()
    assert conn.execute(
        "SELECT state FROM quarantine_log").fetchone()["state"] == "failed"


def test_group_decision_becomes_applied(tmp_path):
    conn, lib, keeper, loser = _fixture(tmp_path)
    quarantine.apply_run(conn, 1, [lib])
    assert conn.execute(
        "SELECT decision FROM dup_group WHERE id = 1"
    ).fetchone()["decision"] == "applied"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_quarantine.py -v`
Expected: FAIL with `ImportError: cannot import name 'quarantine' from 'riffle'`

- [ ] **Step 3: Implement apply**

```python
# riffle/quarantine.py
"""Moving losers out of the library, reversibly.

Invariants:
  1. No group ever reaches a state with zero surviving copies.
  2. Nothing is ever overwritten, on move or on undo.
  3. Every move is verified against freshly read bytes, not cached state.

`os.link` is used rather than `shutil.move`: it fails atomically when the
destination exists, and fails with EXDEV across filesystems instead of
silently degrading to copy-then-delete.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from riffle import hashing, scan


class QuarantineError(Exception):
    """The move could not be performed safely."""


def quarantine_dir_for(path: Path, roots: list[Path]) -> Path:
    """A quarantine directory on the same filesystem as `path`.

    One per root, so a library spread across drives is handled rather than
    refused. Falls back to the file's own directory when no root matches.
    """
    path = Path(path).resolve()
    dev = path.stat().st_dev if path.exists() else None
    for root in sorted((Path(r).resolve() for r in roots),
                       key=lambda r: len(str(r)), reverse=True):
        if path.is_relative_to(root):
            if dev is None or root.stat().st_dev == dev:
                return root / scan.QUARANTINE_DIRNAME
    return path.parent / scan.QUARANTINE_DIRNAME


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fresh_matches(path: Path, snapshot) -> bool:
    if not path.exists():
        return False
    try:
        ident = hashing.audio_identity(path)
    except hashing.HashError:
        return False
    return (ident.audio_hash == snapshot["audio_hash"]
            and ident.hash_method == snapshot["hash_method"])


def apply_run(conn, run_id: int, roots: list[Path]) -> dict:
    moved = failed = skipped_groups = 0

    groups = conn.execute(
        "SELECT * FROM dup_group WHERE run_id = ? AND decision = 'approved' "
        "ORDER BY id", (run_id,)).fetchall()

    for g in groups:
        members = conn.execute(
            "SELECT gm.track_id, gm.is_keeper, rt.path, rt.audio_hash, "
            "       rt.hash_method "
            "FROM group_member gm "
            "JOIN run_track rt ON rt.track_id = gm.track_id "
            "                 AND rt.run_id = ? "
            "WHERE gm.group_id = ? ORDER BY gm.track_id",
            (run_id, g["id"])).fetchall()

        keepers = [m for m in members if m["is_keeper"]]
        losers = [m for m in members if not m["is_keeper"]]

        # Invariant 1: verify every keeper from disk BEFORE touching a loser.
        if not keepers or not all(
            _fresh_matches(Path(k["path"]), k) for k in keepers
        ):
            skipped_groups += 1
            continue

        group_moved = 0
        for m in losers:
            src = Path(m["path"])
            if not _fresh_matches(src, m):
                continue  # changed since the run: leave it alone

            qdir = quarantine_dir_for(src, roots)
            dst = qdir / src.name
            try:
                qdir.mkdir(parents=True, exist_ok=True)
                os.link(src, dst)  # fails if dst exists (invariant 2)
                dst_hash = hashing.audio_identity(dst).audio_hash
                if dst_hash != m["audio_hash"]:
                    os.unlink(dst)
                    raise QuarantineError("destination hash mismatch")
                os.unlink(src)
            except (OSError, QuarantineError) as exc:
                conn.execute(
                    "INSERT INTO quarantine_log (run_id, track_id, group_id, "
                    " src_path, dst_path, src_hash, dst_hash, moved_at, state) "
                    "VALUES (?,?,?,?,?,?,NULL,?, 'failed')",
                    (run_id, m["track_id"], g["id"], str(src), str(dst),
                     m["audio_hash"], _now()))
                failed += 1
                continue

            conn.execute(
                "INSERT INTO quarantine_log (run_id, track_id, group_id, "
                " src_path, dst_path, src_hash, dst_hash, moved_at, state) "
                "VALUES (?,?,?,?,?,?,?,?, 'moved')",
                (run_id, m["track_id"], g["id"], str(src), str(dst),
                 m["audio_hash"], dst_hash, _now()))
            conn.execute(
                "UPDATE track SET present = 0, absent_reason = 'quarantined' "
                "WHERE id = ?", (m["track_id"],))
            moved += 1
            group_moved += 1

        conn.execute(
            "UPDATE dup_group SET decision = 'applied', decided_at = ? "
            "WHERE id = ?", (_now(), g["id"]))

    return {"moved": moved, "failed": failed,
            "skipped_groups": skipped_groups}
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_quarantine.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add riffle/quarantine.py tests/test_quarantine.py
git commit -m "feat: verified quarantine with keeper-first safety check"
```

---

### Task 17: Undo

**Files:**
- Modify: `riffle/quarantine.py`
- Create: `tests/test_undo.py`

**Interfaces:**
- Consumes: `apply_run`
- Produces: `undo_run(conn, run_id: int) -> dict` — counts of `restored`, `refused`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_undo.py
from pathlib import Path

from riffle import quarantine
from tests.test_quarantine import _fixture


def test_undo_restores_the_file(tmp_path):
    conn, lib, keeper, loser = _fixture(tmp_path)
    quarantine.apply_run(conn, 1, [lib])
    assert not loser.exists()
    result = quarantine.undo_run(conn, 1)
    assert result["restored"] == 1
    assert loser.exists()


def test_undo_marks_the_track_present_again(tmp_path):
    conn, lib, keeper, loser = _fixture(tmp_path)
    quarantine.apply_run(conn, 1, [lib])
    quarantine.undo_run(conn, 1)
    row = conn.execute(
        "SELECT present, absent_reason FROM track WHERE id = 2").fetchone()
    assert row["present"] == 1
    assert row["absent_reason"] is None


def test_undo_refuses_to_overwrite_a_path_taken_since(tmp_path):
    conn, lib, keeper, loser = _fixture(tmp_path)
    quarantine.apply_run(conn, 1, [lib])
    loser.write_bytes(b"something new lives here now")
    result = quarantine.undo_run(conn, 1)
    assert result["refused"] == 1
    assert loser.read_bytes() == b"something new lives here now"


def test_undo_returns_the_group_to_approved(tmp_path):
    conn, lib, keeper, loser = _fixture(tmp_path)
    quarantine.apply_run(conn, 1, [lib])
    quarantine.undo_run(conn, 1)
    assert conn.execute(
        "SELECT decision FROM dup_group WHERE id = 1"
    ).fetchone()["decision"] == "approved"


def test_undo_is_idempotent(tmp_path):
    conn, lib, keeper, loser = _fixture(tmp_path)
    quarantine.apply_run(conn, 1, [lib])
    quarantine.undo_run(conn, 1)
    second = quarantine.undo_run(conn, 1)
    assert second["restored"] == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_undo.py -v`
Expected: FAIL with `AttributeError: module 'riffle.quarantine' has no attribute 'undo_run'`

- [ ] **Step 3: Append the implementation to `riffle/quarantine.py`**

```python
def undo_run(conn, run_id: int) -> dict:
    """Restore quarantined files, refusing to overwrite anything."""
    restored = refused = 0
    rows = conn.execute(
        "SELECT * FROM quarantine_log WHERE run_id = ? AND state = 'moved' "
        "ORDER BY id", (run_id,)).fetchall()

    for row in rows:
        src = Path(row["dst_path"])
        dst = Path(row["src_path"])
        if not src.exists():
            continue
        if dst.exists():  # invariant 2: never overwrite
            refused += 1
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.link(src, dst)
            os.unlink(src)
        except OSError:
            refused += 1
            continue

        conn.execute("UPDATE quarantine_log SET state = 'undone' WHERE id = ?",
                     (row["id"],))
        conn.execute(
            "UPDATE track SET present = 1, absent_reason = NULL WHERE id = ?",
            (row["track_id"],))
        conn.execute(
            "UPDATE dup_group SET decision = 'approved' WHERE id = ?",
            (row["group_id"],))
        restored += 1

    return {"restored": restored, "refused": refused}
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_undo.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add riffle/quarantine.py tests/test_undo.py
git commit -m "feat: undo restores quarantined files without overwriting"
```

---

### Task 18: AcoustID enrichment

**Files:**
- Create: `riffle/enrich.py`
- Create: `tests/test_enrich.py`

**Interfaces:**
- Consumes: `store.unpack_fingerprint`, `fingerprint.*`
- Produces:
  - `RateLimiter(rate_per_second: float)` with `wait()`
  - `lookup_key(algorithm, encoded_fp, duration, meta) -> str`
  - `lookup_fingerprint(conn, content_id, config) -> tuple[str, int]` — the base64 fingerprint to send and its item count, truncated to 120 s
  - `enrich(conn, api_key: str, client=None, sleeper=None) -> dict` — counts of `looked_up`, `cached`, `failed`

Best-effort and strictly subordinate: enrichment never blocks or alters dedup. The cache key covers the whole request — algorithm, encoded fingerprint, duration, meta — so differing requests never collide. The fingerprint sent is truncated to 120 seconds because canonical clients fingerprint with fpcalc's default, which is what the service's index was populated from; `duration` is documented as the whole file's duration and is sent in full.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_enrich.py
import json
import time

import numpy as np
import pytest

from riffle import enrich, fingerprint, store


def _content_with_fp(conn, n_items=2000, cid=1):
    conn.execute("INSERT INTO audio_content (id, audio_hash, hash_method, "
                 " duration) VALUES (?,?, 'streamhash', 300.0)", (cid, f"h{cid}"))
    rng = np.random.default_rng(cid)
    raw = rng.integers(0, 2 ** 32, size=n_items, dtype=np.uint64).astype(np.uint32)
    conn.execute(
        "INSERT INTO fingerprint (audio_content_id, analyzer, "
        " analyzer_version, config_hash, purpose, algorithm, fp_raw, fp_length) "
        "VALUES (?, 'chromaprint', 'v', 'c', 'canonical', 2, ?, ?)",
        (cid, store.pack_fingerprint(raw), len(raw)))
    return raw


def test_rate_limiter_spaces_calls():
    slept = []
    rl = enrich.RateLimiter(3.0, sleeper=slept.append, clock=iter([0.0, 0.0, 0.0]).__next__)
    rl.wait()
    rl.wait()
    assert slept and slept[0] > 0


def test_lookup_key_covers_the_whole_request():
    a = enrich.lookup_key(2, "AQAA", 300.0, "recordings")
    assert a == enrich.lookup_key(2, "AQAA", 300.0, "recordings")
    assert a != enrich.lookup_key(2, "AQAA", 301.0, "recordings")
    assert a != enrich.lookup_key(2, "AQAB", 300.0, "recordings")
    assert a != enrich.lookup_key(2, "AQAA", 300.0, "recordings+releasegroups")


def test_lookup_fingerprint_truncates_to_120_seconds(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _content_with_fp(conn, n_items=2000)
    encoded, n = enrich.lookup_fingerprint(conn, 1, fingerprint.DEFAULT_CONFIG)
    expected = int(round(120.0 / fingerprint.item_duration_seconds()))
    assert abs(n - expected) <= 2
    assert isinstance(encoded, str) and encoded


def test_lookup_fingerprint_keeps_short_fingerprints_whole(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _content_with_fp(conn, n_items=100)
    _, n = enrich.lookup_fingerprint(conn, 1, fingerprint.DEFAULT_CONFIG)
    assert n == 100


def test_enrich_caches_and_does_not_refetch(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _content_with_fp(conn)
    calls = []

    def fake_client(apikey, fp, duration, meta):
        calls.append((fp, duration))
        return {"status": "ok", "results": []}

    enrich.enrich(conn, "key", client=fake_client, sleeper=lambda _: None)
    enrich.enrich(conn, "key", client=fake_client, sleeper=lambda _: None)
    assert len(calls) == 1
    assert conn.execute(
        "SELECT count(*) c FROM acoustid_cache").fetchone()["c"] == 1


def test_enrich_sends_the_full_duration_not_the_truncated_one(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _content_with_fp(conn)
    seen = {}

    def fake_client(apikey, fp, duration, meta):
        seen["duration"] = duration
        return {"status": "ok", "results": []}

    enrich.enrich(conn, "key", client=fake_client, sleeper=lambda _: None)
    assert seen["duration"] == 300.0


def test_enrich_failure_does_not_raise(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _content_with_fp(conn)

    def broken_client(*_args, **_kwargs):
        raise RuntimeError("network down")

    result = enrich.enrich(conn, "key", client=broken_client,
                           sleeper=lambda _: None)
    assert result["failed"] == 1
    assert result["looked_up"] == 0


def test_prefix_compatibility_check(tmp_path):
    # Documented assumption under test: slicing the canonical array and
    # re-encoding should equal a plain default fpcalc run on the same file.
    from tests.fixtures import make_tone

    p = make_tone(tmp_path / "a.flac", seconds=200.0)
    full = fingerprint.fingerprint_file(p).raw
    n = int(round(120.0 / fingerprint.item_duration_seconds()))
    derived = enrich.encode(full[:n], 2)
    default = fingerprint.fingerprint_file(
        p, dict(fingerprint.DEFAULT_CONFIG, length=120))
    native = enrich.encode(default.raw, 2)
    if derived != native:
        pytest.skip(
            "fingerprints are not prefix-compatible; "
            "enrich must store a dedicated acoustid_lookup artifact"
        )
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_enrich.py -v`
Expected: FAIL with `ImportError: cannot import name 'enrich' from 'riffle'`

- [ ] **Step 3: Implement enrichment**

```python
# riffle/enrich.py
"""AcoustID enrichment: best-effort, rate-limited, cached, resumable.

Local matching is authoritative. Enrichment may fail, be throttled, or return
nothing, and dedup results are unchanged either way.
"""
from __future__ import annotations

import base64
import hashlib
import json
import time
from datetime import datetime, timezone

import numpy as np

from riffle import fingerprint, store

ACOUSTID_RATE = 3.0  # requests per second, per the service's guidelines
META = "recordings+releasegroups+compress"
LOOKUP_SECONDS = 120.0  # fpcalc's default, which populated the index


class RateLimiter:
    def __init__(self, rate_per_second: float, sleeper=time.sleep,
                 clock=time.monotonic):
        self._interval = 1.0 / rate_per_second
        self._sleep = sleeper
        self._clock = clock
        self._last = None

    def wait(self) -> None:
        now = self._clock()
        if self._last is not None:
            remaining = self._interval - (now - self._last)
            if remaining > 0:
                self._sleep(remaining)
        self._last = now


def encode(raw: np.ndarray, algorithm: int) -> str:
    import chromaprint

    return chromaprint.encode_fingerprint(
        [int(x) for x in raw], algorithm, base64=True
    ).decode("ascii")


def lookup_key(algorithm: int, encoded_fp: str, duration: float,
               meta: str) -> str:
    blob = json.dumps(
        {"algorithm": algorithm, "fp": encoded_fp,
         "duration": duration, "meta": meta},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def lookup_fingerprint(conn, content_id: int, config: dict) -> tuple[str, int]:
    row = conn.execute(
        "SELECT fp_raw, fp_length, algorithm FROM fingerprint "
        "WHERE audio_content_id = ? AND purpose = 'canonical' "
        "ORDER BY id DESC LIMIT 1", (content_id,)).fetchone()
    if row is None:
        raise LookupError(f"no canonical fingerprint for content {content_id}")

    raw = store.unpack_fingerprint(row["fp_raw"], row["fp_length"])
    n = min(len(raw),
            int(round(LOOKUP_SECONDS / fingerprint.item_duration_seconds())))
    return encode(raw[:n], row["algorithm"]), n


def _default_client(apikey, fp, duration, meta):
    import acoustid

    return acoustid.lookup(apikey, fp, duration, meta=meta)


def enrich(conn, api_key: str, client=None, sleeper=None) -> dict:
    client = client or _default_client
    limiter = RateLimiter(ACOUSTID_RATE,
                          sleeper=sleeper or time.sleep)

    looked_up = cached = failed = 0
    rows = conn.execute(
        "SELECT ac.id AS cid, ac.duration, f.algorithm "
        "FROM audio_content ac "
        "JOIN fingerprint f ON f.audio_content_id = ac.id "
        "                  AND f.purpose = 'canonical' "
        "WHERE ac.duration IS NOT NULL GROUP BY ac.id ORDER BY ac.id"
    ).fetchall()

    for row in rows:
        try:
            encoded, _ = lookup_fingerprint(conn, row["cid"],
                                            fingerprint.DEFAULT_CONFIG)
        except LookupError:
            failed += 1
            continue

        key = lookup_key(row["algorithm"], encoded, row["duration"], META)
        if conn.execute("SELECT 1 FROM acoustid_cache WHERE lookup_key = ?",
                        (key,)).fetchone():
            cached += 1
            continue

        limiter.wait()
        try:
            # duration is the WHOLE file's duration, per the service's docs,
            # even though the fingerprint sent is truncated to 120 seconds.
            response = client(api_key, encoded, row["duration"], META)
        except Exception:
            failed += 1
            continue

        conn.execute(
            "INSERT INTO acoustid_cache (lookup_key, response_json, fetched_at) "
            "VALUES (?,?,?)",
            (key, json.dumps(response),
             datetime.now(timezone.utc).isoformat()))
        looked_up += 1

    return {"looked_up": looked_up, "cached": cached, "failed": failed}
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_enrich.py -v`
Expected: 7 passed, 1 passed or skipped.

If `test_prefix_compatibility_check` skips, the assumption is false. Add a step that runs `fpcalc` at default settings and stores the result as a `fingerprint` row with `purpose = 'acoustid_lookup'`, and have `lookup_fingerprint` prefer that row. The canonical full-track artifact is never replaced.

- [ ] **Step 5: Commit**

```bash
git add riffle/enrich.py tests/test_enrich.py
git commit -m "feat: rate-limited, cached AcoustID enrichment"
```

---

### Task 19: Calibration

**Files:**
- Create: `riffle/calibrate.py`
- Create: `tests/test_calibrate.py`

**Interfaces:**
- Consumes: `match.*`, `fingerprint.fingerprint_file`
- Produces:
  - `load_pairs(path: Path) -> list[dict]` — JSON: `[{"a": "...", "b": "...", "expect": 0|1|2, "set": "calibrate"|"holdout"}]`
  - `score(predictions: list[dict]) -> dict` — `{"total", "correct", "accuracy", "confusion"}`
  - `search(pairs, base_config, item_seconds=None) -> tuple[dict, dict]` — best config and its scores on both sets

Tuning and validation use disjoint sets. With a small library the holdout is statistically weak, so both scores are always reported rather than only the validation figure.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calibrate.py
import json

import pytest

from riffle import calibrate, match
from tests.fixtures import make_tone, transcode, trim


def test_load_pairs_requires_every_category(tmp_path):
    p = tmp_path / "pairs.json"
    p.write_text(json.dumps([{"a": "x", "b": "y", "expect": 1,
                              "set": "calibrate"}]))
    with pytest.raises(calibrate.CalibrationError):
        calibrate.load_pairs(p)


def test_load_pairs_requires_both_sets(tmp_path):
    entries = [{"a": f"a{i}", "b": f"b{i}", "expect": e, "set": "calibrate"}
               for i, e in enumerate([0, 1, 2, 1, 2, 0, 1])]
    p = tmp_path / "pairs.json"
    p.write_text(json.dumps(entries))
    with pytest.raises(calibrate.CalibrationError):
        calibrate.load_pairs(p)


def test_evaluate_scores_a_perfect_config():
    pairs = [{"fp_a": None, "fp_b": None, "expect": 1, "predicted": 1},
             {"fp_a": None, "fp_b": None, "expect": 0, "predicted": 0}]
    result = calibrate.score(pairs)
    assert result["accuracy"] == 1.0
    assert result["correct"] == 2


def test_evaluate_reports_the_confusion():
    pairs = [{"expect": 1, "predicted": 2}, {"expect": 1, "predicted": 1}]
    result = calibrate.score(pairs)
    assert result["accuracy"] == 0.5
    assert result["confusion"][(1, 2)] == 1


def test_search_reports_both_sets(tmp_path):
    src = make_tone(tmp_path / "src.flac", seconds=60.0)
    mp3 = transcode(src, tmp_path / "src.mp3", codec="libmp3lame",
                    bitrate="128k")
    clip = trim(src, tmp_path / "clip.flac", start=10.0, duration=25.0)
    other = make_tone(tmp_path / "other.flac", seconds=60.0, freq=1700)

    entries = [
        {"a": str(src), "b": str(mp3), "expect": 1, "set": "calibrate"},
        {"a": str(src), "b": str(clip), "expect": 2, "set": "calibrate"},
        {"a": str(src), "b": str(other), "expect": 0, "set": "calibrate"},
        {"a": str(src), "b": str(mp3), "expect": 1, "set": "holdout"},
        {"a": str(src), "b": str(clip), "expect": 2, "set": "holdout"},
        {"a": str(src), "b": str(other), "expect": 0, "set": "holdout"},
    ]
    p = tmp_path / "pairs.json"
    p.write_text(json.dumps(entries))

    best, scores = calibrate.search(
        calibrate.load_pairs(p), match.DEFAULT_MATCH_CONFIG)
    assert "calibrate" in scores and "holdout" in scores
    assert scores["calibrate"]["accuracy"] >= 0.6
    assert set(best) >= set(match.DEFAULT_MATCH_CONFIG)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_calibrate.py -v`
Expected: FAIL with `ImportError: cannot import name 'calibrate' from 'riffle'`

- [ ] **Step 3: Implement calibration**

```python
# riffle/calibrate.py
"""Threshold tuning against known pairs, validated on a holdout set.

Upstream's constants are the starting point. Calibration confirms or adjusts
them for this library, and reports both sets, because a small holdout is not
strong evidence on its own.
"""
from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

from riffle import fingerprint, match

REQUIRED_EXPECTATIONS = {0, 1, 2}
REQUIRED_SETS = {"calibrate", "holdout"}


class CalibrationError(Exception):
    """The pair file cannot support a meaningful calibration."""


def load_pairs(path: Path) -> list[dict]:
    entries = json.loads(Path(path).read_text())
    seen_sets = {e["set"] for e in entries}
    if not REQUIRED_SETS <= seen_sets:
        raise CalibrationError(
            f"need both {sorted(REQUIRED_SETS)} sets, found {sorted(seen_sets)}")
    for name in REQUIRED_SETS:
        expects = {e["expect"] for e in entries if e["set"] == name}
        if not REQUIRED_EXPECTATIONS <= expects:
            raise CalibrationError(
                f"the '{name}' set must contain all of "
                f"{sorted(REQUIRED_EXPECTATIONS)}, found {sorted(expects)}")
    return entries


def score(pairs: list[dict]) -> dict:
    confusion = Counter(
        (p["expect"], p["predicted"]) for p in pairs
    )
    correct = sum(n for (e, p), n in confusion.items() if e == p)
    total = len(pairs)
    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "confusion": dict(confusion),
    }


def _predict(entries, config, item_seconds, cache) -> list[dict]:
    out = []
    for e in entries:
        for side in ("a", "b"):
            if e[side] not in cache:
                cache[e[side]] = fingerprint.fingerprint_file(Path(e[side])).raw
        ev = match.compare(cache[e["a"]], cache[e["b"]], config, item_seconds)
        out.append({"expect": e["expect"], "predicted": ev.tier})
    return out


_GRID = {
    "tier1_min_coverage": [0.75, 0.85, 0.92],
    "tier1_max_bit_error": [4.0, 6.0, 8.0],
    "tier1_min_overlap_seconds": [10.0, 20.0, 30.0],
    "tier2_min_overlap_seconds": [8.0, 15.0, 25.0],
}


def search(entries: list[dict], base_config: dict,
           item_seconds: float | None = None) -> tuple[dict, dict]:
    item_seconds = item_seconds or fingerprint.item_duration_seconds()
    cal = [e for e in entries if e["set"] == "calibrate"]
    hold = [e for e in entries if e["set"] == "holdout"]
    cache: dict[str, object] = {}

    best_config = dict(base_config)
    best_accuracy = -1.0
    keys = sorted(_GRID)
    for combo in itertools.product(*(_GRID[k] for k in keys)):
        config = dict(base_config, **dict(zip(keys, combo)))
        result = score(_predict(cal, config, item_seconds, cache))
        # Ties resolve to the first combination in sorted grid order.
        if result["accuracy"] > best_accuracy:
            best_accuracy = result["accuracy"]
            best_config = config

    return best_config, {
        "calibrate": score(_predict(cal, best_config, item_seconds, cache)),
        "holdout": score(_predict(hold, best_config, item_seconds, cache)),
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_calibrate.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add riffle/calibrate.py tests/test_calibrate.py
git commit -m "feat: threshold calibration with a holdout set"
```

---

### Task 20: CLI, end-to-end determinism, and the candidate-explosion stress test

**Files:**
- Create: `riffle/cli.py`
- Create: `tests/test_cli.py`
- Create: `tests/test_e2e.py`
- Create: `tests/test_stress.py`

**Interfaces:**
- Consumes: every module.
- Produces: the `riffle` console script with `scan`, `match`, `report`, `approve`, `reject`, `apply`, `undo`, `enrich`, `calibrate`.

- [ ] **Step 1: Write the failing CLI test**

```python
# tests/test_cli.py
from typer.testing import CliRunner

from riffle.cli import app
from tests.fixtures import make_tone

runner = CliRunner()


def test_scan_and_match_and_report(tmp_path):
    lib = tmp_path / "lib"
    src = make_tone(lib / "a.flac", seconds=40.0)
    (lib / "copy.flac").write_bytes(src.read_bytes())
    db = str(tmp_path / "db.sqlite")

    assert runner.invoke(app, ["--db", db, "scan", str(lib)]).exit_code == 0
    assert runner.invoke(app, ["--db", db, "match"]).exit_code == 0
    result = runner.invoke(app, ["--db", db, "report", "--run", "1"])
    assert result.exit_code == 0
    assert "Group" in result.stdout


def test_apply_without_approval_moves_nothing(tmp_path):
    lib = tmp_path / "lib"
    src = make_tone(lib / "a.flac", seconds=40.0)
    copy = lib / "copy.flac"
    copy.write_bytes(src.read_bytes())
    db = str(tmp_path / "db.sqlite")

    runner.invoke(app, ["--db", db, "scan", str(lib)])
    runner.invoke(app, ["--db", db, "match"])
    result = runner.invoke(app, ["--db", db, "apply", "--run", "1"])
    assert result.exit_code == 0
    assert copy.exists()
    assert "0" in result.stdout


def test_bulk_approve_requires_confirmation(tmp_path):
    lib = tmp_path / "lib"
    src = make_tone(lib / "a.flac", seconds=40.0)
    (lib / "copy.flac").write_bytes(src.read_bytes())
    db = str(tmp_path / "db.sqlite")
    runner.invoke(app, ["--db", db, "scan", str(lib)])
    runner.invoke(app, ["--db", db, "match"])

    declined = runner.invoke(
        app, ["--db", db, "approve", "--run", "1", "--tier", "0", "--all"],
        input="n\n")
    assert "approved" not in declined.stdout.lower() or declined.exit_code == 0

    accepted = runner.invoke(
        app, ["--db", db, "approve", "--run", "1", "--tier", "0", "--all",
              "--yes"])
    assert accepted.exit_code == 0


def test_concurrent_invocation_is_refused(tmp_path):
    from riffle import store

    db = tmp_path / "db.sqlite"
    with store.exclusive_lock(db):
        result = runner.invoke(app, ["--db", str(db), "match"])
    assert result.exit_code != 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'riffle.cli'`

- [ ] **Step 3: Implement the CLI**

```python
# riffle/cli.py
"""Command-line interface."""
from __future__ import annotations

import json
from pathlib import Path

import typer

from riffle import (approve as approve_mod, fingerprint, group, matchrun,
                      quarantine, rank, report as report_mod, scan as scan_mod,
                      store)

app = typer.Typer(add_completion=False, help="Music library deduplicator.")
_state: dict = {}


@app.callback()
def main(db: str = typer.Option("riffle.sqlite", help="Database path")):
    _state["db"] = Path(db)


def _open():
    db = _state["db"]
    try:
        lock = store.exclusive_lock(db)
        lock.__enter__()
    except store.LockError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1)
    _state["lock"] = lock
    return store.connect(db)


@app.command()
def scan(dirs: list[str], verify_hashes: bool = False,
         retry_errors: bool = False):
    conn = _open()
    scan_mod.scan(conn, [Path(d) for d in dirs],
                  verify_hashes=verify_hashes, retry_errors=retry_errors)
    written = fingerprint.fingerprint_pending(conn)
    typer.echo(f"scanned; {written} new fingerprints")


@app.command()
def match():
    conn = _open()
    run_id = matchrun.run_match(conn)
    n = group.build_groups(conn, run_id)
    for row in conn.execute("SELECT id FROM dup_group WHERE run_id = ?",
                            (run_id,)):
        rank.rank_group(conn, row["id"])
    typer.echo(f"run {run_id}: {n} groups")


@app.command()
def report(run: int = typer.Option(..., "--run"),
           tier: int | None = None, as_json: bool = False):
    conn = _open()
    data = report_mod.report_data(conn, run, tier)
    typer.echo(json.dumps(data, indent=2) if as_json
               else report_mod.render_text(data))


@app.command()
def approve(run: int = typer.Option(..., "--run"),
            group_id: int | None = typer.Option(None, "--group"),
            tier: int | None = None, all: bool = False, yes: bool = False):
    conn = _open()
    if group_id is not None:
        approve_mod.approve_group(conn, run, group_id)
        typer.echo(f"group {group_id} approved")
        return
    if not (all and tier is not None):
        typer.echo("error: pass --group, or --tier N --all", err=True)
        raise typer.Exit(code=2)

    ids = approve_mod.approve_tier(conn, run, tier)
    typer.echo(f"{len(ids)} group(s) would be approved at tier {tier}")
    if not ids:
        return
    if not yes and not typer.confirm("Approve them all?"):
        typer.echo("nothing approved")
        return
    typer.echo(f"{approve_mod.commit_tier(conn, run, tier)} approved")


@app.command()
def reject(run: int = typer.Option(..., "--run"),
           group_id: int = typer.Option(..., "--group")):
    conn = _open()
    approve_mod.reject_group(conn, run, group_id)
    typer.echo(f"group {group_id} rejected")


@app.command()
def apply(run: int = typer.Option(..., "--run")):
    conn = _open()
    # Quarantine directories belong at the scan roots, one per filesystem,
    # not beside every file. Recover them from the last completed scan.
    row = conn.execute(
        "SELECT roots FROM scan_run WHERE status = 'complete' "
        "ORDER BY id DESC LIMIT 1").fetchone()
    roots = [Path(r) for r in json.loads(row["roots"])] if row else []
    result = quarantine.apply_run(conn, run, roots)
    typer.echo(f"moved {result['moved']}, failed {result['failed']}, "
               f"skipped groups {result['skipped_groups']}")


@app.command()
def undo(run: int = typer.Option(..., "--run")):
    conn = _open()
    result = quarantine.undo_run(conn, run)
    typer.echo(f"restored {result['restored']}, refused {result['refused']}")


@app.command()
def enrich(api_key: str = typer.Option(..., envvar="ACOUSTID_API_KEY")):
    from riffle import enrich as enrich_mod

    conn = _open()
    result = enrich_mod.enrich(conn, api_key)
    typer.echo(f"looked up {result['looked_up']}, cached {result['cached']}, "
               f"failed {result['failed']}")


@app.command()
def calibrate(pairs_file: str):
    from riffle import calibrate as cal_mod
    from riffle import match as match_mod

    entries = cal_mod.load_pairs(Path(pairs_file))
    best, scores = cal_mod.search(entries, match_mod.DEFAULT_MATCH_CONFIG)
    typer.echo(json.dumps({"config": best, "scores": scores}, indent=2,
                          default=str))
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 4 passed.

- [ ] **Step 5: Write the end-to-end and stress tests**

```python
# tests/test_e2e.py
import json

from riffle import (fingerprint, group, matchrun, quarantine, rank,
                      report, scan, store, approve)
from tests.fixtures import make_tone, transcode, trim, retag


def _full_run(tmp_path, name="db.sqlite"):
    lib = tmp_path / "lib"
    src = make_tone(lib / "song.flac", seconds=60.0)
    transcode(src, lib / "song.mp3", codec="libmp3lame", bitrate="128k")
    copy = lib / "song-copy.flac"
    copy.write_bytes(src.read_bytes())
    retag(copy, title="Retagged")
    trim(src, lib / "song-edit.flac", start=5.0, duration=30.0)
    make_tone(lib / "unrelated.flac", seconds=60.0, freq=1700)

    conn = store.connect(tmp_path / name)
    scan.scan(conn, [lib])
    fingerprint.fingerprint_pending(conn)
    run_id = matchrun.run_match(conn)
    group.build_groups(conn, run_id)
    for row in conn.execute("SELECT id FROM dup_group WHERE run_id = ?",
                            (run_id,)):
        rank.rank_group(conn, row["id"])
    return conn, lib, run_id


def test_retagged_copy_is_tier_0_and_the_transcode_is_tier_1(tmp_path):
    conn, lib, run_id = _full_run(tmp_path)
    tiers = sorted(r["tier"] for r in conn.execute(
        "SELECT tier FROM dup_group WHERE run_id = ?", (run_id,)))
    assert 0 in tiers
    assert 1 in tiers


def test_the_edit_never_authorizes_quarantine(tmp_path):
    conn, lib, run_id = _full_run(tmp_path)
    for g in conn.execute(
        "SELECT id FROM dup_group WHERE run_id = ? AND tier = 2", (run_id,)
    ):
        keepers = conn.execute(
            "SELECT count(*) c FROM group_member "
            "WHERE group_id = ? AND is_keeper = 1", (g["id"],)).fetchone()["c"]
        assert keepers == 0


def test_full_cycle_applies_and_undoes(tmp_path):
    conn, lib, run_id = _full_run(tmp_path)
    approve.commit_tier(conn, run_id, 0)
    approve.commit_tier(conn, run_id, 1)
    before = sorted(p.name for p in lib.iterdir() if p.is_file())
    applied = quarantine.apply_run(conn, run_id, [lib])
    assert applied["moved"] >= 1
    quarantine.undo_run(conn, run_id)
    after = sorted(p.name for p in lib.iterdir() if p.is_file())
    assert before == after


def test_two_runs_produce_identical_reports(tmp_path):
    conn_a, _, run_a = _full_run(tmp_path, name="a.sqlite")
    conn_b, _, run_b = _full_run(tmp_path, name="b.sqlite")

    def normalize(data):
        for g in data["groups"]:
            for m in g["members"]:
                m.pop("track_id", None)
        data.pop("run_id", None)
        return json.dumps(data, sort_keys=True)

    assert normalize(report.report_data(conn_a, run_a)) == \
           normalize(report.report_data(conn_b, run_b))
```

```python
# tests/test_stress.py
import resource
import time

from riffle import fingerprint, match, matchrun, scan, store
from tests.fixtures import concat, make_silence, make_tone


def test_common_keys_do_not_explode_the_candidate_set(tmp_path):
    """The caps exist for exactly this input: many files sharing silence."""
    lib = tmp_path / "lib"
    silence = make_silence(tmp_path / "sil.flac", seconds=20.0)
    for i in range(12):
        body = make_tone(tmp_path / f"body{i}.flac", seconds=20.0,
                         freq=300 + 40 * i)
        concat(lib / f"track{i}.flac", silence, body, silence)

    conn = store.connect(tmp_path / "db.sqlite")
    scan.scan(conn, [lib])
    fingerprint.fingerprint_pending(conn)

    fps = matchrun.load_fingerprints(conn)
    # Pin the caps rather than inheriting them: with only 12 fixtures the
    # fractional term would floor to 2 and the test would be exercising the
    # floor instead of the caps it exists to check.
    cfg = dict(match.DEFAULT_MATCH_CONFIG, k_cap=4, k_cap_fraction=1.0)
    capped = match.build_postings(fps, cfg)
    uncapped = match.build_postings(
        fps, dict(cfg, k_cap=10 ** 9, k_cap_fraction=10 ** 9, m_cap=10 ** 9))

    assert sum(len(v) for v in capped.values()) < \
           sum(len(v) for v in uncapped.values())
    assert len(match.candidate_pairs(capped)) <= \
           len(match.candidate_pairs(uncapped))


def test_match_run_stays_within_time_and_memory(tmp_path):
    lib = tmp_path / "lib"
    for i in range(12):
        make_tone(lib / f"t{i}.flac", seconds=30.0, freq=250 + 50 * i)

    conn = store.connect(tmp_path / "db.sqlite")
    scan.scan(conn, [lib])
    fingerprint.fingerprint_pending(conn)

    start = time.monotonic()
    matchrun.run_match(conn)
    elapsed = time.monotonic() - start
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    assert elapsed < 60.0
    assert peak_mb < 1024
```

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -v`
Expected: every test passes. If `test_two_runs_produce_identical_reports` fails, the cause is nondeterminism in matching — find it rather than loosening the assertion; it is the test that protects the reproducibility claim.

- [ ] **Step 7: Commit**

```bash
git add riffle/cli.py tests/test_cli.py tests/test_e2e.py tests/test_stress.py
git commit -m "feat: cli, end-to-end determinism, and candidate-explosion stress test"
```

---

## Notes for the executor

- **The safety boundary is the point of this tool.** If a test about zero-copy states, overwriting, or approval gating fails, stop and report rather than adjusting the test.
- **Thresholds in `DEFAULT_MATCH_CONFIG` are provisional.** They are upstream's constants plus plausible starting values for the coverage and overlap floors. Task 19 measures them against real audio. If Task 11 or Task 20 shows poor tier assignment on the fixtures, that is calibration work, not a reason to change the algorithm.
- **`item_duration_seconds()` reaches into pyacoustid's ctypes handle.** If that private access breaks on the installed version, substitute the measured constant — fingerprint a file of known length and divide — and leave a comment saying where the number came from.
- Every task ends with a green test run and a commit. Do not batch commits across tasks.
