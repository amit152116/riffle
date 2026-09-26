# Schema Optimization Design: Migration 7

**Date**: 2026-09-26
**Status**: Draft
**Scope**: Normalize confirmed duplicated data, add confirmed missing indexes, on the live schema in `riffle/store.py` (SQLite 3.45.1, currently at `SCHEMA_VERSION = 6`)

---

## 1. Method

Every finding below was verified by tracing the actual read/write path through the real pipeline code (`group.py`, `rank.py`, `report.py`, `quarantine.py`, `matchrun.py`, `approve.py`, `quality.py`, `scan.py`, `enrich.py`, `metadata.py`, `collection.py`), not inferred from column names or schema shape alone. Several things that looked like duplication on first read of `store.py` turned out to be deliberate, correct design once traced — those are documented as "confirmed NOT a problem" so a future pass doesn't re-flag them.

This is a live database with a real user's ~770-track library. No destructive changes; Migration 7 is additive/rebuild-in-place only.

**Migration mechanics — this one can't be a plain SQL string.** Every prior migration (`_MIGRATION_1` .. `_MIGRATION_6`) is a SQL string run through `conn.executescript(sql)`. Migration 7 can't follow that pattern for two reasons, both found during spec self-review, not assumed:

1. `PRAGMA foreign_keys` is a documented no-op when changed while a transaction is active — the OFF/ON toggle around the four rebuilds has to happen *outside* any `BEGIN`/`COMMIT`, which `executescript()`'s single call can't sequence against Python-level logic.
2. `PRAGMA foreign_key_check` returns violation rows like a query — but `executescript()` discards all results. Putting it inside a script means any violation is silently thrown away, not caught. To actually inspect the result and decide whether to roll back, it has to run through `conn.execute(...).fetchall()`.

The `artists_json` → `mb_artist`/`mb_recording_artist` backfill (§2.1) also needs Python-level JSON parsing per row, which no SQL string can do.

So `_MIGRATIONS` is extended to accept a callable, not just a string: `list[str | Callable[[sqlite3.Connection], None]]`. The runner in `connect()` becomes:

```python
for i, step in enumerate(_MIGRATIONS, start=1):
    if current < i:
        if callable(step):
            step(conn)
        else:
            conn.executescript(step)
        conn.execute("DELETE FROM schema_version")
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (i,))
```

Migration 7 is one callable, `_migration_7(conn)`, doing exactly what SQLite's own documented 12-step procedure prescribes, in order:

```python
def _migration_7(conn):
    conn.execute("PRAGMA foreign_keys=OFF")
    try:
        conn.execute("BEGIN")
        conn.executescript(_MIGRATION_7_DDL)   # new tables, indexes, DROP COLUMN, all 4 rebuilds -- see below
        _backfill_mb_artists(conn)             # Python: parse artists_json, populate mb_artist/mb_recording_artist
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError(f"Migration 7 left {len(violations)} dangling reference(s): {violations}")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.execute("PRAGMA foreign_keys=ON")
```

`BEGIN` is inside the `try`, not before it — so `finally`'s re-enable always runs even in the (vanishingly unlikely) case `BEGIN` itself fails, not just when the DDL/backfill/check fails.

`_MIGRATION_7_DDL` is the SQL string containing every `CREATE TABLE`, `CREATE INDEX`, `ALTER TABLE ... DROP COLUMN`, and the four rebuild sequences from §2.3 and §5 below — no `BEGIN`/`COMMIT`/`PRAGMA` statements inside it; those are all handled by `_migration_7` itself, once, around everything. `_backfill_mb_artists` uses `INSERT OR IGNORE` (for `mb_artist`, since the same artist recurs across many matches) and a plain `INSERT` guarded by the table's own `PRIMARY KEY (match_id, position)` (for `mb_recording_artist`) — both safe to re-run if `_migration_7` is retried after a rollback, since the whole function is one transaction: either every piece lands, or none does, and a retry starts from the same pre-migration state every time.

---

## 2. Confirmed duplication — fixed

### 2.1 `musicbrainz_match.artists_json` → normalized artist tables

**Problem, verified**: `metadata.py` stores an AcoustID match's artist list as a JSON string (`json.dumps([{"name":..., "mbid":...}, ...])`). The only consumer, `render_metadata`'s "Top Artists" section, does `GROUP BY artists_json` on the raw JSON string — grouping by the literal serialized text, not by artist. A collaboration track (e.g. "A.R. Rahman, Chinmayi") is credited only as that exact ensemble string; neither artist is ever credited individually, and nothing can query "every track matched to artist X" without deserializing JSON in application code first. This is a real product limitation the JSON-blob design causes, not a hypothetical one.

**Fix**:

```sql
CREATE TABLE mb_artist (
    mbid TEXT PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE mb_recording_artist (
    match_id    INTEGER NOT NULL REFERENCES musicbrainz_match(id),
    position    INTEGER NOT NULL,
    artist_mbid TEXT NOT NULL REFERENCES mb_artist(mbid),
    PRIMARY KEY (match_id, position)
);
CREATE INDEX recording_artist_mbid ON mb_recording_artist(artist_mbid);
```

`position` preserves the original artist ordering from the AcoustID response (lead artist first, etc.) — needed to reconstruct display order without relying on row insertion order.

**Migration data step**: for every existing `musicbrainz_match` row with non-NULL `artists_json`, parse it and upsert into `mb_artist` (`INSERT OR IGNORE`, since the same artist recurs across many tracks) and `mb_recording_artist`. This is a Python step run once during the migration (in `store.py`, alongside the DDL), not pure SQL — SQLite has no native JSON-array-to-rows function reliable across versions for this shape.

**`artists_json` column**: kept in place, not dropped, for one migration cycle — cheap insurance in case the backfill needs re-running or a bug surfaces. `metadata.py`'s `parse_all` is updated to write both the JSON column (unchanged, for backward compatibility during the transition) and the new normalized rows. `render_metadata`'s "Top Artists" section is rewritten to `GROUP BY artist_mbid` through the join, crediting each collaborator individually.

**Code changes**: `riffle/metadata.py` (`parse_all` writes to `mb_artist`/`mb_recording_artist`; `render_metadata` queries them instead of `artists_json` for the Top Artists section). `riffle/store.py` (Migration 7 DDL + backfill function). Tests: `tests/test_metadata.py` (new tests for the normalized write path and the per-artist "Top Artists" grouping; a regression test using a real multi-artist fixture like the existing `SAMPLE_RESPONSE` to confirm both artists get individually counted).

### 2.2 `group_member.audio_content_id` — dropped

**Problem, verified**: traced `group.py:_write_group` — every `group_member` row's `audio_content_id` is set from the exact same `cid` used to look up the `track_id` in the same loop iteration; it can never independently diverge from `track.audio_content_id`. Traced every reader: `rank.py:43` joins `audio_content` via `t.audio_content_id`, never touching the column. `report.py:30` does join through `gm.audio_content_id`, but `track_id` is already selected in the same query — the join is rewritable through `t.audio_content_id` with an identical result. `quarantine.py` never references the column at all. Confirmed: pure redundant copy, never load-bearing.

**Fix**:

```sql
ALTER TABLE group_member DROP COLUMN audio_content_id;
```

Supported directly (SQLite 3.35+, confirmed 3.45.1 here) — no table rebuild needed, since the column isn't part of a primary key, isn't unique, isn't indexed, and nothing else references it.

**Code changes**: `riffle/group.py` (`_write_group` stops inserting the column). `riffle/report.py` (line 30's join rewritten to go through `t.audio_content_id`). Tests: `tests/test_group.py`/`tests/test_report.py` — existing tests should pass unchanged if they don't assert on the column directly; grep confirms nothing does.

### 2.3 `track.tag_completeness` → generated column

**Problem, verified**: traced `scan.py:_read_tags` — `completeness` is computed in Python as `sum(1 for k in ("title","artist","album","genre") if tags[k])`, from the *same row's own* tag values, then stored as a separate `INTEGER` column written alongside them on every scan/rescan. This is a stored value that is a pure function of other columns in the same row — the textbook case for a computed column. Currently correct only because every write path remembers to recompute it; a future write path that updates a tag without recomputing completeness would silently desync it.

**Fix** — requires a full rebuild of `track` (SQLite has no `ALTER COLUMN`; a stored column can't be converted to `GENERATED ALWAYS AS` in place). Part of `_MIGRATION_7_DDL` (see the migration-mechanics note in §1 for how the `PRAGMA`/transaction bracket around this and the other three rebuilds is actually sequenced):

```sql
CREATE TABLE track_new (
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
    tag_completeness  INTEGER GENERATED ALWAYS AS (
        (tag_title  IS NOT NULL AND tag_title  != '') +
        (tag_artist IS NOT NULL AND tag_artist != '') +
        (tag_album  IS NOT NULL AND tag_album  != '') +
        (tag_genre  IS NOT NULL AND tag_genre  != '')
    ) STORED,
    last_seen_scan_id INTEGER REFERENCES scan_run(id),
    present           INTEGER NOT NULL DEFAULT 1,
    absent_reason     TEXT CHECK (absent_reason IN ('missing','quarantined')),
    scanned_at        TEXT
);

INSERT INTO track_new (id, path, size, mtime, dev, inode, nlink,
    audio_content_id, bitrate, tag_title, tag_artist, tag_album, tag_genre,
    last_seen_scan_id, present, absent_reason, scanned_at)
SELECT id, path, size, mtime, dev, inode, nlink,
    audio_content_id, bitrate, tag_title, tag_artist, tag_album, tag_genre,
    last_seen_scan_id, present, absent_reason, scanned_at
FROM track;

DROP TABLE track;
ALTER TABLE track_new RENAME TO track;

CREATE INDEX track_content ON track(audio_content_id);
CREATE INDEX track_present ON track(present);
CREATE INDEX track_inode ON track(dev, inode);
```

The `id` column is copied explicitly (not omitted), so every existing row keeps its exact rowid — every FK reference into `track` from `quality_flag`, `group_member`, `track_similarity`, etc. stays valid across the rebuild. `_migration_7`'s `PRAGMA foreign_key_check` (run once, after every rebuild below has executed) is the hard verification step — if it returns any row, the whole migration rolls back rather than silently continuing.

**Note on the boolean expression**: the original Python check is `if tags[k]` (truthy — an empty string doesn't count). The SQL expression uses `IS NOT NULL AND != ''` to match that exactly, not just `IS NOT NULL` alone (which would wrongly count a present-but-blank tag).

**Code changes**: `riffle/scan.py` (`_read_tags` stops computing `completeness`; the INSERT/UPSERT in the scan loop drops the `tag_completeness` parameter — SQLite computes and stores it automatically on every insert and update). Tests: `tests/test_scan.py` (verify a scanned track's `tag_completeness` matches expectations without the app ever setting it directly); `tests/test_store.py` (a migration test with a pre-existing `track` row, confirming the rebuild preserves `id` and existing FK references).

---

## 3. Confirmed missing indexes

Real query shapes, traced to their call sites — not general "add an index everywhere" advice.

```sql
CREATE INDEX af_bpm ON audio_features(bpm);
CREATE INDEX af_key ON audio_features(key_name);
CREATE INDEX cluster_assignment_run_cluster ON cluster_assignment(run_id, cluster_id);
CREATE INDEX acoustid_cache_content ON acoustid_cache(audio_content_id);
CREATE INDEX fingerprint_content ON fingerprint(audio_content_id);
CREATE INDEX dup_group_run_filter ON dup_group(run_id, tier, formed_by_chain, decision);
```

- `audio_features(bpm)`, `audio_features(key_name)`: `collection.py`'s `browse()` filters on `bpm_range`/`key`, and `stats()`/`render_features()` bucket by BPM range — currently a full table scan.
- `cluster_assignment(run_id, cluster_id)`: `collection.py`'s `browse(cluster=...)` filters exactly this way (fixed as part of the C1 review finding earlier this session; this index makes that fixed query fast, not just correct).
- `acoustid_cache(audio_content_id)`: `metadata.py`'s `parse_all` LEFT JOINs on this column with no index today.
- `fingerprint(audio_content_id)`: `matchrun.py`'s `load_fingerprints` and `fingerprint.py`'s `fingerprint_pending` both filter/join on it.
- `dup_group(run_id, tier, formed_by_chain, decision)`: `approve.py`'s `approve_tier` filters on all four columns together.

None of these require a table rebuild — plain `CREATE INDEX`, safe to add to a live table.

---

## 4. Verified NOT duplication — deliberately left unchanged

Documented so a future audit doesn't re-flag these:

- **`group_content` vs `group_member`**: traced `report.py:48-50` — `group_content` records audio-content membership even when *no track is currently present* for that content (every copy quarantined or missing). `group_member` structurally cannot represent that case (it's populated from `SELECT id FROM track WHERE audio_content_id = ? AND present = 1`). Both tables stay.
- **`run_track`**: traced `quarantine.py:apply_run` — this is the point-in-time snapshot of every present track's path/hash *at match time*, read back at *apply time* to detect drift (a file moved or changed in between runs). Deliberate temporal versioning, core to the quarantine safety invariant. Untouched.
- **`quality_flag.track_id`**: traced `quality.py:render_quality` — needed to answer "which file path do I show the user" for a clipping/low-volume report, a reverse lookup `audio_features` never needs to make. Correct as designed; not inconsistent with `audio_features` lacking the same column.
- **`acoustid_cache`'s two keys** (`lookup_key` PK, `audio_content_id` column): traced `enrich.py` — `lookup_key` is checked *before* the API call specifically to avoid redundant AcoustID lookups for identical fingerprint+duration+meta combinations, independent of which `audio_content_id` a track happens to have at insert time. `audio_content_id` is a legitimate later addition (Migration 4) for join convenience. Correct as designed.
- **`musicbrainz_match.release_title`/`release_mbid`**: unlike artists, a release is single-valued per match (no 1NF violation — one release per row, not an array), and nothing currently browses by MusicBrainz release. Left as plain columns.
- **BLOB-packed vectors** (`fingerprint.fp_raw`, `audio_features.mfcc_mean`, `cluster_run.scaler_params`, `cluster_centroid.centroid`): correct SQLite pattern for fixed-length numeric arrays. Unpacking into per-dimension columns would add columns with no query benefit, since nothing filters on individual vector dimensions.
- **TEXT ISO8601 timestamps** (every `*_at` column): already SQLite's own idiomatic choice — lexicographic string sort equals chronological sort, and SQLite has no native `DATETIME` type to convert *to*. Converting to INTEGER epoch would trade a working, readable format for a marginal storage saving with no query benefit at this scale.

---

## 5. Missing FK declarations — included after reconsideration

**Revised during design**: an earlier draft of this spec deferred this section, reasoning that three extra table rebuilds (`run_track`, `pair`, `quarantine_log`) were too much migration risk on a live database for a purely defensive benefit, since every write path was already traced and confirmed correct. The user clarified this riffle Python project is a first prototype informing a future Android rebuild, not a long-lived production system — the migration-risk caution that applies to a system users depend on long-term doesn't apply here in the same way. Including the rebuilds.

`run_track.track_id`, `pair.a_content_id`, `pair.b_content_id`, and `quarantine_log.run_id`/`track_id`/`group_id` are `INTEGER NOT NULL` without a `REFERENCES` clause today, unlike equivalent columns elsewhere in the schema. Each requires the same rebuild pattern as §2.3 (SQLite cannot add a `REFERENCES` constraint to an existing column in place). These three rebuilds are also part of `_MIGRATION_7_DDL`, run inside the same single transaction as `track`'s rebuild — no separate `PRAGMA`/transaction handling per table:

```sql
CREATE TABLE run_track_new (
    run_id      INTEGER NOT NULL REFERENCES match_run(id),
    track_id    INTEGER NOT NULL REFERENCES track(id),
    path        TEXT NOT NULL,
    size        INTEGER,
    mtime       REAL,
    audio_hash  TEXT,
    hash_method TEXT,
    PRIMARY KEY (run_id, track_id)
);
INSERT INTO run_track_new SELECT run_id, track_id, path, size, mtime,
    audio_hash, hash_method FROM run_track;
DROP TABLE run_track;
ALTER TABLE run_track_new RENAME TO run_track;

CREATE TABLE pair_new (
    run_id               INTEGER NOT NULL REFERENCES match_run(id),
    a_content_id         INTEGER NOT NULL REFERENCES audio_content(id),
    b_content_id         INTEGER NOT NULL REFERENCES audio_content(id),
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
INSERT INTO pair_new SELECT run_id, a_content_id, b_content_id, best_offset,
    peak_votes, peak_vote_ratio, matched_span_items, matched_span_seconds,
    coverage_a, coverage_b, mean_bit_error, segment_count, tier,
    verified_direct FROM pair;
DROP TABLE pair;
ALTER TABLE pair_new RENAME TO pair;

CREATE TABLE quarantine_log_new (
    id       INTEGER PRIMARY KEY,
    run_id   INTEGER NOT NULL REFERENCES match_run(id),
    track_id INTEGER NOT NULL REFERENCES track(id),
    group_id INTEGER NOT NULL REFERENCES dup_group(id),
    src_path TEXT NOT NULL,
    dst_path TEXT NOT NULL,
    src_hash TEXT,
    dst_hash TEXT,
    moved_at TEXT,
    state    TEXT NOT NULL CHECK (state IN ('moved','failed','undone'))
);
INSERT INTO quarantine_log_new SELECT id, run_id, track_id, group_id,
    src_path, dst_path, src_hash, dst_hash, moved_at, state
    FROM quarantine_log;
DROP TABLE quarantine_log;
ALTER TABLE quarantine_log_new RENAME TO quarantine_log;
```

`quarantine_log`'s `id` is a single-column `INTEGER PRIMARY KEY`, copied explicitly like `track.id` in §2.3, so existing rowids are preserved. `run_track` and `pair` have no surrogate `id` (their PK is the natural composite key already), so nothing external references *their* rows by id — only *they* reference outward, which is exactly what's being fixed.

With this included, no separate `integrity_check` helper is needed: every FK-shaped column in the schema now has a declared `REFERENCES`, so `PRAGMA foreign_keys=ON` (already set on every `connect()`) catches a bad reference at write time going forward, and `_migration_7`'s own `PRAGMA foreign_key_check` (§1) catches any pre-existing violation before the migration commits.

---

## 6. Migration summary

| Change | Mechanism | Rebuild required |
|---|---|---|
| `mb_artist`, `mb_recording_artist` + backfill | New tables (`_MIGRATION_7_DDL`) + Python backfill (`_backfill_mb_artists`) | No |
| Drop `group_member.audio_content_id` | `ALTER TABLE ... DROP COLUMN` | No |
| `track.tag_completeness` → generated | Full rebuild | Yes |
| `run_track.track_id` → `REFERENCES track(id)` | Full rebuild | Yes |
| `pair.a_content_id`/`b_content_id` → `REFERENCES audio_content(id)` | Full rebuild | Yes |
| `quarantine_log.run_id`/`track_id`/`group_id` → declared FKs | Full rebuild | Yes |
| 6 new indexes (§3) | `CREATE INDEX` | No |

Everything above runs as one atomic unit inside `_migration_7(conn)` (§1) — all four rebuilds, the new tables/indexes/column drop, and the Python backfill share a single `PRAGMA foreign_keys=OFF` / `BEGIN` / ... / `PRAGMA foreign_key_check` / `COMMIT` / `PRAGMA foreign_keys=ON` sequence. `SCHEMA_VERSION = 7`; `_MIGRATIONS` is extended to accept a callable alongside the existing SQL-string entries.

## 7. Testing strategy

TDD per the project's existing discipline: each schema change gets a failing test before the DDL/code change, per the pattern already used for Migrations 1-6 (`tests/test_store.py`). Specifically:

- Migration applies cleanly to a fresh DB and to a DB carrying realistic pre-Migration-7 data (a `track` row with tags, a `group_member` row, a `musicbrainz_match` row with `artists_json`, a `run_track`/`pair`/`quarantine_log` row each) — verifying every rebuild and the backfill work against non-empty tables, not just empty schema.
- `tag_completeness` computed correctly for: all four tags present, all absent, a mix, and the empty-string edge case (a tag key exists but is blank) — this is the one place the Python-vs-SQL boolean logic could diverge if not written carefully (see §2.3's note).
- `mb_artist`/`mb_recording_artist` backfill correctly reconstructs artist lists from the existing `SAMPLE_RESPONSE`-shaped test fixtures already in `tests/test_metadata.py`, including the multi-artist case.
- `group_member` inserts/reads work identically after the column drop (existing tests, run unchanged, must stay green).
- After the four rebuilds, every existing FK reference into `track`, `run_track`, `pair`, and `quarantine_log` still resolves — `PRAGMA foreign_key_check` returns no rows, both as an assertion inside the migration itself and as an explicit test.
- A test that deliberately constructs an invalid reference (e.g. a `pair` row pointing at a nonexistent `audio_content_id`) via direct SQL confirms `sqlite3.IntegrityError` is now raised where it previously wasn't — proving the new FK declarations are actually enforced, not just present as documentation.
- **Migration atomicity**: a test that monkeypatches `_backfill_mb_artists` (or another step inside `_migration_7`) to raise partway through, then confirms — on the same connection, without a second migration attempt — that `track`, `run_track`, `pair`, and `quarantine_log` are all still in their *original* (pre-Migration-7) shape (old columns present, no `_new` leftover tables, `schema_version` still reads 6), matching the atomicity-on-failure pattern already used this session for `rebuild_clusters`/`full_rebuild` (`tests/test_cluster.py::test_rebuild_clusters_is_atomic_on_failure`). A partial rebuild left on disk after a crash is exactly the failure mode this migration's transaction bracket exists to prevent — it needs its own test, not just an assumption that the `try`/`except`/`finally` in `_migration_7` works.

Full existing suite (currently 289 tests) must stay green throughout.
