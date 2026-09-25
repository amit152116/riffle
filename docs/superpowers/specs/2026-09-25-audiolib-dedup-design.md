# audiolib — Spec 1: Library Base + Duplicate Detection

Date: 2026-09-25
Status: Draft for review (revision 3)
Scope: Base layer + Subsystem A (dedup). Subsystems B (features/collections) and C (shuffle/recommendation) get their own specs.

## Context

The goal is a local music library analyzer for a personal collection of roughly 2,000–20,000 files, serving three eventual use cases:

1. Remove duplicate songs from collected music files.
2. Sort and build collections by musical features or genre.
3. Shuffle better, and recommend the next track from what has been played.

These are three subsystems over one shared base. This spec covers the base and use case 1 only. Use cases 2 and 3 share a single embedding pipeline and are deliberately deferred so their storage layout is decided when their model is chosen, not guessed at now.

The original framing was "like Shazam". Shazam's landmark-hashing algorithm is built to match a short, noisy microphone clip against a database. Deduplicating whole clean files in a local library is a different problem, and Chromaprint/AcoustID is the tool built for it. Using Chromaprint rather than hand-building a Wang-style engine was decided explicitly.

What the Shazam material contributes is the **scoring method**: offset-histogram voting. That technique is fingerprint-agnostic, and it is what makes trimmed intros, radio edits, and partial overlaps detectable without a second DSP pipeline.

**Chromaprint already implements this method internally** (see "Upstream matching algorithm"). It is not exposed in the public C API, so it is reimplemented here in numpy — but with upstream's constants and structure rather than invented ones.

## Requirements

Confirmed with the user:

| Decision | Value |
|---|---|
| Fingerprint engine | Chromaprint (`fpcalc`), not hand-built landmark hashing |
| Language | Python |
| Library size | 2,000–20,000 files |
| Duplicate definition | Same recording **and** near-variants (radio edits, trimmed intros, different masters) |
| Keeper selection | Auto-ranked, **user confirms**; losers quarantined, never deleted |
| Network | AcoustID/MusicBrainz lookups allowed, cached locally |
| Playback integration | Deferred (subsystem C) |

"User confirms" is a data-model obligation, not a convention — see "Approval".

## Verified facts

Checked against primary sources on 2026-09-25. **Each fact is labelled with the version it was checked against**, because the installable package is older than upstream master.

### Local environment

- `ffmpeg` at `/usr/bin/ffmpeg`. Muxers `hash`, `md5`, `streamhash`, `framehash`, `framemd5` available.
- `python3` 3.12.3, `uv` present.
- `fpcalc` **not installed**. `apt-cache policy libchromaprint-tools` → candidate `1.5.1-5` (pop-os noble/universe). Dependencies include `libchromaprint1`, which pyacoustid's ctypes bindings require.

### Version pin: why 1.5.1

Upstream is v1.6.1 (2026-07-28); the distro package is v1.5.1 (2021-12-23). **1.5.1 is used deliberately**, because it is the version `apt` provides and building Chromaprint from source adds a toolchain dependency this project does not otherwise need.

The relevant difference is a security fix in 1.6.1, quoted from `NEWS.txt`:

> Fixed a heap buffer overflow when decoding a fingerprint that has extra data after the encoded values. Applications that decode fingerprints from untrusted sources should update.

**This project never decodes a fingerprint from an untrusted source.** `fpcalc -raw` emits plain integers as text, so the normal path involves no Chromaprint decoding at all. `chromaprint.decode_fingerprint` is reached only in the prefix-compatibility validation test, applied to output this tool itself produced by running `fpcalc`. AcoustID lookup responses carry metadata, not fingerprints, and are never fed to the decoder.

Two obligations follow, and both are requirements rather than notes: the decoder is never applied to bytes the tool did not generate, and if a future feature needs to accept fingerprints from elsewhere — imports, submissions, a shared database — that feature must first move to Chromaprint ≥ 1.6.1.

Between 1.5.0 and 1.5.1 upstream records "no functional source code changes", so behaviour matches the 1.5.0 release notes.

### fpcalc (verified against tag **v1.5.1**, the version to be installed)

- `static double g_max_duration = 120;` — **the default fingerprints only the first 120 seconds.**
- `const size_t stream_limit = g_max_duration * reader.GetSampleRate();` guarded by `if (stream_limit > 0)` — therefore **`-length 0` means unlimited**.
- Flags in v1.5.1: `-format/-f`, `-channels/-c`, `-rate/-r`, `-length/-t`, `-chunk`, `-algorithm/-a`, `-text`, `-json`, `-plain`, `-overlap`, `-ts`, `-raw`, `-signed`, `-ignore-errors`, `-version`.
- All three facts are identical in master and v1.5.1.

### Public C API (master; symbol set is stable)

`chromaprint_get_item_duration`, `chromaprint_get_item_duration_ms`, `chromaprint_get_sample_rate`, `chromaprint_get_delay`, `chromaprint_encode_fingerprint`, `chromaprint_decode_fingerprint`, `chromaprint_hash_fingerprint`, and the fingerprinting calls.

**`FingerprintMatcher` is not exported.** The matching algorithm must be reimplemented.

Item duration comes from `chromaprint_get_item_duration() / chromaprint_get_sample_rate()` at runtime rather than being hardcoded, so `matched_span_seconds` stays correct across algorithm settings. For the default algorithm this is about 0.124 s per item, roughly 8 items per second.

### pyacoustid

`fingerprint_file(path, force_fpcalc=)`, `lookup(apikey, fp, duration)`, `parse_lookup_result(data)`, `compare_fingerprints(a, b)`; module `chromaprint` exposes `decode_fingerprint(data, base64=True) -> (uint32 array, algorithm)` and `encode_fingerprint(fingerprint, algorithm, base64=True)`.

### Web services

- AcoustID: **3 requests/second**, non-commercial, `client` API key required. `duration` is documented as "duration of the whole audio file in seconds". `meta` accepts `recordings`, `releasegroups`, `compress`, among others.
- MusicBrainz: **1 request/second per IP**, enforced by returning HTTP 503 for *all* requests once exceeded; a contactable `User-Agent` is mandatory.
- AcoustID's server index (`acoustid-index` / `fpindex`) is described upstream as an "inverted index for searching audio fingerprints (an ID plus a set of hashes); search finds fingerprint IDs whose hash set intersects the query." Set intersection over fingerprint hashes is production-proven, not novel.

## Upstream matching algorithm

From `chromaprint/src/fingerprint_matcher.{h,cpp}`. Present identically in master and v1.5.1.

### Constants actually used by `FingerprintMatcher::Match()`

| Constant | Value | Role |
|---|---|---|
| `ALIGN_BITS` | 12 | `ALIGN_STRIP(x) = x >> (32 - 12)` — alignment key is the **top 12 bits** |
| `kDefaultMatchThreshold` | 10.0 | A segment matches when its mean Hamming distance (0–32) is below this |
| Gaussian filter | sigma 8.0, order 3 | Smooths the per-item bit-error series before segmentation |
| Gradient peak | > 0.15, local max, non-adjacent | Segment boundary detection |
| Segment merge | \|score difference\| < 0.7 | Adjacent segments with similar scores are merged |

### Reference-only constants — deliberately NOT used

| Constant | Value | Status |
|---|---|---|
| `ACOUSTID_MAX_BIT_ERROR` | 2 | Defined in the file but **not referenced by `Match()`** |
| `ACOUSTID_MAX_ALIGN_OFFSET` | 120 | Likewise unused by `Match()` |
| `ACOUSTID_QUERY_START/LENGTH/BITS` | 80 / 120 / 28 | Likewise unused by `Match()` |

These belong to AcoustID's server-side query path, not to pairwise matching. **`match.py` must not impose a ±120 alignment window or a ≤2 bit-error filter.** Doing so would silently discard exactly the long-offset matches this tool exists to find. They are listed here only so that a reader who opens the upstream file does not mistake them for part of the algorithm.

### Procedure

Build `(align_key, index, source)` triples for both fingerprints, sort, and for each equal-key run emit `offset_diff = offset1 + fp2_size - offset2` into a histogram. Peaks are bins with count > 1 that are local maxima. For the best alignment, compute the per-item Hamming distance over the overlapping region, Gaussian-smooth, take the absolute gradient, and cut segments at gradient peaks. Each segment scores the mean **raw** (unsmoothed) bit count; segments below the threshold are kept, adjacent similar-scoring ones merged.

`Segment(pos1, pos2, duration, score)` is the definition of "matched span" used throughout this spec.

### Determinism: upstream's `rand()` is not reproduced

Upstream computes, in both master and v1.5.1:

```
bit_counts[i] = HammingDistance(*it1++, *it2++) + rand() * (0.001f / RAND_MAX);
```

There is no `srand()` anywhere in the file. **This spec requires deterministic output, so the jitter is omitted**, and equal values are resolved by index order using `>=` comparisons scanning low index to high.

Why omission is safe rather than merely convenient: the jitter has magnitude ≤ 0.001 against Hamming distances that are integers in 0–32. After a sigma-8 Gaussian filter the resulting gradient contribution is orders of magnitude below the 0.15 peak threshold, so it cannot create or suppress a segment boundary. Its only real effect upstream is to break exact ties in the local-maximum comparison, which a deterministic index-order rule handles without randomness.

The implementation is therefore semantically equivalent for ordinary inputs and intentionally deterministic. It does not reproduce upstream's incidental random tie-breaking, and a test asserts that repeated runs over the same fixtures produce byte-identical reports.

**The masking rationale is also corrected.** An earlier draft argued that low-order bits are most disturbed by re-encoding. Chromaprint items are classifier outputs, not magnitudes, so that reasoning does not hold and is removed. Upstream keeps the **top** 12 bits; that is the default here, configurable and calibrated.

## Architecture

Python 3.12, environment via `uv`, CLI via `typer`. Each module has one responsibility and is testable in isolation.

```
audiolib/
  store.py       # SQLite schema, migrations, queries
  scan.py        # walk, stat, audio-stream hash, tags, presence, errors
  fingerprint.py # fpcalc invocation -> raw uint32 array
  match.py       # candidate index, offset voting, segmentation -> pair evidence
  group.py       # content-level graph, clique/chain, tiers, track expansion
  rank.py        # keeper selection
  report.py      # JSON + human-readable output
  approve.py     # decision state transitions
  quarantine.py  # verified moves, manifest, undo
  enrich.py      # AcoustID: rate-limited, resumable, cached, best-effort
  cli.py
```

External binaries: `ffmpeg` (present), `fpcalc` (must be installed; requires user approval).

SQLite runs in WAL mode with a single writer. Concurrent `audiolib` invocations are not supported; a lock file makes the second invocation fail with a clear message rather than corrupting a run.

## Data model

SQLite, with a `schema_version` table from the first migration so subsystems B and C can add tables cleanly.

### Identity: audio content, not file bytes

A full-file hash changes whenever tags are edited, which would force re-fingerprinting the whole library after any tagging pass, and would fail to recognise the most common exact duplicate — the same rip carrying different tags.

Identity is therefore a hash of the **audio stream**, obtained by stream-copying packets with no decode:

```
ffmpeg -i <file> -map 0:a -c:a copy -f streamhash -hash sha256 -
```

This is I/O-bound only. If stream copy fails for a container, the implementation falls back to a whole-file hash and records which method was used.

`hash_method` and `audio_hash` together form the identity. A surrogate key carries it, so the two methods genuinely occupy separate namespaces rather than competing for one primary key:

```
audio_content
  id                   INTEGER PRIMARY KEY
  audio_hash           TEXT NOT NULL
  hash_method          TEXT NOT NULL      -- streamhash | whole_file
  duration             REAL
  codec, sample_rate, channels
  first_seen_at        TEXT
  UNIQUE (hash_method, audio_hash)
```

Everything downstream references `audio_content.id`, never the hash string, so no table has to carry the `(hash_method, audio_hash)` pair around or risk comparing across namespaces.

**Stability under retagging is a property of `streamhash`, not of the design.** A `streamhash` identity survives retagging and moving: only `track` changes. A `whole_file` fallback identity does **not** — editing tags changes the file, hence the hash, hence the identity, and the fingerprint is recomputed. The report flags any track on a `whole_file` identity so this weaker guarantee is visible rather than assumed away.

```
fingerprint
  id                   INTEGER PRIMARY KEY
  audio_content_id     INTEGER REFERENCES audio_content(id)
  purpose              TEXT NOT NULL      -- canonical | acoustid_lookup
  algorithm            INTEGER
  tool_version         TEXT               -- captured `fpcalc -version`
  config_hash          TEXT
  fp_raw               BLOB
  fp_length            INTEGER            -- element count
  computed_at          TEXT
  UNIQUE (audio_content_id, purpose, algorithm, tool_version, config_hash)

scan_run
  id                   INTEGER PRIMARY KEY
  started_at, completed_at
  roots                TEXT               -- JSON list
  status               TEXT               -- running | complete | failed

track
  id                   INTEGER PRIMARY KEY
  path                 TEXT UNIQUE
  size, mtime
  dev, inode, nlink    INTEGER            -- hardlink identity
  audio_content_id     INTEGER REFERENCES audio_content(id)
  bitrate              INTEGER
  tag_title, tag_artist, tag_album, tag_genre
  tag_completeness     INTEGER
  last_seen_scan_id    INTEGER REFERENCES scan_run(id)
  present              BOOLEAN
  absent_reason        TEXT               -- missing | quarantined
  scanned_at           TEXT

ingest_error
  path                 TEXT PRIMARY KEY
  dev, inode, size, mtime                 -- the state that failed
  stage                TEXT               -- hash | tags | fingerprint
  message              TEXT
  attempts             INTEGER
  last_attempt_at      TEXT
  resolved_at          TEXT

match_run
  id                   INTEGER PRIMARY KEY
  created_at           TEXT
  fingerprint_config   TEXT               -- JSON
  match_config         TEXT               -- JSON: align bits, K, M, thresholds
  software_version     TEXT
  status               TEXT               -- running | complete | failed

run_track                                 -- immutable snapshot
  run_id               INTEGER REFERENCES match_run(id)
  track_id             INTEGER
  path, size, mtime, audio_hash, hash_method
  PRIMARY KEY (run_id, track_id)

pair                                      -- content level, never track level
  run_id               INTEGER REFERENCES match_run(id)
  a_content_id         INTEGER            -- CHECK (a_content_id < b_content_id)
  b_content_id         INTEGER
  best_offset          INTEGER
  peak_votes           INTEGER
  peak_vote_ratio      REAL
  matched_span_items   INTEGER
  matched_span_seconds REAL
  coverage_a           REAL
  coverage_b           REAL
  mean_bit_error       REAL
  segment_count        INTEGER
  tier                 INTEGER
  verified_direct      BOOLEAN            -- computed outside the capped index
  PRIMARY KEY (run_id, a_content_id, b_content_id)

dup_group
  id                   INTEGER PRIMARY KEY
  run_id               INTEGER REFERENCES match_run(id)
  tier                 INTEGER
  formed_by_chain      BOOLEAN
  decision             TEXT               -- proposed | approved | rejected | applied
  decided_at           TEXT

group_content
  group_id             INTEGER REFERENCES dup_group(id)
  audio_content_id     INTEGER REFERENCES audio_content(id)

group_member
  group_id             INTEGER REFERENCES dup_group(id)
  track_id             INTEGER
  audio_content_id     INTEGER
  is_keeper            BOOLEAN
  rank_score           TEXT               -- the ordered tuple, for audit

acoustid_cache
  lookup_key           TEXT PRIMARY KEY   -- SHA256 of the full request
  response_json        TEXT
  fetched_at           TEXT

quarantine_log
  run_id, track_id, group_id
  src_path, dst_path   TEXT
  src_hash, dst_hash   TEXT
  moved_at             TEXT
  state                TEXT               -- moved | failed | undone
```

Properties that follow:

- Tracks sharing one `audio_content_id` are **tier 0** — identical encoded audio stream — detected before any fingerprinting or matching.
- Matching runs **once per unique audio stream**. Three retagged copies of one rip are aligned once.
- Several fingerprints coexist for one stream across tool versions, configs, and purposes.
- `match_run` plus `run_track` make reports reproducible: an old run resolves to the paths it actually saw, not to wherever a `track_id` points today.

### Fingerprint serialization

`fp_raw` is the fingerprint as **little-endian `uint32`, C-contiguous, no header**. `fp_length` is the element count, so `len(fp_raw) == fp_length * 4` is an invariant asserted on both read and write.

## Scan pipeline

Every scan opens a `scan_run` recording its roots.

1. Walk the configured roots, filtering by audio extension. **Quarantine directories are always excluded**, or quarantined files would be re-ingested and reappear as duplicates.
2. Record `(dev, inode, nlink)`. Paths sharing a `(dev, inode)` are the **same file** under different names, not duplicates; they are never proposed for quarantine against each other.
3. Fast path: if `(path, size, mtime)` is unchanged, reuse the stored `audio_content_id` and skip re-hashing. `--verify-hashes` forces re-hashing — the escape hatch for contents changing without size or mtime changing.
4. Otherwise compute the audio-stream hash and attach the track to its `audio_content` row, creating it if new.
5. Read tags and stream properties with `mutagen`; compute `tag_completeness`.
6. Fingerprint only content rows lacking a `canonical` fingerprint, or whose `tool_version` / `config_hash` no longer match current configuration.
7. Set `last_seen_scan_id` and `present = 1` for every track seen.

### Presence and disappearance

When a scan completes successfully, tracks **under its roots** that were not seen are marked `present = 0` with `absent_reason = 'missing'`. Tracks outside those roots are untouched, so scanning one directory never marks the rest of the library absent. A failed or interrupted scan marks nothing absent.

Only tracks with `present = 1` participate in the next `match`. This keeps stale entries from generating phantom duplicate groups against files the user deleted months ago.

Quarantining sets `present = 0` with `absent_reason = 'quarantined'`, distinguishing a file the tool moved from one the user removed — `undo` needs that distinction, and a quarantined file must never be reported as a missing keeper.

### Error retry

`ingest_error` stores the **file state** that failed — `(dev, inode, size, mtime)` — alongside the stage and message.

- Unchanged state on a later scan: not retried automatically, so an undecodable file is not re-attempted every run.
- Changed state: retried, because the file the error described no longer exists.
- `--retry-errors` forces retries regardless, for transient failures such as a disconnected drive.
- `resolved_at` is set rather than the row deleted, so the history stays auditable.

## Fingerprinting

`fpcalc -raw -length 0` — full-length, explicitly. The 120-second default would make every trim, edit, or partial overlap past the two-minute mark invisible, which is precisely what this tool must catch.

Output is decoded to a `numpy.uint32` array and stored per the serialization rule. At about 8 items/second a four-minute track is roughly 2,000 values (~8 KB); 20,000 tracks is on the order of 160 MB.

`tool_version` and `config_hash` are recorded so a Chromaprint upgrade or config change invalidates fingerprints deterministically rather than silently mixing incompatible data.

## Matching

Every run is recorded in `match_run` with the exact configuration used. Matching operates on `audio_content` rows reachable from at least one `present = 1` track.

### Candidate generation

The index is built transiently in numpy per run and is **not persisted**; a postings table would carry tens of millions of rows for no benefit.

`candidate_key(item, config)` maps each raw item to an index key. The default is upstream's `ALIGN_STRIP`: the **top 12 bits**. Configurable and calibrated.

Keys are generated for every fingerprint, sorted with `argsort`, and equal keys grouped into posting lists. Two caps bound the work, both in `match_config`:

- **`K` — stop-key cap across fingerprints.** Silence, fades, and common patterns produce identical keys across thousands of tracks. Pair emission from a posting list is O(n²), so one such key alone would generate millions of spurious candidates with a false offset spike near zero. Posting lists covering more than `K` fingerprints are dropped.
- **`M` — occurrence cap within one fingerprint per key.** A key repeating at many positions inside a single track produces a Cartesian product of δ values for every pair it joins. At most `M` positions per fingerprint per key are carried, chosen deterministically (first `M` by position).

### Offset voting and segmentation

For each candidate pair and each shared key, compute `δ = pos_a − pos_b` and histogram the δ values. Following upstream, peaks are bins with count > 1 that are local maxima; the highest becomes `best_offset`.

For that alignment: per-item Hamming distance over the overlapping region, Gaussian smoothing (sigma 8.0, order 3), absolute gradient, segment cuts at gradient peaks above 0.15 — **with no random jitter and deterministic tie-breaking by index order**. Each segment scores the mean raw bit count; segments below the match threshold (default 10.0) are kept, adjacent segments differing by less than 0.7 merged.

Evidence recorded per pair:

- `best_offset`, `peak_votes`
- `peak_vote_ratio` = peak bin votes / total votes cast — a **normalized** spike strength, since raw vote counts scale with track length and mean nothing alone
- `matched_span_items` and `matched_span_seconds` — summed duration of kept segments, converted using the runtime item duration. An **absolute** measure: 27 seconds matched is 90% of a 30-second track but 11% of a four-minute one, and those are not equivalent evidence
- `coverage_a` = matched span / `fp_length` of A, `coverage_b` likewise. Both stored, never blended — a clip inside a longer track has high `coverage_a` and low `coverage_b`, and that asymmetry is the signal
- `mean_bit_error` over kept segments, and `segment_count`

### Memory envelope

Raw fingerprint data is roughly 160 MB at 20,000 tracks; index and position arrays put expected peak usage near 1 GB. If the library outgrows that, the fallback partitions the key space into buckets and processes one at a time, accumulating pair evidence across buckets — identical results, lower peak memory.

## Grouping

Grouping happens at the **content** level and is expanded to tracks only at the end. The caps `K` and `M` make candidate generation deliberately **lossy**, so a missing edge is not evidence of a non-match — a true A–C pair may never have been generated. Hence the verification step.

1. **Collapse** tracks sharing an `audio_content_id` into a tier-0 equivalence class. This is where tier 0 enters: as the vertex definition, not as an edge.
2. **Build** a graph whose vertices are `audio_content` ids and whose edges are tier-1 pairs.
3. **Component**: take connected components.
4. **Verify**: within each component, run direct pairwise matching for every member pair, bypassing `K` and `M`. Components are small, so this is cheap, and a global stop-key cap is meaningless for a single pair. Results are stored with `verified_direct = 1`.
5. **Classify**: a component whose members are pairwise tier-1 complete is a **clique** (`formed_by_chain = 0`). One still connected only through intermediates — A–B and B–C tier 1 but A–C not, after direct verification — is `formed_by_chain = 1` and forced into the review tier regardless of scores.
6. **Expand** each content group back to every present track referencing those contents, recorded in `group_content` and `group_member`.
7. **Rank** keepers at the track level.

Without step 4, `formed_by_chain` would fire on candidate-generation misses rather than genuine non-matches.

### Tiers

- **Tier 0 — identical encoded audio stream.** Shared `audio_content_id`. No matching required.
- **Tier 1 — same recording.** `min(coverage_a, coverage_b)` high, `mean_bit_error` low, `peak_vote_ratio` above threshold, **and `matched_span_seconds >= min_same_recording_overlap_seconds`**. The absolute floor matters for short files, where two brief clips can reach 100% coverage on both sides from a tiny matched region.
- **Tier 2 — variant.** A strong spike with `max(coverage)` high and `min(coverage)` low (a clip inside a longer track), or both partial — radio edits, trimmed intros, songs inside a mix. Requires `matched_span_seconds >= min_variant_overlap_seconds` so a small coincidental region cannot become a "variant".

**Only tier 0 and tier 1 edges form keeper/quarantine groups. Tier 2 edges never authorize quarantine.** This is the safety boundary, and it is what stops `album track → radio edit → live version → remix` collapsing into one keeper-selection group. A radio edit is not an inferior copy of the album version; ranking it a loser for being shorter would be wrong.

## Keeper ranking

Tier 0 and tier 1 groups only, deterministic and reproducible:

`lossless > bitrate > tag completeness > longest duration > oldest mtime`, ties broken by path sort.

Candidates are the present tracks behind the group's contents. Paths sharing a `(dev, inode)` with the keeper are never proposed as losers. The ordered tuple is stored in `group_member.rank_score` for audit.

## Approval

The requirement is "auto-ranked, user confirms", so confirmation lives in the data model rather than in a convention about what a command implies.

Every group is created `decision = 'proposed'`. `apply` acts **only** on groups whose decision is `approved`, and a group with no approval is skipped and counted in the summary.

```
audiolib report --run 17                      # review proposals
audiolib approve --run 17 --group 12          # approve one
audiolib reject  --run 17 --group 12          # never propose again in this run
audiolib approve --run 17 --tier 1 --all      # bulk, requires interactive confirmation
audiolib apply   --run 17                     # act on approved groups only
```

Bulk approval exists because a library of this size can produce hundreds of groups and per-group approval alone would be unusable. It is explicit, scoped to a tier, and prints a summary needing confirmation before it commits. `--yes` is accepted for scripted use and is the only way to bypass the prompt.

`apply` sets `decision = 'applied'`. `undo` returns groups to `approved`.

## Quarantine

Invariants, stated as invariants rather than as a claim about which syscalls appear in the source:

1. **No group ever reaches a state with zero surviving copies.**
2. **Nothing is ever overwritten**, on move or on undo.
3. **Every move is verified against freshly read bytes**, not cached database state.

Mechanism: `os.link(src, dst)` then remove the source. `os.link` fails atomically if the destination exists, and fails with `EXDEV` across filesystems — which is the point. A cross-device `shutil.move` silently degrades to copy-then-delete, so a quarantine directory lives **on each filesystem** (`<root>/.audiolib-quarantine/`), and a library spread over several drives is handled rather than refused. These directories are excluded from scanning.

`audiolib apply <run-id>` proceeds per approved group:

1. Fresh-hash the **keeper** from disk first. If it is missing or its hash differs from the run snapshot, **skip the entire group** and warn. Moving losers when the keeper has vanished would leave no copies at all.
2. For each proposed loser, fresh-hash from disk and compare against `run_track`. The scan's `(path, size, mtime)` fast path is a cache shortcut and is not trusted for a destructive-ish action.
3. Link into quarantine, verify the destination hash, then unlink the source.
4. Record `src_hash`, `dst_hash`, and a per-file `state`; mark the track `present = 0, absent_reason = 'quarantined'`. A partial failure leaves the run in an explicit state, not an ambiguous one.

`audiolib undo <run-id>` restores from the manifest and refuses to overwrite any path that has appeared since the move.

## Enrichment

Best-effort and strictly subordinate: **local matching is authoritative, and enrichment never blocks or alters dedup.** It may fail, be rate-limited, or return nothing, and dedup results stand unchanged.

AcoustID lookups at no more than 3 requests/second, with `meta=recordings+releasegroups+compress` so metadata arrives inline, avoiding MusicBrainz's 1 request/second wall for the common case. The job is resumable.

Cache key is a SHA256 over the **entire request** — algorithm, encoded fingerprint, duration, meta configuration — so differing requests never collide.

**Lookup uses a 120-second fingerprint with the full file duration.** `duration` is documented as the whole file's duration, while canonical clients fingerprint with fpcalc's 120-second default, so a truncated fingerprint matches what the service's index was populated from.

The lookup fingerprint is obtained as follows:

1. *Preferred:* slice the stored canonical array and re-encode with `chromaprint.encode_fingerprint(raw[:n], algorithm)`.
2. *Validation:* assert the result equals a plain default `fpcalc` run on the same file.
3. *If incompatible:* run `fpcalc` at default settings and store the result as a **separate** `fingerprint` row with `purpose = 'acoustid_lookup'`.

The dedicated lookup fingerprint never replaces the canonical full-track one. Prefix-compatibility is an assumption under test, not an asserted fact.

MusicBrainz is contacted only when genre tags are needed, at 1 request/second, with a contactable `User-Agent`, deduplicated by release-group.

## CLI

```
audiolib scan <dirs> [--verify-hashes] [--retry-errors]
audiolib match
audiolib report [--run N] [--tier N]
audiolib approve --run N (--group G | --tier T --all) [--yes]
audiolib reject  --run N --group G
audiolib apply   --run N
audiolib undo    --run N
audiolib enrich
audiolib calibrate <pairs-file>
```

## Calibration

Threshold selection is a build step, not an afterthought. Upstream's constants are the starting point; calibration confirms or adjusts them for this library.

Tuned: `candidate_key` width, stop-key cap `K`, occurrence cap `M`, `peak_vote_ratio` threshold, coverage thresholds for tier 1 vs tier 2, `mean_bit_error` threshold, `min_same_recording_overlap_seconds`, `min_variant_overlap_seconds`.

**Tuning and validation use disjoint sets.** Thresholds are chosen on a calibration set and then accepted or rejected on a held-out validation set. Both include every category, so a configuration cannot look excellent by fitting one kind of example:

- same recording, identical encoding
- lossy transcode of the same recording
- trimmed head or tail
- radio edit
- mix or partial overlap
- unrelated track of the same genre
- unrelated track of near-identical duration

With a small library the holdout is statistically weak, so `calibrate` reports performance on **both** sets rather than only the validation figure. Chosen values are written into `match_run.match_config` for every run.

## Testing

Fixtures generated with `ffmpeg` from a few source files.

Functional:

- Re-encodes to mp3, flac, m4a at several bitrates — tier 1 ground truth.
- Retagged copies with identical audio — tier 0 ground truth. Editing tags with `mutagen` on mp3 (ID3v2 and ID3v1), flac, and m4a must leave the `streamhash` identity unchanged.
- A container forced onto the `whole_file` fallback — asserts the weaker guarantee is recorded and surfaced, not silently assumed.
- Head- and tail-trimmed versions and a synthetic concatenated "mix" — tier 2 ground truth.
- Unrelated tracks — negatives.
- Very short files with high coverage but tiny absolute overlap — asserts the tier 1 floor rejects them.
- A constructed A~B~C chain where A and C do not match after direct verification — asserts `formed_by_chain` and that the group cannot authorize quarantine.
- Hardlinked copies — asserts they are never proposed as losers against each other.
- `apply` on an unapproved group — asserts it is skipped.
- `apply` with a deleted keeper — asserts the whole group is skipped.
- `apply` with a modified loser — asserts it is skipped with a warning.
- Cross-filesystem quarantine — asserts `EXDEV` is handled via that filesystem's quarantine directory, never by copy-then-delete.
- `undo` against an occupied path — asserts refusal.
- A file deleted between scans — asserts `present = 0`, and that a scan of an unrelated root leaves it alone.
- A failed scan — asserts nothing is marked absent.
- `ingest_error` with unchanged state — asserts no retry; with changed state — asserts retry; with `--retry-errors` — asserts retry regardless.
- Round-trip of `fp_raw` — asserts `len(fp_raw) == fp_length * 4` and exact array equality.
- **Determinism** — two full runs over identical fixtures produce byte-identical reports.

Stress test for candidate explosion, the reason the caps exist: a fixture with long silence, repeated sections, near-identical intros, and many files sharing common keys. Asserts candidate count stays bounded, `K` and `M` take effect, peak memory stays within envelope, runtime within budget.

## Out of scope

Embeddings, genre classification, clustering, shuffle, recommendation, playback integration. No feature-vector column, because the embedding dimensions depend on a model not yet chosen. Schema versioning is in place so those tables arrive without migration pain.

Recorded so the Python pin does not block that work later: `essentia-tensorflow` release `2.1b6.dev1438` (2026-05-19) ships **cp314 wheels only**, while `2.1b6.dev1389` (2025-07-24) covers cp39–cp313. With Python pinned at 3.12, subsystem B would use `2.1b6.dev1389`.

## Approvals still required

- Installing `libchromaprint-tools` (system package; provides `fpcalc` and `libchromaprint1`).
- Creating the `uv` environment and installing Python dependencies.

Neither happens before this spec and its implementation plan are approved.
