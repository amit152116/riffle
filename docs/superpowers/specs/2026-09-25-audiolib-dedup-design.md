# audiolib — Spec 1: Library Base + Duplicate Detection

Date: 2026-09-25
Status: Draft for review (revision 2)
Scope: Base layer + Subsystem A (dedup). Subsystems B (features/collections) and C (shuffle/recommendation) get their own specs.

## Context

The goal is a local music library analyzer for a personal collection of roughly 2,000–20,000 files, serving three eventual use cases:

1. Remove duplicate songs from collected music files.
2. Sort and build collections by musical features or genre.
3. Shuffle better, and recommend the next track from what has been played.

These are three subsystems over one shared base. This spec covers the base and use case 1 only. Use cases 2 and 3 share a single embedding pipeline and are deliberately deferred so their storage layout is decided when their model is chosen, not guessed at now.

The original framing was "like Shazam". Shazam's landmark-hashing algorithm is built to match a short, noisy microphone clip against a database. Deduplicating whole clean files in a local library is a different problem, and Chromaprint/AcoustID is the tool built for it. Using Chromaprint rather than hand-building a Wang-style engine was decided explicitly.

What the Shazam material contributes is the **scoring method**: offset-histogram voting. That technique is fingerprint-agnostic, and it is what makes trimmed intros, radio edits, and partial overlaps detectable without a second DSP pipeline.

**Chromaprint already implements this method internally** (see "Upstream matching algorithm" below). It is not exposed in the public C API, so it is reimplemented here in numpy — but with upstream's constants and structure rather than invented ones.

## Requirements

Confirmed with the user:

| Decision | Value |
|---|---|
| Fingerprint engine | Chromaprint (`fpcalc`), not hand-built landmark hashing |
| Language | Python |
| Library size | 2,000–20,000 files |
| Duplicate definition | Same recording **and** near-variants (radio edits, trimmed intros, different masters) |
| Keeper selection | Auto-ranked, user confirms; losers quarantined, never deleted |
| Network | AcoustID/MusicBrainz lookups allowed, cached locally |
| Playback integration | Deferred (subsystem C) |

## Verified facts

Checked against primary sources on 2026-09-25. **Each fact is labelled with the version it was checked against**, because the installable package is older than upstream master.

### Local environment

- `ffmpeg` at `/usr/bin/ffmpeg`. Muxers `hash`, `md5`, `streamhash`, `framehash`, `framemd5` available.
- `python3` 3.12.3, `uv` present.
- `fpcalc` **not installed**. `apt-cache policy libchromaprint-tools` → candidate `1.5.1-5` (pop-os noble/universe). Dependencies include `libchromaprint1`, which pyacoustid's ctypes bindings require. Upstream is at v1.6.1 (2026-07-28); **the version that will actually be installed is 1.5.1**. The tool records `fpcalc -version` output at fingerprint time rather than assuming.

### fpcalc (verified against tag **v1.5.1**, the version to be installed)

- `static double g_max_duration = 120;` — **the default fingerprints only the first 120 seconds.**
- `const size_t stream_limit = g_max_duration * reader.GetSampleRate();` guarded by `if (stream_limit > 0)` — therefore **`-length 0` means unlimited**.
- Flags present in v1.5.1: `-format/-f`, `-channels/-c`, `-rate/-r`, `-length/-t`, `-chunk`, `-algorithm/-a`, `-text`, `-json`, `-plain`, `-overlap`, `-ts`, `-raw`, `-signed`, `-ignore-errors`, `-version`.
- These three facts are identical in master and v1.5.1.

### Public C API (master; symbol set is stable)

`chromaprint_get_item_duration`, `chromaprint_get_item_duration_ms`, `chromaprint_get_sample_rate`, `chromaprint_get_delay`, `chromaprint_encode_fingerprint`, `chromaprint_decode_fingerprint`, `chromaprint_hash_fingerprint`, and the fingerprinting calls.

**`FingerprintMatcher` is not exported.** The matching algorithm must be reimplemented.

Item duration is obtained at runtime from `chromaprint_get_item_duration() / chromaprint_get_sample_rate()` rather than hardcoded, so `matched_span_seconds` stays correct across algorithm settings. For the default algorithm this is roughly 0.124 s per item, about 8 items per second.

### pyacoustid

`fingerprint_file(path, force_fpcalc=)`, `lookup(apikey, fp, duration)`, `parse_lookup_result(data)`, `compare_fingerprints(a, b)`; module `chromaprint` exposes `decode_fingerprint(data, base64=True) -> (uint32 array, algorithm)` and `encode_fingerprint(fingerprint, algorithm, base64=True)`.

### Web services

- AcoustID: **3 requests/second**, non-commercial, `client` API key required. `duration` is documented as "duration of the whole audio file in seconds". `meta` accepts `recordings`, `releasegroups`, `compress`, among others.
- MusicBrainz: **1 request/second per IP**, enforced by returning HTTP 503 for *all* requests once exceeded; a contactable `User-Agent` is mandatory.
- AcoustID's server index (`acoustid-index` / `fpindex`) is described upstream as an "inverted index for searching audio fingerprints (an ID plus a set of hashes); search finds fingerprint IDs whose hash set intersects the query." Set intersection over fingerprint hashes is the production-proven approach, not a novel one.

## Upstream matching algorithm

From `chromaprint/src/fingerprint_matcher.{h,cpp}` (master). These constants are **adopted as defaults and cited**, not invented:

| Constant | Value | Role |
|---|---|---|
| `ALIGN_BITS` | 12 | `ALIGN_STRIP(x) = x >> (32 - 12)` — the alignment key is the **top 12 bits** of each item |
| `kDefaultMatchThreshold` | 10.0 | A segment is a match when its mean Hamming distance (0–32) is below this |
| `ACOUSTID_MAX_BIT_ERROR` | 2 | Stricter bit-error bound used in AcoustID's own context |
| `ACOUSTID_MAX_ALIGN_OFFSET` | 120 | Maximum alignment offset in AcoustID's context |
| Gaussian filter | sigma 8.0, order 3 | Smooths the per-item bit-error series before segmentation |
| Gradient peak | > 0.15, local max, non-adjacent | Segment boundary detection |
| Segment merge | \|score difference\| < 0.7 | Adjacent segments with similar scores are merged |

The procedure: build `(align_key, index, source)` triples for both fingerprints, sort, and for each equal-key run emit `offset_diff = offset1 + fp2_size - offset2` into a histogram. Peaks are bins with count > 1 that are local maxima. For the best alignment, compute per-item Hamming distance over the overlapping region, Gaussian-smooth it, take the gradient, and cut segments at gradient peaks. Each segment's score is the mean **raw** (unsmoothed) bit count over it; segments scoring below the threshold are kept, and adjacent similar-scoring ones merged.

Two consequences for this spec:

- **The masking rationale is corrected.** An earlier draft argued that low-order bits are most disturbed by re-encoding. Chromaprint items are classifier outputs, not magnitudes, so that reasoning does not hold and has been removed. Upstream keeps the **top 12 bits**; that is the default here, and it remains configurable and calibrated.
- **`Segment(pos1, pos2, duration, score)` is the definition of "matched span"** used throughout this spec, rather than an ad-hoc one.

## Architecture

Python 3.12, environment via `uv`, CLI via `typer`. Each module has one responsibility and is testable in isolation.

```
audiolib/
  store.py       # SQLite schema, migrations, queries
  scan.py        # walk, stat, audio-stream hash, tags, error recording
  fingerprint.py # fpcalc invocation -> raw uint32 array
  match.py       # candidate index, offset voting, segmentation -> pair evidence
  group.py       # pair evidence -> components, clique/chain, tiers
  rank.py        # keeper selection
  report.py      # JSON + human-readable output
  quarantine.py  # verified moves, manifest, undo
  enrich.py      # AcoustID: rate-limited, resumable, cached, best-effort
  cli.py
```

External binaries: `ffmpeg` (present), `fpcalc` (must be installed; requires user approval).

## Data model

SQLite, with a `schema_version` table from the first migration so subsystems B and C can add tables cleanly.

### Identity: audio content, not file bytes

A full-file hash changes whenever tags are edited, which would force re-fingerprinting the whole library after any tagging pass, and would fail to recognise the most common exact duplicate — the same rip carrying different tags.

Identity is therefore a hash of the **audio stream**, obtained by stream-copying packets with no decode:

```
ffmpeg -i <file> -map 0:a -c:a copy -f streamhash -hash sha256 -
```

This is I/O-bound only. If stream copy fails for a container, the implementation falls back to a whole-file hash and records which method was used. **Hashes are only ever compared within the same `hash_method`**; a `streamhash` value and a `whole_file` value are different namespaces and are never treated as equal or unequal to each other.

```
audio_content
  audio_hash           TEXT PRIMARY KEY
  hash_method          TEXT NOT NULL      -- streamhash | whole_file
  duration             REAL
  codec, sample_rate, channels
  first_seen_at        TEXT

fingerprint
  id                   INTEGER PRIMARY KEY
  audio_hash           TEXT REFERENCES audio_content(audio_hash)
  purpose              TEXT NOT NULL      -- canonical | acoustid_lookup
  algorithm            INTEGER
  tool_version         TEXT               -- captured `fpcalc -version`
  config_hash          TEXT               -- hash of the invocation config
  fp_raw               BLOB
  fp_length            INTEGER            -- element count
  computed_at          TEXT
  UNIQUE (audio_hash, purpose, algorithm, tool_version, config_hash)

track
  id                   INTEGER PRIMARY KEY
  path                 TEXT UNIQUE
  size, mtime
  dev, inode, nlink    INTEGER            -- hardlink identity
  audio_hash           TEXT REFERENCES audio_content(audio_hash)
  bitrate              INTEGER
  tag_title, tag_artist, tag_album, tag_genre
  tag_completeness     INTEGER
  scanned_at           TEXT

ingest_error
  path                 TEXT PRIMARY KEY
  stage                TEXT               -- hash | tags | fingerprint
  message              TEXT
  attempts             INTEGER
  last_attempt_at      TEXT

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

pair                                      -- keyed on audio content, not tracks
  run_id               INTEGER REFERENCES match_run(id)
  a_hash               TEXT               -- CHECK (a_hash < b_hash)
  b_hash               TEXT
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
  PRIMARY KEY (run_id, a_hash, b_hash)

dup_group
  id                   INTEGER PRIMARY KEY
  run_id               INTEGER REFERENCES match_run(id)
  tier                 INTEGER
  formed_by_chain      BOOLEAN

group_member
  group_id             INTEGER REFERENCES dup_group(id)
  track_id             INTEGER
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

Properties that follow from this layout:

- Tracks sharing one `audio_hash` (same `hash_method`) are **tier 0** — identical encoded audio stream — detected before any fingerprinting or matching.
- Retagging or moving a file updates `track` only; `audio_content` and its fingerprints are untouched.
- Matching runs **once per unique audio stream**, not once per file. Three retagged copies of one rip are aligned once.
- Several fingerprints can coexist for one stream across tool versions, configs, and purposes.
- `match_run` plus `run_track` make reports genuinely reproducible: an old run resolves to the paths it actually saw, not to whatever a `track_id` points at today.

### Fingerprint serialization

`fp_raw` is the decoded fingerprint as **little-endian `uint32`, C-contiguous, no header**. `fp_length` is the number of `uint32` elements, so `len(fp_raw) == fp_length * 4` is an invariant the store asserts on read and write.

## Scan pipeline

1. Walk the configured roots, filtering by audio extension. **The quarantine directories are always excluded**, or quarantined files would be re-ingested on the next scan and reappear as duplicates.
2. Record `(dev, inode, nlink)`. Paths sharing a `(dev, inode)` are the **same file** reached by different names, not duplicates; they are recorded once and never proposed for quarantine against each other.
3. Fast path: if `(path, size, mtime)` is unchanged, reuse the stored `audio_hash` and skip re-hashing. `--verify-hashes` forces re-hashing; this is the escape hatch for contents changing without size or mtime changing.
4. Otherwise compute the audio-stream hash and attach the track to its `audio_content` row, creating it if new.
5. Read tags and stream properties with `mutagen`; compute `tag_completeness`.
6. Fingerprint only `audio_content` rows lacking a `canonical` fingerprint, or whose `tool_version` / `config_hash` no longer match current configuration.
7. Failures at any stage are written to `ingest_error` with an attempt count, so a file that cannot be decoded is reported once and not retried forever.

## Fingerprinting

`fpcalc -raw -length 0` — full-length, explicitly. The 120-second default would make every trim, edit, or partial overlap past the two-minute mark invisible, which is precisely what this tool must catch.

Output is decoded to a `numpy.uint32` array and stored per the serialization rule above. At roughly 8 items/second, a four-minute track is about 2,000 values (~8 KB); 20,000 tracks is on the order of 160 MB.

`tool_version` and `config_hash` are recorded so a Chromaprint upgrade or configuration change invalidates fingerprints deterministically instead of silently mixing incompatible data.

## Matching

Every run is recorded in `match_run` with the exact configuration used.

### Candidate generation

The index is built transiently in numpy per run and is **not persisted**; a postings table would carry tens of millions of rows for no benefit.

`candidate_key(item, config)` maps each raw item to an index key. The default is upstream's `ALIGN_STRIP`: the **top 12 bits**. The width is configurable and calibrated.

Keys are generated for every fingerprint, sorted with `argsort`, and equal keys grouped into posting lists. Two caps bound the work, and both live in `match_config`:

- **`K` — stop-key cap across fingerprints.** Silence, fades, and other common patterns produce identical keys across thousands of tracks. Because pair emission from a posting list is O(n²), one such key alone would generate millions of spurious candidates with a false offset spike near zero. Posting lists covering more than `K` fingerprints are dropped.
- **`M` — occurrence cap within one fingerprint per key.** A key repeating at many positions inside a single track produces a Cartesian product of δ values for every pair it participates in. At most `M` positions per fingerprint per key are carried, chosen deterministically (first `M` by position) so runs are reproducible.

### Offset voting and segmentation

For each candidate pair, and each shared key, compute `δ = pos_a − pos_b` and histogram the δ values. Following upstream, peaks are bins with count > 1 that are local maxima; the highest is taken as `best_offset`.

For that alignment, compute the per-item Hamming distance over the overlapping region, Gaussian-smooth (sigma 8.0, order 3), take the absolute gradient, and cut segments at gradient peaks above 0.15. Each segment scores the mean raw bit count over it; segments below the match threshold (default 10.0) are kept, and adjacent segments whose scores differ by less than 0.7 are merged.

Evidence recorded per pair:

- `best_offset`, `peak_votes`
- `peak_vote_ratio` = peak bin votes / total votes cast — a **normalized** spike strength, since raw vote counts scale with track length and say nothing on their own
- `matched_span_items` and `matched_span_seconds` — the summed duration of kept segments, converted using the runtime item duration. An **absolute** overlap measure: 27 seconds matched is 90% of a 30-second track but 11% of a four-minute one, and those are not equivalent evidence
- `coverage_a` = matched span / `fp_length` of A, and `coverage_b` likewise. Both are stored, never a single blended figure — a clip inside a longer track has high `coverage_a` and low `coverage_b`, and that asymmetry is the signal
- `mean_bit_error` over the kept segments, and `segment_count`

### Memory envelope

Raw fingerprint data is roughly 160 MB at 20,000 tracks; index and position arrays put expected peak usage near 1 GB. If the library outgrows that, the fallback is to partition the key space into buckets and process one at a time, accumulating pair evidence across buckets — results are unchanged, only peak memory falls.

## Grouping

The caps `K` and `M` make candidate generation deliberately **lossy**, so a missing edge is not evidence of a non-match — a true A–C pair may simply never have been generated. Grouping therefore has a verification step.

1. Build a graph whose edges are tier-1 pairs and take its connected components.
2. Within each component, run **direct pairwise matching for every member pair**, bypassing `K` and `M`. Components are small, so this is cheap, and a global stop-key cap is meaningless for a single pair. Results are stored with `verified_direct = 1`.
3. A component whose members are now pairwise tier-1 complete is a **clique**: `formed_by_chain = 0`.
4. A component that remains connected only through intermediates — A–B and B–C are tier 1 but A–C is not, after direct verification — is marked `formed_by_chain = 1` and forced into the review tier regardless of its scores.

Without step 2, `formed_by_chain` would fire on candidate-generation misses rather than on genuine non-matches.

### Tiers

- **Tier 0 — identical encoded audio stream.** Shared `audio_hash` within one `hash_method`. No matching required.
- **Tier 1 — same recording.** `min(coverage_a, coverage_b)` high, `mean_bit_error` low, `peak_vote_ratio` above threshold.
- **Tier 2 — variant.** A strong spike with `max(coverage)` high and `min(coverage)` low (a clip inside a longer track), or both partial — radio edits, trimmed intros, songs inside a mix. Additionally requires `matched_span_seconds >= min_variant_overlap_seconds`, so a small coincidental region cannot become a "variant".

**Only tier 0 and tier 1 edges can form keeper/quarantine groups. Tier 2 edges never authorize quarantine.** This is the safety boundary, and it is what stops `album track → radio edit → live version → remix` collapsing into a single keeper-selection group. A radio edit is not an inferior copy of the album version; ranking it a loser because it is shorter would be wrong.

## Keeper ranking

Tier 0 and tier 1 groups only, deterministic and reproducible:

`lossless > bitrate > tag completeness > longest duration > oldest mtime`, ties broken by path sort.

Candidates are the tracks behind the group's audio hashes. Paths sharing a `(dev, inode)` with the keeper are never proposed as losers. The full ordered tuple is stored in `group_member.rank_score` for audit.

## Quarantine

The invariants, stated as invariants rather than as a claim about which syscalls appear in the source:

1. **No group ever reaches a state with zero surviving copies.**
2. **Nothing is ever overwritten**, on move or on undo.
3. **Every move is verified against freshly read bytes**, not against cached database state.

Mechanism: `os.link(src, dst)` followed by removing the source. `os.link` fails atomically if the destination exists, and fails with `EXDEV` across filesystems — which is the point. A cross-device `shutil.move` silently degrades to copy-then-delete, so a quarantine directory lives **on each filesystem** (`<root>/.audiolib-quarantine/`), and a library spread over several drives is handled rather than refused. These directories are excluded from scanning.

`audiolib apply <run-id>` proceeds per group:

1. Fresh-hash the **keeper** from disk first. If it is missing or its hash differs from the run snapshot, **skip the entire group** and warn. Moving losers when the keeper has vanished would leave no copies at all.
2. For each proposed loser, fresh-hash it from disk and compare against `run_track`. The scan's `(path, size, mtime)` fast path is a cache shortcut and is not trusted for a destructive-ish action.
3. Link into quarantine, verify the destination hash, then unlink the source.
4. Record `src_hash`, `dst_hash`, and a `state` per file. A partial failure leaves the run in an explicit state rather than an ambiguous one.

`audiolib undo <run-id>` restores from the manifest and refuses to overwrite any path that has appeared since the move.

## Enrichment

Best-effort and strictly subordinate: **local matching is authoritative, and enrichment never blocks or alters dedup.** It may fail, be rate-limited, or return nothing, and dedup results stand unchanged.

AcoustID lookups at no more than 3 requests/second, with `meta=recordings+releasegroups+compress` so metadata arrives inline, avoiding MusicBrainz's 1 request/second wall for the common case. The job is resumable.

Cache key is a SHA256 over the **entire request** — algorithm, encoded fingerprint, duration, and meta configuration — so differing requests never collide in the cache.

**Lookup uses a 120-second fingerprint with the full file duration.** `duration` is documented as the whole file's duration, while canonical clients fingerprint with fpcalc's 120-second default, so a truncated fingerprint matches what the service's index was populated from.

The lookup fingerprint is obtained as follows:

1. *Preferred:* slice the stored canonical array and re-encode with `chromaprint.encode_fingerprint(raw[:n], algorithm)`.
2. *Validation:* assert the result equals a plain default `fpcalc` run on the same file.
3. *If incompatible:* run `fpcalc` at default settings and store the result as a **separate** `fingerprint` row with `purpose = 'acoustid_lookup'`.

The dedicated lookup fingerprint never replaces the canonical full-track one. Prefix-compatibility is an assumption under test, not an asserted fact.

MusicBrainz is contacted only when genre tags are needed, at 1 request/second, with a contactable `User-Agent`, deduplicated by release-group.

## CLI

```
audiolib scan <dirs> [--verify-hashes]   # walk, hash, tag, fingerprint; incremental
audiolib match                           # index, vote, verify, write a match_run
audiolib report [--run N] [--tier N]     # human-readable + JSON
audiolib apply <run-id>                  # verify keeper, verify losers, quarantine
audiolib undo <run-id>                   # restore from manifest
audiolib enrich                          # AcoustID pass
audiolib calibrate <pairs-file>          # tune and validate thresholds
```

## Calibration

Threshold selection is a build step, not an afterthought. Upstream's constants are the starting point; calibration confirms or adjusts them for this library.

Tuned parameters: `candidate_key` width, stop-key cap `K`, occurrence cap `M`, `peak_vote_ratio` threshold, coverage thresholds for tier 1 vs tier 2, `mean_bit_error` threshold, and `min_variant_overlap_seconds`.

**Tuning and validation use disjoint sets.** Thresholds are chosen on a calibration set and then accepted or rejected on a held-out validation set. Both sets include every category, so a configuration cannot look excellent by fitting one kind of example:

- same recording, identical encoding
- lossy transcode of the same recording
- trimmed head or tail
- radio edit
- mix or partial overlap
- unrelated track of the same genre
- unrelated track of near-identical duration

With a small library the holdout is statistically weak, so `calibrate` reports performance on **both** sets rather than only the validation figure, and the chosen values are written into `match_run.match_config` for every run.

## Testing

Fixtures are generated with `ffmpeg` from a few source files.

Functional tests:

- Re-encodes to mp3, flac, m4a at several bitrates — tier 1 ground truth.
- Retagged copies with identical audio — tier 0 ground truth. Editing tags with `mutagen` on mp3 (both ID3v2 and ID3v1), flac, and m4a must leave `audio_hash` unchanged.
- Head- and tail-trimmed versions and a synthetic concatenated "mix" — tier 2 ground truth.
- Unrelated tracks — negatives.
- A constructed A~B~C chain where A and C do not match after direct verification — asserts `formed_by_chain`, and that the group cannot authorize quarantine.
- Hardlinked copies — asserts they are never proposed as losers against each other.
- `apply` with a deleted keeper — asserts the whole group is skipped.
- `apply` with a modified loser — asserts it is skipped with a warning.
- Cross-filesystem quarantine — asserts `EXDEV` is handled by using that filesystem's quarantine directory, never by copy-then-delete.
- `undo` against an occupied path — asserts refusal.
- Round-trip of `fp_raw` — asserts `len(fp_raw) == fp_length * 4` and exact array equality.

Stress test for candidate explosion, which is the reason the caps exist at all: a fixture containing long silence, repeated sections, near-identical intros, and many files sharing common keys. Asserts that candidate count stays bounded, that `K` and `M` take effect, that peak memory stays within the envelope, and that runtime stays within budget.

## Out of scope

Embeddings, genre classification, clustering, shuffle, recommendation, playback integration. No feature-vector column is added, because the embedding dimensions depend on a model not yet chosen. Schema versioning is in place so those tables arrive without migration pain.

Recorded now so the Python pin does not block that work later: `essentia-tensorflow` release `2.1b6.dev1438` (2026-05-19) ships **cp314 wheels only**, while `2.1b6.dev1389` (2025-07-24) covers cp39–cp313. With Python pinned at 3.12, subsystem B would use `2.1b6.dev1389`.

## Approvals still required

- Installing `libchromaprint-tools` (system package; provides `fpcalc` and `libchromaprint1`).
- Creating the `uv` environment and installing Python dependencies.

Neither happens before this spec and its implementation plan are approved.
