# audiolib — Spec 1: Library Base + Duplicate Detection

Date: 2026-09-25
Status: Draft for review
Scope: Base layer + Subsystem A (dedup). Subsystems B (features/collections) and C (shuffle/recommendation) get their own specs.

## Context

The goal is a local music library analyzer for a personal collection of roughly 2,000–20,000 files, serving three eventual use cases:

1. Remove duplicate songs from collected music files.
2. Sort and build collections by musical features or genre.
3. Shuffle better, and recommend the next track from what has been played.

These are three subsystems over one shared base. This spec covers the base and use case 1 only. Use cases 2 and 3 share a single embedding pipeline and are deliberately deferred so that their storage layout is decided when their model is chosen, not guessed at now.

The original framing was "like Shazam". Shazam's landmark-hashing algorithm is built to match a short, noisy microphone clip against a database. Deduplicating whole clean files in a local library is a different problem, and Chromaprint/AcoustID is the tool built for it. The decision to use Chromaprint rather than hand-building a Wang-style engine was made explicitly, not by default.

What the Shazam material does contribute is the **scoring method**: offset-histogram voting. That technique is fingerprint-agnostic and is adopted here on top of Chromaprint, which is what makes trimmed intros, radio edits, and partial overlaps detectable without a second DSP pipeline.

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

## Verified environment facts

Checked against primary sources on 2026-09-25, not from memory:

- `ffmpeg` present at `/usr/bin/ffmpeg`. Muxers `hash`, `md5`, `streamhash`, `framehash`, `framemd5` available.
- `python3` 3.12.3, `uv` present.
- `fpcalc` **not installed**. `apt-cache policy libchromaprint-tools` → candidate `1.5.1-5` (pop-os noble/universe). Its dependencies include `libchromaprint1`, which pyacoustid's ctypes bindings require. Note: upstream Chromaprint is at v1.6.1 (2026-07-28); the distro package is older. The tool records `fpcalc -version` output rather than assuming a version.
- `fpcalc.cpp`: `static double g_max_duration = 120;` — **the default fingerprints only the first 120 seconds**. `const size_t stream_limit = g_max_duration * reader.GetSampleRate();` guarded by `if (stream_limit > 0)` — therefore **`-length 0` means unlimited**. Flags available: `-raw`, `-length`/`-t`, `-overlap`, `-ts`, `-chunk`, `-algorithm`, `-json`.
- pyacoustid exposes `fingerprint_file(path, force_fpcalc=)`, `lookup(apikey, fp, duration)`, `parse_lookup_result(data)`, `compare_fingerprints(a, b)`; the `chromaprint` module exposes `decode_fingerprint(data, base64=True) -> (uint32 array, algorithm)` and `encode_fingerprint(fingerprint, algorithm, base64=True)`.
- AcoustID web service: rate limit **3 requests/second**, non-commercial use, `client` API key required. The `duration` parameter is documented as "duration of the whole audio file in seconds". `meta` accepts `recordings`, `releasegroups`, `compress` among others.
- MusicBrainz API: **1 request/second per IP**, enforced by returning HTTP 503 for *all* requests once exceeded, and a contactable `User-Agent` string is mandatory.
- AcoustID's own server index (`acoustid-index` / `fpindex`) is described upstream as an "inverted index for searching audio fingerprints (an ID plus a set of hashes); search finds fingerprint IDs whose hash set intersects the query". Set-intersection over fingerprint hashes is therefore the production-proven approach, not a novel one.

## Architecture

Python 3.12, dependency and environment management via `uv`, CLI via `typer`. Each module has one responsibility and is testable in isolation.

```
audiolib/
  store.py       # SQLite schema, migrations, queries
  scan.py        # directory walk, stat, audio-stream hash, tag read
  fingerprint.py # fpcalc invocation -> raw uint32 array
  match.py       # candidate index, offset-histogram voting -> pair evidence
  group.py       # pair evidence -> groups, tier assignment, chain flagging
  rank.py        # keeper selection
  report.py      # JSON + human-readable output
  quarantine.py  # revalidated moves, manifest, undo
  enrich.py      # AcoustID lookup: rate-limited, resumable, cached, best-effort
  cli.py
```

External binaries: `ffmpeg` (present), `fpcalc` (must be installed; requires user approval).

## Data model

SQLite. A `schema_version` table exists from the first migration so subsystems B and C can add tables cleanly.

### Identity: audio content, not file bytes

The central modelling decision. A full-file hash changes whenever tags are edited, which would force re-fingerprinting the entire library after any tagging pass, and would fail to recognise the most common exact duplicate — the same rip carrying different tags.

Identity is therefore the hash of the **audio stream**, obtained by stream-copying packets with no decode:

```
ffmpeg -i <file> -map 0:a -c:a copy -f streamhash -hash sha256 -
```

This is I/O-bound only. If stream copy fails for a given container, the implementation falls back to a whole-file hash and records which method was used.

```
audio_content
  audio_hash           TEXT PRIMARY KEY
  duration             REAL
  codec                TEXT
  sample_rate          INTEGER
  channels             INTEGER
  fp_raw               BLOB          -- decoded uint32 array, full length
  fp_length            INTEGER
  fp_algorithm         INTEGER
  fp_tool_version      TEXT          -- captured from `fpcalc -version`
  fp_config_hash       TEXT          -- hash of the fingerprint invocation config
  fingerprinted_at     TEXT

track
  id                   INTEGER PRIMARY KEY
  path                 TEXT UNIQUE
  size                 INTEGER
  mtime                REAL
  audio_hash           TEXT REFERENCES audio_content(audio_hash)
  bitrate              INTEGER
  tag_title            TEXT
  tag_artist           TEXT
  tag_album            TEXT
  tag_genre            TEXT
  tag_completeness     INTEGER       -- derived, for ranking
  scanned_at           TEXT

match_run
  id                   INTEGER PRIMARY KEY
  created_at           TEXT
  fingerprint_config   TEXT          -- JSON
  match_config         TEXT          -- JSON: mask, thresholds, stop-key cap
  software_version     TEXT
  status               TEXT          -- running | complete | failed

pair
  run_id               INTEGER REFERENCES match_run(id)
  a_id                 INTEGER       -- enforced a_id < b_id
  b_id                 INTEGER
  best_offset          INTEGER
  votes                INTEGER
  coverage_a           REAL
  coverage_b           REAL
  ber                  REAL
  tier                 INTEGER
  PRIMARY KEY (run_id, a_id, b_id)

dup_group
  id                   INTEGER PRIMARY KEY
  run_id               INTEGER REFERENCES match_run(id)
  tier                 INTEGER
  formed_by_chain      BOOLEAN

group_member
  group_id             INTEGER REFERENCES dup_group(id)
  track_id             INTEGER REFERENCES track(id)
  is_keeper            BOOLEAN
  rank_score           TEXT          -- the ordered tuple, stored for auditability

acoustid_cache
  fp_hash              TEXT PRIMARY KEY
  response_json        TEXT
  fetched_at           TEXT

quarantine_log
  run_id               INTEGER
  track_id             INTEGER
  src_path             TEXT
  dst_path             TEXT
  group_id             INTEGER
  moved_at             TEXT
```

Two properties follow directly from this layout:

- Several `track` rows sharing one `audio_hash` are **tier 0**: byte-identical audio, detected before any fingerprinting or matching work.
- Retagging or moving a file updates `track` only. `audio_content` and its fingerprint are untouched.

`match_run` makes every report reproducible: two runs with different thresholds coexist without destroying each other's evidence.

## Scan pipeline

1. Walk the configured roots, filtering by audio extension.
2. Fast path: if `(path, size, mtime)` is unchanged from the stored row, reuse the existing `audio_hash` and skip re-hashing. `--verify-hashes` forces re-hashing regardless; this is the authoritative-identity escape hatch for the case where contents change without size or mtime changing.
3. Otherwise compute the audio-stream hash and attach the track to its `audio_content` row, creating it if new.
4. Read tags and stream properties with `mutagen`; compute `tag_completeness`.
5. Fingerprint only `audio_content` rows that lack a fingerprint, or whose `fp_tool_version` / `fp_config_hash` no longer match current configuration.

## Fingerprinting

`fpcalc -raw -length 0` — full-length, explicitly. The 120-second default would make every trim, edit, or partial overlap past the two-minute mark invisible, which is precisely the case this tool must catch.

The raw output is decoded to a `numpy.uint32` array and stored as a BLOB. At roughly 8 items per second, a four-minute track is about 2,000 values (~8 KB); 20,000 tracks is on the order of 160 MB of fingerprint data.

Both `fp_tool_version` and `fp_config_hash` are recorded, so a future Chromaprint upgrade or configuration change invalidates fingerprints deterministically rather than silently mixing incompatible data.

## Matching

Run under a `match_run` row that captures the exact configuration used.

### Candidate generation

The index is built transiently in numpy for each run and is **not persisted**; a postings table would carry tens of millions of rows for no benefit.

A configurable function `candidate_key(token, config)` maps each raw subfingerprint to an index key. The default implementation masks off low-order bits, which are the ones most disturbed by lossy re-encoding; exact 32-bit equality is too brittle to survive a transcode. The mask width is a calibrated parameter, not an architectural constant.

Keys are produced for every track, sorted with `argsort`, and equal keys grouped into posting lists.

**Stop-key cap.** Silence, fades, and other common patterns generate identical keys across thousands of tracks. Because pair emission from a posting list is O(n²), a single such key would generate millions of spurious candidates, all with a false offset spike near zero. Posting lists covering more than `K` tracks are therefore discarded. `K` is a calibrated parameter alongside mask width.

### Offset-histogram voting

For each candidate pair, and for each key they share, compute `δ = pos_a − pos_b` and histogram the δ values. A genuine match produces a tall spike at a single δ; unrelated tracks produce a flat histogram.

From the peak bin:

- `best_offset` — the δ of the peak
- `votes` — the height of the peak
- `coverage_a` = matched aligned span / fingerprint length of A
- `coverage_b` = matched aligned span / fingerprint length of B
- `ber` — bit-error rate over the aligned region

Both coverages are stored, never a single blended figure. A 60-second clip inside a four-minute track has high `coverage_a` and low `coverage_b`, and that asymmetry is exactly the signal that distinguishes it from a full duplicate.

### Memory envelope

Raw fingerprint data is roughly 160 MB at 20,000 tracks. Index and position arrays on top of it put expected peak usage near 1 GB. If the library grows beyond the envelope, the fallback is to partition the masked key space into buckets and process one bucket at a time; the pair evidence is accumulated across buckets, so results are unchanged.

## Grouping and tiers

- **Tier 0 — identical audio.** Shared `audio_hash`. No matching required.
- **Tier 1 — same recording.** `min(coverage_a, coverage_b)` high and `ber` low. Auto-ranked; losers *proposed* for quarantine.
- **Tier 2 — variant.** A strong offset spike with `max(coverage)` high and `min(coverage)` low (a clip inside a longer track), or with both partial. Radio edits, trimmed intros, songs inside a mix. **Everything is kept by default**; these are flagged for review only.

Tier 2 exists because near-variant matching and automatic keeper selection are in tension: a radio edit is not an inferior copy of the album version, it is a different thing, and ranking it as a loser because it is shorter would be wrong.

Threshold matching is **not transitive**. A~B and B~C does not imply A~C, and naive union-find will chain live versions and remixes into one useless blob. Every group therefore stores its per-member pairwise evidence, and any group that cohered only through a chain is marked `formed_by_chain` and forced into the review tier regardless of its scores.

## Keeper ranking

Tier 1 only, deterministic and reproducible:

`lossless > bitrate > tag completeness > longest duration > oldest mtime`, with ties broken by path sort.

The full ordered tuple is stored in `group_member.rank_score` so any decision can be audited after the fact.

## Quarantine

Nothing is ever deleted. The codebase contains no `unlink` call.

`audiolib apply <run-id>` moves proposed losers into a quarantine directory, preserving relative paths, and writes a manifest. `audiolib undo <run-id>` restores from that manifest.

**Applying revalidates first.** Because runs are stored historically, a run can be applied after the library has changed. Each loser's current `(path, size, mtime, audio_hash)` is checked against the run's snapshot; on any mismatch the file is skipped and a warning is emitted. Applying a stale run must not move the wrong file.

## Enrichment

Best-effort and strictly subordinate: **local matching is authoritative, and enrichment never blocks or alters dedup**. It may fail, be rate-limited, or return nothing, and dedup results stand unchanged.

AcoustID lookup at no more than 3 requests/second, with `meta=recordings+releasegroups+compress` so that metadata arrives inline — this avoids MusicBrainz's 1 request/second wall for the common case. Responses are cached by fingerprint hash; the job is resumable.

**Lookup uses a 120-second-truncated fingerprint with the full file duration.** The `duration` parameter is documented as the whole file's duration, while canonical clients fingerprint with `fpcalc`'s 120-second default; sending a truncated fingerprint therefore matches what the service's index was populated from. The truncation is done by slicing the stored raw array and re-encoding with `chromaprint.encode_fingerprint(raw[:n], algorithm)`. *This assumes Chromaprint fingerprints are prefix-compatible.* A test asserts that the re-encoded truncation equals a plain default `fpcalc` run on the same file; if it does not, a second fingerprint pass at default settings is stored instead.

MusicBrainz is contacted only if genre tags are needed, at 1 request/second, with a contactable `User-Agent`, deduplicated by release-group.

## CLI

```
audiolib scan <dirs> [--verify-hashes]   # walk, hash, tag, fingerprint; incremental
audiolib match                           # build index, vote, write a match_run
audiolib report [--run N] [--tier N]     # human-readable + JSON
audiolib apply <run-id>                  # revalidate, then quarantine losers
audiolib undo <run-id>                   # restore from manifest
audiolib enrich                          # AcoustID pass
audiolib calibrate <pairs-file>          # tune thresholds against known pairs
```

## Calibration

Threshold selection is a build step, not an afterthought. `audiolib calibrate` takes a small set of known duplicate and known non-duplicate pairs from the user's actual library and tunes:

- mask width in `candidate_key`
- stop-key cap `K`
- vote threshold
- coverage thresholds for tier 1 vs tier 2
- BER threshold

Defaults ship, but they are recorded as provisional until measured against real data. The chosen values are written into `match_run.match_config` for every run.

## Testing

Fixtures are generated with `ffmpeg` from a small number of source files:

- Re-encodes to mp3, flac, and m4a at several bitrates — tier 1 ground truth.
- Retagged copies with identical audio — tier 0 ground truth, and the regression test for identity stability.
- Head- and tail-trimmed versions, and a concatenation into a synthetic "mix" — tier 2 ground truth.
- Unrelated tracks — negatives.

Assertions cover: tier assignment; keeper selection; `formed_by_chain` flagging on a constructed A~B~C chain where A and C do not match; that `apply` moves rather than deletes and that `undo` restores; that editing tags with `mutagen` on mp3 (both ID3v2 and ID3v1), flac, and m4a leaves `audio_hash` unchanged; and that a stale run is refused by revalidation.

## Out of scope

Embeddings, genre classification, clustering, shuffle, recommendation, and playback integration. No feature-vector column is added here, because the embedding dimensions depend on a model that has not been chosen. Schema versioning is in place so those tables arrive without migration pain.

For reference when subsystem B is specified: `essentia-tensorflow` release `2.1b6.dev1438` (2026-05-19) ships **cp314 wheels only**, while `2.1b6.dev1389` (2025-07-24) covers cp39 through cp313. With Python pinned at 3.12, subsystem B would use `2.1b6.dev1389`. This is recorded now so the Python pin chosen in this spec does not block that work later.

## Approvals still required

- Installing `libchromaprint-tools` (system package, provides `fpcalc` and `libchromaprint1`).
- Creating the `uv` environment and installing Python dependencies.

Neither is done before this spec and its implementation plan are approved.
