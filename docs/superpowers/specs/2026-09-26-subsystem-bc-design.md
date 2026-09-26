# Subsystem B+C Design Spec: Collection Sorting & Smart Shuffle

**Date**: 2026-09-26
**Status**: Implemented (rev 3 — post-implementation review + fix pass, §10 added)
**Scope**: Audio feature extraction, AcoustID parsing, collection browsing/clustering, similarity index, smart shuffle

> **C v1 vs C v2**: This spec covers content-based features only (C v1).
> Personalized recommendation from play history is C v2 — not in scope here.

---

## 1. Overview

Four phases building on the existing Riffle base layer (scan, fingerprint, dedup, quality, health, enrich):

1. **Feature extraction** — Essentia DSP features per track (BPM, key, MFCC, etc.)
2. **Metadata parsing** — structured data from cached AcoustID responses
3. **Collection sorting** — browse/filter/cluster tracks by features and metadata
4. **Smart shuffle** — similarity-based playlist generation with smooth transitions

Each phase ships independently. Later phases depend on earlier ones.

---

## 2. Phase 1: Audio Feature Extraction

### 2.1 New Module: `riffle/features.py`

```python
def extract_file(path: Path) -> dict:
    """Run Essentia DSP algorithms on one audio file.
    
    Returns dict with keys: bpm, bpm_confidence, key_name, scale,
    key_strength, loudness_lufs, danceability, energy, spectral_centroid,
    onset_rate, dynamic_complexity, dissonance, zcr, mfcc_mean (list[float]).
    
    BPM octave correction: if bpm < 60, double it; if bpm > 200, halve it.
    Energy derived from: RMS power normalized to 0-1 range.
    MFCC: 13 mean coefficients across all frames (frameSize=2048, hopSize=1024).
    """

def feature_scan(conn, limit: int | None = None) -> dict:
    """Analyze present tracks not yet in audio_features.
    
    Same pattern as quality_scan:
    - LEFT JOIN to find unanalyzed tracks
    - Skip missing files (increment failed count)
    - Store results in audio_features table
    - Idempotent: re-running is a no-op for already-analyzed tracks
    
    Returns {analyzed: int, failed: int, cached: int}.
    """

def render_features(conn) -> str:
    """Text summary of extracted features.
    
    Shows: total analyzed, BPM distribution (histogram buckets),
    key distribution (top keys), loudness range, danceability range.
    """
```

### 2.2 Schema — Migration 3

```sql
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
```

MFCC stored as BLOB: 13 × float64 = 104 bytes, packed/unpacked via numpy (`tobytes()` / `frombuffer()`). Same pattern as fingerprint storage in `store.py`.

`config_hash` stores a SHA-256 of the extraction parameters (sample rate, frame size, hop size, algorithms used) — consistent with `fingerprint.config_hash`. If extraction config changes, tracks with a stale hash are re-extracted.

Tier 2 columns (genre_discogs, mood_*, embedding) added via later migrations — keeps Tier 1 zero-TF.

### 2.3 CLI

```python
@app.command()
def features(as_json: bool = False,
             limit: int | None = typer.Option(None, help="Max tracks to analyze")):
    """Extract audio features (BPM, key, loudness, etc.) via Essentia."""
```

### 2.4 Essentia Algorithms Used

| Feature | Algorithm | Notes |
|---------|-----------|-------|
| BPM, confidence | `RhythmExtractor2013(method="multifeature")` | Returns bpm, ticks, confidence, estimates, intervals |
| Key, scale, strength | `KeyExtractor()` | Temperley profile by default |
| Loudness | `LoudnessEBUR128(sampleRate=44100)` | Returns momentary, short_term, integrated, range |
| Danceability | `Danceability()` | Returns danceability, dfa_array |
| MFCC | `MFCC()` per frame via `FrameGenerator` | 13 bands, mean across all frames |
| Spectral centroid | `SpectralCentroidTime()` on full signal | Brightness indicator |
| Onset rate | `OnsetRate()` | Onsets per second |
| Dynamic complexity | `DynamicComplexity()` | Loudness variation measure |
| Dissonance | `Dissonance()` per frame (frameSize=2048, hopSize=1024), mean | Harmonic roughness |
| ZCR | `ZeroCrossingRate()` | Percussive vs tonal |
| Energy | `Energy()` on signal, normalized to [0,1] by dividing by max possible | Total signal energy |

Audio loading: `es.MonoLoader(filename=str(path), sampleRate=44100)()`

**Deterministic parameters** (pinned, stored in `config_hash`):
- `sampleRate=44100` (all algorithms)
- `frameSize=2048, hopSize=1024, windowType='hann'` (MFCC, Dissonance)
- `numberBands=13` (MFCC)
- `method='multifeature'` (RhythmExtractor2013)
- BPM octave correction: `bpm < 60 → ×2; bpm > 200 → ÷2`

### 2.5 Testing

- `test_extract_file_normal_tone` — verify all keys present, BPM > 0, key in valid set
- `test_extract_file_silence` — verify low energy, low loudness
- `test_feature_scan_stores_results` — verify row in audio_features after scan
- `test_feature_scan_is_idempotent` — second scan returns cached=1, analyzed=0
- `test_feature_scan_skips_missing` — missing file increments failed
- `test_bpm_octave_correction` — bpm < 60 gets doubled
- `test_mfcc_pack_unpack` — roundtrip BLOB serialization
- `test_render_features` — text output contains expected sections

### 2.6 Dependencies

```
pip install essentia  # ~50MB, Linux/macOS
```

No TensorFlow for Tier 1.

---

## 3. Phase 2: AcoustID Response Parsing

### 3.1 New Module: `riffle/metadata.py`

```python
def parse_acoustid_response(response_json: str) -> dict | None:
    """Extract best recording match from one cached AcoustID JSON response.
    
    AcoustID returns: {status, results: [{id, score, recordings: [{id, title,
    artists: [{id, name}], releasegroups: [{id, title, type}]}]}]}
    
    Takes the highest-scoring result with recordings.
    Returns {acoustid_id, recording_mbid, recording_title,
             artists: [{name, mbid}],  # JSON array — Bollywood tracks often have multiple artists
             release_title, release_mbid, score} or None.
    """

def parse_all(conn) -> dict:
    """Walk acoustid_cache rows whose audio_content_id is not yet in
    musicbrainz_match. Parse each, insert best match.
    
    Returns {parsed: int, no_match: int, failed: int, cached: int}.
    """

def render_metadata(conn) -> str:
    """Summary: total matched, match rate, score distribution,
    top 10 artists by track count, low-confidence matches (score < 0.5).
    """
```

### 3.2 Schema — Migration 4

```sql
CREATE TABLE musicbrainz_match (
    id                 INTEGER PRIMARY KEY,
    audio_content_id   INTEGER NOT NULL REFERENCES audio_content(id),
    acoustid_id        TEXT,
    recording_mbid     TEXT,
    recording_title    TEXT,
    artists_json       TEXT,     -- JSON array: [{"name": "...", "mbid": "..."}]
    release_title      TEXT,
    release_mbid       TEXT,
    score              REAL,
    parsed_at          TEXT,
    UNIQUE (audio_content_id)
);
```

### 3.3 Linking acoustid_cache to audio_content

`acoustid_cache` is keyed by `lookup_key` (a hash of fingerprint + duration + meta). To find the audio_content_id:
- `enrich()` iterates audio_content rows and computes the lookup_key
- We need to store the mapping. Two options:
  1. Add `audio_content_id` column to `acoustid_cache` (schema change)
  2. Recompute the lookup_key per audio_content during parsing

Option 1 is cleaner. Migration 4 also adds:
```sql
ALTER TABLE acoustid_cache ADD COLUMN audio_content_id INTEGER REFERENCES audio_content(id);
```

And `enrich()` is updated to store the `audio_content_id` when inserting cache rows.

### 3.4 CLI

```python
@app.command()
def metadata(as_json: bool = False):
    """Parse cached AcoustID responses into structured metadata."""
```

### 3.5 Testing

- `test_parse_acoustid_response_valid` — known JSON → correct extraction
- `test_parse_acoustid_response_no_recordings` — returns None
- `test_parse_acoustid_response_multiple_results` — picks highest score
- `test_parse_all_idempotent` — second run returns cached only
- `test_render_metadata` — text output

---

## 4. Phase 3: Collection Sorting & Clustering

### 4.1 New Module: `riffle/collection.py`

```python
def browse(conn, *,
           sort_by: str = "artist",
           genre: str | None = None,
           mood: str | None = None,
           bpm_range: tuple[float, float] | None = None,
           key: str | None = None,
           cluster: int | None = None,
           limit: int = 50,
           offset: int = 0) -> list[dict]:
    """Query tracks with filters and sorting.
    
    JOINs: track + audio_features + musicbrainz_match.
    Genre filter uses LIKE for substring matching.
    sort_by validated against: artist, title, bpm, key, loudness,
        energy, danceability, genre, cluster.
    """

def stats(conn) -> dict:
    """Collection statistics.
    
    Returns {
        total_tracks, total_with_features, total_with_metadata,
        genre_distribution: [{genre, count}],
        bpm_histogram: [{bucket, count}],  # 60-80, 80-100, ...200+
        key_distribution: [{key, scale, count}],
        top_artists: [{artist, count}],
        mood_breakdown: {happy, sad, ...} (Tier 2 only),
        cluster_summary: [{id, label, count}],
    }
    """

def render_browse(rows: list[dict]) -> str:
    """Tabular: # | Artist | Title | Genre | BPM | Key | Energy | Cluster"""

def render_stats(data: dict) -> str:
```

### 4.2 New Module: `riffle/cluster.py`

```python
FEATURE_COLUMNS = ["bpm", "energy", "danceability", "loudness_lufs",
                   "spectral_centroid", "onset_rate"]
# Plus 13 MFCC coefficients = 19-dimensional feature vector

def cluster_tracks(conn, n_clusters: int | None = None) -> dict:
    """K-means clustering on normalized feature vectors.
    
    First run: full k-means. Auto-k via silhouette score (range 5-15)
    unless n_clusters specified. Stores centroids in cluster_centroid,
    assigns rows in cluster_assignment.
    
    Subsequent runs (centroids exist, no --rebuild): assign new tracks
    to nearest existing centroid. O(n_new × k).
    
    Edge cases:
    - N < 15 tracks with features: skip clustering, put all in cluster 0
    - Tied silhouette scores: pick smaller k
    - Missing features: skip tracks without full feature vector
    
    Returns {run_id, n_clusters, sizes: list[int], labels: list[str]}.
    """

def rebuild_clusters(conn, n_clusters: int | None = None) -> dict:
    """Full k-means rebuild with Hungarian ID stabilization.
    
    1. Run k-means on ALL tracks with features
    2. Match new centroids to old via scipy.optimize.linear_sum_assignment
    3. Relabel to preserve existing IDs where possible
    4. Insert new cluster_run, update cluster_centroid, write cluster_assignment
    """

def label_cluster(conn, cluster_id: int, run_id: int) -> str:
    """Auto-generate label from cluster's assigned tracks (not centroid).
    BPM: <80=Slow, 80-120=Mid-tempo, >120=Upbeat (median of tracks)
    Energy: <0.3=Calm, 0.3-0.7=Moderate, >0.7=Energetic (median)
    Scale: majority vote of tracks' key scale (major/minor)
    Centroid brightness: >median spectral_centroid=Bright, else=Warm
    """

def render_clusters(conn) -> str:
    """Summary: cluster ID, label, track count, 3 example tracks per cluster."""
```

### 4.3 Schema additions (part of Migration 5)

```sql
CREATE TABLE cluster_run (
    id                  INTEGER PRIMARY KEY,
    n_clusters          INTEGER NOT NULL,
    n_tracks            INTEGER NOT NULL,
    scaler_params       BLOB,    -- StandardScaler mean/std, pickled or packed
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
```

Only the latest `cluster_run` is "active" for browse/filter. Previous runs preserved for comparison.

### 4.4 CLI

```python
@app.command()
def browse(sort_by: str = "artist", genre: str | None = None,
           mood: str | None = None, bpm_min: float | None = None,
           bpm_max: float | None = None, key: str | None = None,
           cluster: int | None = None, limit: int = 50,
           as_json: bool = False):

@app.command()
def stats(as_json: bool = False):

@app.command()
def cluster(n: int | None = typer.Option(None),
            rebuild: bool = False, as_json: bool = False):
```

### 4.5 Testing

- `test_browse_default` — returns tracks sorted by artist
- `test_browse_filter_bpm_range` — only tracks in range
- `test_browse_filter_genre_substring` — LIKE matching
- `test_stats_bpm_histogram` — correct bucket counts
- `test_cluster_assigns_all_tracks` — no unassigned (all in cluster_assignment)
- `test_cluster_is_idempotent` — second run assigns new only
- `test_cluster_rebuild_stabilizes_ids` — Hungarian algorithm preserves IDs
- `test_cluster_label_from_tracks` — label derived from track majority, not centroid
- `test_cluster_drift_detection` — warns when >20% new tracks
- `test_cluster_small_library` — N < 15 puts all in cluster 0
- `test_cluster_missing_features_skipped` — tracks without features excluded
- `test_cluster_scaler_stored` — StandardScaler params saved in cluster_run

### 4.6 Dependencies

```
pip install scikit-learn  # KMeans, StandardScaler, silhouette_score
pip install scipy         # linear_sum_assignment
```

---

## 5. Phase 4: Smart Shuffle & Similarity

### 5.1 New Module: `riffle/similarity.py`

```python
CIRCLE_OF_FIFTHS = ["C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F"]
DEFAULT_WEIGHTS = {"mfcc": 0.5, "bpm": 0.25, "key": 0.15, "energy": 0.1}

def mfcc_cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance (0=identical, 2=opposite) between 13-dim MFCC vectors."""

def key_distance(key_a: str, scale_a: str, key_b: str, scale_b: str) -> int:
    """Steps on circle of fifths (0-6). Relative major/minor = 0."""

def normalize_components(mfcc_dist: float, bpm_diff: float,
                         key_dist: int, energy_diff: float) -> dict:
    """Normalize all similarity components to [0, 1] before weighting.
    - mfcc: divide by 2.0 (max cosine distance)
    - bpm: min(abs_diff / 60.0, 1.0)  — cap at 60 BPM difference
    - key: divide by 6.0 (max circle-of-fifths distance)
    - energy: already [0, 1] (both inputs normalized)
    """

def combined_score(features_a: dict, features_b: dict,
                   weights: dict = DEFAULT_WEIGHTS) -> float:
    """Normalized weighted combination. Lower = more similar.
    All components normalized to [0,1] via normalize_components()
    BEFORE applying weights."""

def build_similarity(conn, top_k: int = 20) -> dict:
    """Incremental nearest-neighbor index.
    
    1. Find tracks with features but no similarity entries
    2. Load ALL feature vectors (MFCC + BPM + key + energy)
    3. Compute combined_score for new vs ALL pairs directly
       (no MFCC-only pre-filtering — at <5000 tracks, full scoring
       is fast enough: ~6ms at 770, ~273ms at 5000)
    4. For each new track: store top-K by combined_score
    5. For existing tracks: if new track scores better than their
       worst neighbor, replace it
    
    Returns {new_tracks: int, pairs_stored: int, existing_updated: int}.
    """

def full_rebuild(conn, top_k: int = 20) -> dict:
    """Delete all track_similarity rows, recompute from scratch.
    Computes combined_score for all pairs, stores top-K per track."""
```

### 5.2 New Module: `riffle/shuffle.py`

```python
def find_similar(conn, track_id: int, n: int = 10) -> list[dict]:
    """Return n most similar tracks from track_similarity.
    JOINs with track for path/artist/title."""

def resolve_track(conn, query: str) -> int:
    """Find track_id by path substring or title match.
    Raises if ambiguous (multiple matches) or not found."""

def smart_shuffle(conn, *,
                  seed_track_id: int | None = None,
                  n: int = 20,
                  genre: str | None = None,
                  bpm_range: tuple[float, float] | None = None) -> list[dict]:
    """Generate a playlist via greedy nearest-neighbor chain.
    
    Hard constraints (track excluded if ANY violated):
    - Already in playlist
    - Known duplicate of a track in playlist (from Subsystem A's
      dup_group/group_content — query approved/applied groups)
    - Outside genre/BPM filter (if specified)
    
    Soft preferences (penalties/bonuses on combined_score):
    - penalty +0.5: same artist as any of last 3 tracks
    - penalty +0.3: BPM jump > 15 from previous
    - bonus -0.2: key distance ≤ 1 (circle of fifths)
    
    Algorithm:
    1. Build candidate pool (hard filters applied upfront)
    2. Build duplicate exclusion set from dup_group
    3. Pick seed (given or random from pool)
    4. For each next slot:
       a. Query top-K similar to current track
       b. Apply hard constraints (in pool, not in playlist, not dup)
       c. Apply soft preferences to remaining candidates
       d. Pick lowest-scoring (most similar after penalties)
       e. Fallback: random from pool if no candidates
    5. Return ordered playlist
    """

def render_similar(rows: list[dict]) -> str:
    """Table: # | Score | Artist | Title | BPM | Key"""

def render_playlist(rows: list[dict]) -> str:
    """Table: # | Artist | Title | BPM | Key | Transition"""
```

### 5.3 Schema — Migration 5

```sql
CREATE TABLE track_similarity (
    track_a_id       INTEGER NOT NULL REFERENCES track(id),
    track_b_id       INTEGER NOT NULL REFERENCES track(id),
    mfcc_norm        REAL NOT NULL,  -- normalized [0,1]: raw / 2.0
    bpm_norm         REAL,           -- normalized [0,1]: min(abs_diff/60, 1)
    key_norm         REAL,           -- normalized [0,1]: raw / 6.0
    energy_norm      REAL,           -- already [0,1]
    combined_score   REAL NOT NULL,  -- weighted sum of normalized components
    PRIMARY KEY (track_a_id, track_b_id),
    CHECK (track_a_id < track_b_id)
);
CREATE INDEX sim_track_a ON track_similarity(track_a_id, combined_score);
CREATE INDEX sim_track_b ON track_similarity(track_b_id, combined_score);
```

Combined with cluster_run, cluster_centroid, and cluster_assignment tables from Phase 3.

### 5.4 CLI

```python
@app.command()
def similar(track: str, n: int = 10, as_json: bool = False):
    """Find tracks similar to a given track (path substring match)."""

@app.command()
def shuffle(seed: str | None = None, n: int = 20,
            genre: str | None = None,
            bpm_min: float | None = None, bpm_max: float | None = None,
            as_json: bool = False):
    """Generate a smart playlist."""

@app.command()
def build_index(top_k: int = 20, rebuild: bool = False, as_json: bool = False):
    """Pre-compute similarity index. Incremental by default."""
```

### 5.5 Testing

- `test_mfcc_cosine_distance_identical` — distance = 0
- `test_mfcc_cosine_distance_orthogonal` — distance = 1
- `test_key_distance_same` — 0
- `test_key_distance_fifth` — 1
- `test_key_distance_tritone` — 6
- `test_key_distance_relative_minor` — 0 (Am ↔ C)
- `test_normalize_components_ranges` — all outputs in [0, 1]
- `test_combined_score_uses_normalized` — score is weighted sum of [0,1] values
- `test_build_similarity_stores_top_k` — correct number of rows
- `test_build_similarity_uses_combined_score` — top-K selected by combined_score, not MFCC alone
- `test_build_similarity_incremental` — new track gets neighbors, existing updated
- `test_find_similar_returns_ordered` — sorted by combined_score
- `test_smart_shuffle_excludes_duplicates` — known dups from Subsystem A never both appear (hard)
- `test_smart_shuffle_respects_genre_filter` — all tracks match genre (hard)
- `test_smart_shuffle_artist_diversity` — same artist rarely back-to-back (soft, statistical)
- `test_smart_shuffle_bpm_smoothness` — BPM jumps usually ≤15 (soft, statistical)
- `test_resolve_track_substring` — finds by partial path
- `test_resolve_track_ambiguous` — raises on multiple matches

---

## 6. Migration Summary

| Migration | Phase | Tables/Columns |
|-----------|-------|---------------|
| 3 | 1 | `CREATE TABLE audio_features` |
| 4 | 2 | `CREATE TABLE musicbrainz_match`, `ALTER acoustid_cache ADD audio_content_id` |
| 5 | 3+4 | `CREATE TABLE cluster_run`, `CREATE TABLE cluster_centroid`, `CREATE TABLE cluster_assignment`, `CREATE TABLE track_similarity` |

---

## 7. New Dependencies

| Package | Phase | Size | Purpose |
|---------|-------|------|---------|
| `essentia` | 1 | ~50MB | Audio feature extraction |
| `scikit-learn` | 3 | ~30MB | KMeans, StandardScaler, silhouette_score |
| `scipy` | 3 | (already with sklearn) | linear_sum_assignment for cluster stability |

**Licensing note**: Essentia is AGPLv3. Riffle is a local personal tool — no distribution or service implications. If distributing Riffle commercially, evaluate MTG/UPF's commercial license option.

---

## 8. New CLI Commands Summary

| Command | Phase | Description |
|---------|-------|-------------|
| `riffle features` | 1 | Extract audio features via Essentia |
| `riffle metadata` | 2 | Parse cached AcoustID responses |
| `riffle browse` | 3 | Filter and sort collection |
| `riffle stats` | 3 | Collection distributions |
| `riffle cluster` | 3 | K-means clustering |
| `riffle build-index` | 4 | Pre-compute similarity index |
| `riffle similar <track>` | 4 | Find similar tracks |
| `riffle shuffle` | 4 | Generate smart playlist |

---

## 9. File Map

| File | Phase | Purpose |
|------|-------|---------|
| `riffle/features.py` | 1 | Essentia extraction, feature_scan |
| `riffle/metadata.py` | 2 | AcoustID JSON parsing |
| `riffle/collection.py` | 3 | Browse, filter, stats queries |
| `riffle/cluster.py` | 3 | K-means clustering, centroid management |
| `riffle/similarity.py` | 4 | Distance metrics, similarity index |
| `riffle/shuffle.py` | 4 | Playlist generation, track resolution |
| `riffle/store.py` | 1-4 | Migrations 3-5 |
| `riffle/cli.py` | 1-4 | New commands |
| `tests/test_features.py` | 1 | Feature extraction tests |
| `tests/test_metadata.py` | 2 | Parsing tests |
| `tests/test_collection.py` | 3 | Browse/stats tests |
| `tests/test_cluster.py` | 3 | Clustering tests |
| `tests/test_similarity.py` | 4 | Distance/index tests |
| `tests/test_shuffle.py` | 4 | Playlist generation tests |

---

## 10. Similarity & Discovery Contract (post-implementation clarifications)

Added after implementation and a whole-branch review surfaced several
spec-level ambiguities. These are binding for any future change to
`similarity.py`, `shuffle.py`, `cluster.py`, or `collection.py`.

1. **Feature components are normalized before weighting.** Unchanged from
   §5.1 — implemented via `normalize_components()` / `score_components()`.

2. **Missing components are omitted and remaining weights renormalized.**
   A component whose inputs are unavailable — BPM that is `NULL` or `<= 0`
   (Essentia writes `0.0` for beatless/silent tracks, not `NULL`), or a
   `NULL` key — is dropped from the weighted sum entirely, and the
   remaining weights are rescaled to sum to 1. It is never defaulted to a
   literal `0` (a fabricated perfect BPM match) or `"C major"` (a
   fabricated key match). MFCC and energy are required by the feature
   query and always contribute. Implemented in
   `similarity.score_components()`; `shuffle.py`'s BPM-jump penalty and
   key-compatibility bonus honor the same missing-data rule.

3. **`track_similarity` stores directed top-K neighbors, not symmetric
   mutual relationships.** Schema (Migration 6): `(track_id, neighbor_id)`
   primary key, one row per directed edge. Track A can hold track B in its
   own top-K without B necessarily holding A in its top-K back. This
   replaced the original undirected `(track_a_id, track_b_id)` with
   `a < b` design, which conflated "pair exists" with "both tracks agree
   it's a top-K neighbor" and made per-track eviction impossible to do
   correctly.

4. **Similarity index eviction is per-track and exact.** `build_similarity`
   fills a track's row count up to `top_k` unconditionally while it has
   room; once at `top_k`, a new candidate is inserted only if it beats the
   track's current worst neighbor, and that worst row is deleted in the
   same operation so the count never exceeds `top_k`. Because storage is
   directed, this only ever touches the track's own rows — inserting or
   evicting for track A never touches track B's rows.

5. **Quarantined/missing tracks are excluded from normal discovery.**
   `browse`, `similar` (`find_similar`), `cluster` (`load_feature_matrix`,
   `render_clusters`), and `shuffle` (`smart_shuffle`) all filter to
   `track.present = 1`, which covers both `absent_reason = 'missing'` and
   `absent_reason = 'quarantined'`. A track quarantined after being
   clustered or indexed must disappear from rendered output on the next
   read, even without a rebuild.

6. **Similarity artifacts are tied to a versioned similarity
   configuration.** `track_similarity.config_hash` stores
   `similarity.similarity_config_hash()` (a SHA-256 of `DEFAULT_WEIGHTS`)
   on every row. A future change to the weighting scheme is detectable by
   comparing this hash, the same pattern used for `audio_features.config_hash`
   and `fingerprint.config_hash`.

7. **Shuffle treats duplicate-recording suppression as a hard constraint,
   including tracks sharing one `audio_content_id`.** Beyond the
   `dup_group`/`group_content` approved-duplicate exclusion from
   Subsystem A, `smart_shuffle` also excludes all tracks that share the
   *same* `audio_content_id` as any track already in the playlist — two
   present paths pointing at byte-identical audio, before dedup approval,
   must never both appear.

8. **`--as-json` always emits valid JSON, nothing else.** When `--as-json`
   is passed to `features`, `metadata`, `cluster`, or `build-index`, the
   command's entire stdout output is the JSON-encoded result dict (matching
   `browse`/`stats`/`similar`/`shuffle`, which already did this correctly)
   — no human-readable summary line precedes or follows it. Errors always
   go to stderr with a non-zero exit code, `--as-json` or not.

9. **A user-provided shuffle seed must satisfy the active hard filters.**
   If `--seed` resolves to a track outside the current `--genre`/BPM-range
   filter, `smart_shuffle` raises a clear `ValueError` rather than silently
   substituting a random seed from the filtered pool.

**Known deferred gaps** (tracked, not yet closed):

- Auto-k clustering (`n_clusters=None` with all-identical features) is not
  covered by a test that exercises the silhouette-failure path; the
  fallback behavior when every `silhouette_score` call raises is
  unverified.
- `track_similarity`'s per-track eviction is `O(new × existing)` in pure
  Python and does not use an ANN index; acceptable at the sizes this tool
  targets (see the incremental-design research), revisit if it becomes a
  bottleneck.
- Test nondeterminism: `test_smart_shuffle_artist_diversity` (and similar
  statistical assertions) use an unseeded `random`, so they are
  theoretically flaky. Not yet seeded.
- A track/candidate with no `tag_artist` is exempt from the same-artist
  penalty in `smart_shuffle` by construction (`cand_artist is not None`
  guard), rather than by an explicit "no artist recorded" rule — behavior
  is correct but undocumented as a deliberate policy versus an oversight.
- `acoustid_cache` rows written before Migration 4 have
  `audio_content_id = NULL` and are permanently invisible to
  `metadata.parse_all()`; re-running `enrich` does not backfill them on a
  cache hit. Not fixed — flagged for the user to decide whether any
  pre-Migration-4 data needs a one-time backfill.
