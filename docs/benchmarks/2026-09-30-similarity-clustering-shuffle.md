# Similarity, clustering and shuffle benchmark (2026-09-30)

Question: do pretrained audio embeddings and the other ideas from
`Music_Intelligence_Systems_Research_20260930` beat riffle's MFCC-based
similarity, k-means clustering and greedy shuffle? Rule: adopt what is clearly
better on measured data, benchmark what is doubtful.

Code: `bench/select_subset.py`, `bench/extract.py`, `bench/evaluate.py`,
`bench/cluster_eval.py`, `bench/cluster_fixed_k.py`, `bench/cluster_hier.py`,
`bench/shuffle_eval.py`. Features are cached in `~/riffle-bench/cache`; the
library database is only read.

## Subset (416 of 1,848 tracks)

82 tracks from 40 fingerprint-confirmed duplicate groups (30 pairing an mp3 with
a YouTube rip), 40 tracks from 20 intro/outro edit pairs, and 150 Punjabi + 150
Hindi tracks labelled by folder, balanced between YouTube rips and older library
files and excluding every duplicate-group member. Six tracks lack baseline
features because Essentia's beat tracker raises on them (now handled).

Caveats: 416 tracks is small (differences under ~0.03 are noise); folder labels
are weak (a language/style proxy); neighbour search runs inside the subset, so
duplicate retrieval is easier than on the full library.

## 1. Similarity: adopted

| Representation | Duplicate at rank 1 | Same-label in top 10 (chance 0.5) | Across YouTube/library |
|---|---|---|---|
| current (MFCC + tempo/key/energy) | 0.85 | 0.56 | 0.54 |
| MFCC, standardised, no MFCC0, mean+std | 0.90 | 0.68 | 0.64 |
| **Discogs-EffNet embedding** | **0.99** | **0.81** | **0.76** |
| EffNet, PCA-whitened | 0.99 | 0.74 | 0.67 |
| EffNet style probabilities | 0.89 | 0.80 | 0.77 |
| musicnn embedding | 0.93 | 0.76 | 0.74 |
| EffNet + tempo/key/energy blended | 0.90 | 0.76 | 0.74 |

* EffNet wins every objective test by margins far above noise; blending tempo,
  key and energy into it makes it worse, so they became shuffle rules instead.
* Hubness (share of tracks appearing in nobody's top-10): EffNet 5.4%, and 0.7%
  after Gaussian mutual proximity, with no loss of accuracy. Adopted.
* No source bias found for the current features (0.49, chance 0.5); EffNet is
  slightly source-biased (0.54).
* Percentile-ranking the current score's components lifts its label precision
  0.56 -> 0.68, confirming the component-scale problem, but it costs duplicate
  retrieval and still trails EffNet.

## 2. Clustering: adopted after a second round

First round: k-means on the embedding matched folder labels better at k >= 8
(purity 0.81 vs 0.74, NMI 0.20 vs 0.11) and kept duplicates together (0.91-1.00 vs
0.70-0.80) but was far less stable across subsamples (ARI 0.08-0.32 vs 0.40-0.71);
Ward and GMM did not fix it; HDBSCAN after UMAP reached 0.58 with 12% noise. That
ruled out swapping the algorithm as-is.

Second round: spectral clustering of the mutual-proximity neighbour graph
(`bench/cluster_graph.py`), same k, 416 tracks:

| k | method | purity | NMI | duplicates together | stability | sizes |
|---|---|---|---|---|---|---|
| 6 | k-means, current features | 0.727 | 0.111 | 0.77 | 0.63 | 136/5 |
| 6 | k-means, embedding | 0.749 | 0.181 | 0.91 | 0.17 | 131/22 |
| 6 | **spectral, kNN graph** | **0.815** | **0.223** | **1.00** | **0.63** | 113/23 |
| 10 | k-means, current features | 0.747 | 0.111 | 0.75 | 0.45 | 80/1 |
| 10 | **spectral, kNN graph** | **0.807** | **0.208** | **0.91** | **0.66** | 74/3 |

Average linkage scored the best purity (0.887) but returned one giant cluster
plus singletons. The number of clusters is **not** auto-detected: eigengap and
silhouette chose k = 5/19/10/10/5 and 5/7/4/5/16 on the full subset and four
80% subsamples, and quality is nearly flat across k (purity 0.81-0.86, NMI ~0.21),
so k is a granularity setting; the default is 8.

Full-library check (1,839 contents) exposed a flaw the subset hid: with 10
neighbours per node, two of eight clusters had 3 tracks. Variants
(`bench/cluster_full.py`, purity against folder labels on 1,468 labelled tracks):

| k | variant | sizes | purity | NMI | stability |
|---|---|---|---|---|---|
| 8 | 10 neighbours | 459 425 302 291 238 118 3 3 | 0.777 | 0.178 | 0.65 |
| 8 | **20 neighbours** | 409 302 290 227 192 187 135 97 | **0.801** | 0.169 | 0.60 |
| 8 | 10 + weak link between all pairs | 820 243 205 ... | 0.689 | 0.096 | 0.62 |
| 8 | 10, then merge tiny clusters | 6 clusters | 0.775 | 0.178 | 0.65 |
| 12 | 10 neighbours | ..., 8 4 3 3 | 0.790 | 0.166 | 0.61 |
| 12 | **20 neighbours** | 348 ... 80 (all >= 80) | **0.828** | 0.172 | **0.66** |

Adopted: neighbours = min(20, max(10, n // 50)) (10 up to ~500 tracks, 20 from
~1,000), k = 8 by default, new tracks placed by a vote of their 10 nearest
clustered neighbours so existing assignments never move, `--rebuild` relabels
new clusters to match the previous run by centroid.
Limitations: labels are only a language/style proxy; clusters are not named yet
(the EffNet style probabilities could name them); on the full library the two
Hindi clusters (459 and 425 tracks) look similar and were not separated by any
label I have.

## 3. Shuffle: adopted

The current shuffle keeps a playlist in the seed's style only 51% of the time
(chance 50%) while 99% of its transitions are key-compatible with a 2 BPM average
jump: its distance follows tempo, key and energy.

Retrieving a pool of nearest embedding neighbours and re-ranking with artist,
key and octave-aware tempo rules gave (100 seeds, 20-track playlists):

| Pool | Style coherence | Key-compatible | BPM jump |
|---|---|---|---|
| current shuffle | 0.515 | 0.99 | 2.0 |
| 20 neighbours | 0.683 | 0.74 | 5.2 |
| 40 | 0.659 | 0.88 | 5.7 |
| **80** | 0.665 | 0.93 | 5.6 |
| 80 + continuous tempo term 0.3 (octave-aware step) | **0.720** | **0.92** | **3.7** |
| 80 + continuous tempo term 1.0 | 0.706 | 0.78 | 1.9 |

Rule strength beyond 1x did not help (the pool limits it); a wider pool and a
continuous tempo term did. Adopted: pool 80, artist 0.5, step 0.3, continuous
0.3, key bonus 0.2, half/double time treated as the same tempo. MMR diversity
re-ranking gave no clear variety gain at lambda 0.7 and was not adopted.
Judges: folder-label coherence, key compatibility and BPM jump are independent
of the embedding; "smooth" and "variety" use the embedding distance and favour
it.

## Implemented

`riffle cluster` on embeddings (spectral, above), `riffle embed` (parallel, one model per worker), migration 9 (`audio_embedding`),
embedding index with mutual proximity (`riffle build-index`, method `auto`,
100 neighbours per track), shuffle rules above (legacy index keeps the old
rules), shuffle now includes tracks without tempo/key features, parallel and
deduplicated `riffle features`, beat-tracker failure no longer discards a track.

## Validation on the full library (1,848 tracks)

Embeddings for all 1,839 audio contents (0 failures); `riffle build-index`
takes 1.7 s (100 neighbours per track).

* Nearest neighbours are the same recording first (distance ~0.00-0.03), then
  same-style songs: Punjabi stays with Punjabi, *Rockstar: Tum Ho* pulls
  Bollywood romantic songs.
* Hubness on the real index: 0.9% of tracks appear in nobody's top 10 (max 36
  appearances, mean 10).
* First shuffles played an mp3 and its YouTube copy back to back (`Gabbroo`,
  `Oh Kyu Ni Jaan Ske`): dedup deliberately does not group same-song edits.
  Embedding-space distances: same-recording pairs median 0.009 (90th pct 0.096,
  99th 0.27), edited same-song pairs median 0.018, every track's nearest
  neighbour median 0.139. An audit of the 55 pairs under 0.05 that are not
  known duplicates found 19 of 22 sampled were the same song (the rest were
  compilations or lookalikes). The shuffle now skips any track within 0.05 of
  one already played: 0 repeated songs in 60 real playlists (11,400 pairs).
  Recall is partial by design (86% of same-recording pairs, 76% of edits, 45%
  of partial matches fall under 0.05).

## Not done / open

* Naming clusters (e.g. from the 400 Discogs style probabilities, which are not stored yet).
* Embeddings use a 60 s middle excerpt; a longer window was not tested.
* Recommendation from plays and skips (no listening data exists yet).
* Results are from one library; rerun `bench/` after major library changes.
