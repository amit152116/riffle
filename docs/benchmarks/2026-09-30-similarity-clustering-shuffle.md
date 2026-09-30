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

## 2. Clustering: NOT changed

At equal k (k-means), EffNet clusters match the folder labels better at k >= 8
(purity 0.81 vs 0.74, NMI 0.20 vs 0.11) and keep known duplicates together
(0.91-1.00 vs 0.70-0.80), but they are far less stable when songs are added or
removed (stability ARI 0.08-0.32 vs 0.40-0.71). Ward and GMM did not fix it;
HDBSCAN after UMAP reached 0.58 with 12% noise points. The embedding space looks
like a continuous spread, not crisp groups. A cluster list that reshuffles is
not usable, so clustering stays as is. Untested idea: communities of the
mutual-proximity neighbour graph, which would change gradually.

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

`riffle embed` (parallel, one model per worker), migration 9 (`audio_embedding`),
embedding index with mutual proximity (`riffle build-index`, method `auto`,
100 neighbours per track), shuffle rules above (legacy index keeps the old
rules), shuffle now includes tracks without tempo/key features, parallel and
deduplicated `riffle features`, beat-tracker failure no longer discards a track.

## Not done / open

* Clustering on a neighbour-graph community method (idea above).
* Embeddings use a 60 s middle excerpt; a longer window was not tested.
* Recommendation from plays and skips (no listening data exists yet).
* Results are from one library; rerun `bench/` after major library changes.
