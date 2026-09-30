"""Clustering benchmark on the cached subset (read-only).

Methods
  prod-kmeans     riffle's current approach: k-means on 6 scalar features +
                  13 MFCC means, standardised, k chosen by silhouette (5..15)
  effnet-kmeans   k-means on PCA(32) of the EffNet embedding, k by silhouette
  effnet-hdbscan  UMAP(10-d, cosine) then HDBSCAN (noise allowed)
  effnet-hdb-min5 / min15   the same with other minimum cluster sizes

Metrics
  label NMI/ARI   agreement with the folder labels (punjabi vs hindi),
                  on clustered (non-noise) labelled tracks
  source NMI      agreement with YouTube-vs-library origin (LOWER is better:
                  clusters should reflect music, not encoding)
  dup together    share of known same-recording pairs placed in one cluster
  noise, k, sizes
  stability       mean ARI between clusterings of random 80% subsamples,
                  on the tracks both contain

    python bench/cluster_eval.py
"""
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BENCH = Path(os.path.expanduser("~/riffle-bench"))
SCALARS = ["bpm", "energy", "danceability", "loudness_lufs",
           "spectral_centroid", "onset_rate"]


def load():
    subset = json.load(open(BENCH / "subset.json"))
    rows, feats = [], []
    for t in subset:
        f = BENCH / "cache" / f"{t['id']}.npz"
        if not f.exists():
            continue
        with np.load(f, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        if "mfcc_mean" in d and "effnet_emb" in d:
            rows.append(t); feats.append(d)
    return rows, feats


def kmeans_best(X, kmin=5, kmax=15, seed=42):
    best = (-1, None, None)
    for k in range(kmin, min(kmax + 1, len(X))):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        lab = km.fit_predict(X)
        s = silhouette_score(X, lab)
        if s > best[0]:
            best = (s, k, lab)
    return best[2]


def features_prod(feats):
    X = np.array([[float(f[k]) for k in SCALARS] + list(f["mfcc_mean"]) for f in feats])
    return StandardScaler().fit_transform(X)


def features_effnet_pca(feats, n=32):
    E = np.stack([f["effnet_emb"] for f in feats]).astype(np.float64)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    return StandardScaler().fit_transform(PCA(n, random_state=0).fit_transform(E))


def embedding_matrix(feats):
    E = np.stack([f["effnet_emb"] for f in feats]).astype(np.float64)
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)


def umap_embed(E, seed=0):
    import umap
    return umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.0,
                     metric="cosine", random_state=seed).fit_transform(E)


def run_method(name, feats, idx=None, seed=0):
    sub = feats if idx is None else [feats[i] for i in idx]
    if name == "prod-kmeans":
        return kmeans_best(features_prod(sub))
    if name == "effnet-kmeans":
        return kmeans_best(features_effnet_pca(sub))
    size = {"effnet-hdbscan": 10, "effnet-hdb-min5": 5, "effnet-hdb-min15": 15}[name]
    Z = umap_embed(embedding_matrix(sub), seed=seed)
    return HDBSCAN(min_cluster_size=size, min_samples=3).fit_predict(Z)


def evaluate(name, rows, feats):
    lab = run_method(name, feats)
    n = len(rows)
    clustered = lab >= 0
    ks = sorted(set(lab[clustered]))
    sizes = sorted([int((lab == k).sum()) for k in ks], reverse=True)
    y = np.array([r["label"] or "" for r in rows])
    src = np.array([r["source"] for r in rows])
    lm = clustered & (y != "")
    nmi = normalized_mutual_info_score(y[lm], lab[lm]) if lm.sum() > 1 else float("nan")
    ari = adjusted_rand_score(y[lm], lab[lm]) if lm.sum() > 1 else float("nan")
    snmi = normalized_mutual_info_score(src[clustered], lab[clustered])
    groups = {}
    for i, r in enumerate(rows):
        if "dup" in r["roles"] and r["group"] is not None:
            groups.setdefault(r["group"], []).append(i)
    pairs = [(a, b) for g in groups.values() for i, a in enumerate(g) for b in g[i + 1:]]
    together = np.mean([lab[a] == lab[b] and lab[a] >= 0 for a, b in pairs]) if pairs else float("nan")
    # stability
    rng = np.random.default_rng(1)
    aris = []
    runs = []
    for s in range(4):
        idx = np.sort(rng.choice(n, int(0.8 * n), replace=False))
        runs.append((idx, run_method(name, feats, idx, seed=s)))
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            (ia, la), (ib, lb) = runs[i], runs[j]
            common = np.intersect1d(ia, ib)
            pa = dict(zip(ia, la)); pb = dict(zip(ib, lb))
            aris.append(adjusted_rand_score([pa[c] for c in common], [pb[c] for c in common]))
    return (name, len(ks), 100 * (1 - clustered.mean()), sizes[0] if sizes else 0,
            sizes[-1] if sizes else 0, nmi, ari, snmi, together, float(np.mean(aris)))


def main():
    rows, feats = load()
    print(f"tracks: {len(rows)}")
    head = (f"{'method':<17}{'k':>3}{'noise%':>7}{'max':>5}{'min':>5} |{'labelNMI':>9}{'labelARI':>9}"
            f"{'srcNMI':>8} |{'dup-tog':>8}{'stab-ARI':>9}")
    print(head + "\n" + "-" * len(head))
    for name in ("prod-kmeans", "effnet-kmeans", "effnet-hdb-min5", "effnet-hdbscan", "effnet-hdb-min15"):
        r = evaluate(name, rows, feats)
        print(f"{r[0]:<17}{r[1]:>3}{r[2]:>7.1f}{r[3]:>5}{r[4]:>5} |{r[5]:>9.3f}{r[6]:>9.3f}{r[7]:>8.3f} |{r[8]:>8.2f}{r[9]:>9.2f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
