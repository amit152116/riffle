"""Spectral-clustering variants on the FULL library (read-only).

The 416-track subset hid a flaw: at full scale small tightly linked groups
break off as clusters of 3 tracks. Variants
  base      kNN=10
  knn20     kNN=20
  eps       kNN=10 plus a weak link between every pair
  merge     kNN=10, then clusters under 2% of the library join the nearest centroid
Metrics: sizes, purity/NMI against the folder labels (punjabi vs hindi) on the
labelled tracks, and stability (ARI over four 80% subsamples).
"""
import sqlite3, sys, warnings
from pathlib import Path
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from riffle import similarity, cluster
ROOT = "/home/amit_152116/myDisk/Media/Amit's Music/"
conn = sqlite3.connect("file:/home/amit_152116/riffle-mymusic.sqlite?mode=ro", uri=True); conn.row_factory = sqlite3.Row
ids, X = similarity.load_content_embeddings(conn)
n = len(ids)
def label(path):
    rel = path[len(ROOT):] if path.startswith(ROOT) else path; t = rel.split("/")
    if t[0] == "youtube music": return {"punjabi": "punjabi", "Hindi": "hindi"}.get(t[1])
    if t[0] in ("Punjabi", "punjabi song"): return "punjabi"
    if t[0] in ("Heartfalling 1", "Heartfalling 2", "Retro"): return "hindi"
y = np.array([label(conn.execute("select path from track where audio_content_id=? and present=1 limit 1", (c,)).fetchone()[0]) or "" for c in ids])
U = cluster._unit(X)

def graph(idx, variant):
    D = similarity.mutual_proximity(similarity.cosine_distance_matrix(X[idx]))
    k = 20 if variant == "knn20" else 10
    A = similarity.knn_affinity(D, k=k)
    if variant == "eps":
        A = A + 1e-3 * (1.0 - np.eye(len(idx)))
    return A

def run(idx, k, variant, seed=42):
    lab = SpectralClustering(k, affinity="precomputed", assign_labels="cluster_qr", random_state=seed).fit_predict(graph(idx, variant))
    if variant == "merge":
        Ui = U[idx]; minsize = max(int(0.02 * len(idx)), 2)
        while True:
            sizes = np.bincount(lab, minlength=lab.max() + 1); small = [c for c in np.unique(lab) if sizes[c] < minsize]
            if not small or len(np.unique(lab)) <= 2: break
            c = small[0]
            cents = {d: Ui[lab == d].mean(0) for d in np.unique(lab)}
            cents = {d: v / np.linalg.norm(v) for d, v in cents.items()}
            tgt = max((d for d in cents if d != c), key=lambda d: cents[c] @ cents[d]); lab[lab == c] = tgt
        _, lab = np.unique(lab, return_inverse=True)
    return lab

def purity(lab, m):
    return sum(max((y[m & (lab == c)] == v).sum() for v in ("punjabi", "hindi")) for c in np.unique(lab[m])) / m.sum()

rng = np.random.default_rng(1)
subs = [np.sort(rng.choice(n, int(0.8 * n), replace=False)) for _ in range(4)]
lm = y != ""
print(f"contents: {n}; labelled: {lm.sum()}")
head = f"{'variant':<7}{'k':>3}{'->':>3} {'sizes':<38}|{'purity':>7}{'NMI':>7}{'stab':>6}"
print(head + "\n" + "-" * len(head))
for k in (8, 12):
    for variant in ("base", "knn20", "eps", "merge"):
        lab = run(np.arange(n), k, variant)
        sizes = sorted(np.bincount(lab), reverse=True)
        runs = [(idx, run(idx, k, variant, seed=s)) for s, idx in enumerate(subs)]
        aris = []
        for i in range(4):
            for j in range(i + 1, 4):
                (ia, la), (ib, lb) = runs[i], runs[j]
                c = np.intersect1d(ia, ib); pa = dict(zip(ia, la)); pb = dict(zip(ib, lb))
                aris.append(adjusted_rand_score([pa[t] for t in c], [pb[t] for t in c]))
        print(f"{variant:<7}{k:>3}{len(sizes):>3} {str([int(s) for s in sizes]):<38}|{purity(lab, lm):7.3f}{normalized_mutual_info_score(y[lm], lab[lm]):7.3f}{np.mean(aris):6.2f}", flush=True)
    print()
