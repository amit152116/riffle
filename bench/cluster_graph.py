"""Neighbour-graph clustering on the EffNet embedding (mutual-proximity kNN graph).

Compared with k-means on the embedding and on riffle's current features at the
same k: label purity/NMI, source NMI (lower better), known duplicates kept
together, and stability (ARI between clusterings of 80% subsamples).
"""
import sys, warnings
from pathlib import Path
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import evaluate as ev
import cluster_eval as ce
from cluster_fixed_k import rep, purity

rows, feats = ev.load()
n = len(rows)
y = np.array([r["label"] or "" for r in rows]); src = np.array([r["source"] for r in rows])
groups = {}
for i, r in enumerate(rows):
    if "dup" in r["roles"] and r["group"] is not None: groups.setdefault(r["group"], []).append(i)
pairs = [(a, b) for g in groups.values() for i, a in enumerate(g) for b in g[i + 1:]]
rng = np.random.default_rng(1)
subs = [np.sort(rng.choice(n, int(0.8 * n), replace=False)) for _ in range(4)]

def distance(fs):
    E = np.stack([f["effnet_emb"] for f in fs]).astype(float)
    return ev.mutual_proximity(ev.cosine_dist(E))

def affinity(D, k=10):
    """Symmetric kNN graph; edge weight falls with the mutual-proximity distance."""
    m = len(D); A = np.zeros((m, m))
    Dn = D.copy(); np.fill_diagonal(Dn, np.inf)
    nn = np.argsort(Dn, axis=1)[:, :k]
    sigma = np.median(Dn[np.arange(m)[:, None], nn])
    for i in range(m):
        A[i, nn[i]] = np.exp(-(Dn[i, nn[i]] / sigma) ** 2)
    return np.maximum(A, A.T)

def fit(method, fs, k, seed=0):
    if method == "kmeans-eff": return KMeans(k, n_init=10, random_state=seed).fit_predict(rep("effnet", fs))
    if method == "kmeans-prod": return KMeans(k, n_init=10, random_state=seed).fit_predict(rep("prod", fs))
    D = distance(fs)
    if method == "spectral-kNN": return SpectralClustering(k, affinity="precomputed", random_state=seed, assign_labels="cluster_qr").fit_predict(affinity(D))
    if method == "avg-link-MP": return AgglomerativeClustering(k, metric="precomputed", linkage="average").fit_predict(D)
    if method == "complete-MP": return AgglomerativeClustering(k, metric="precomputed", linkage="complete").fit_predict(D)

head = f"{'method':<14}{'k':>3} |{'purity':>7}{'NMI':>7}{'srcNMI':>8}{'sizes':>13} |{'dup-tog':>8}{'stab-ARI':>9}"
print(f"tracks: {n}\n" + head + "\n" + "-" * len(head))
for k in (6, 10):
    for method in ("kmeans-prod", "kmeans-eff", "spectral-kNN", "avg-link-MP", "complete-MP"):
        lab = fit(method, feats, k); lm = y != ""
        sizes = sorted(np.bincount(lab), reverse=True)
        together = np.mean([lab[a] == lab[b] for a, b in pairs])
        runs = [(idx, fit(method, [feats[i] for i in idx], k, seed=s)) for s, idx in enumerate(subs)]
        aris = []
        for i in range(4):
            for j in range(i + 1, 4):
                (ia, la), (ib, lb) = runs[i], runs[j]
                c = np.intersect1d(ia, ib); pa = dict(zip(ia, la)); pb = dict(zip(ib, lb))
                aris.append(adjusted_rand_score([pa[t] for t in c], [pb[t] for t in c]))
        print(f"{method:<14}{k:>3} |{purity(lab, lm):7.3f}{normalized_mutual_info_score(y[lm], lab[lm]):7.3f}"
              f"{normalized_mutual_info_score(src, lab):8.3f}{str(sizes[0]) + '/' + str(sizes[-1]):>13} |{together:8.2f}{np.mean(aris):9.2f}", flush=True)
    print()
