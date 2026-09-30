"""How should the number of spectral clusters be chosen? Try eigengap and
silhouette (on the mutual-proximity distance) and see whether each picks a
consistent k across subsamples, and how purity/stability behave over k."""
import sys, warnings
from pathlib import Path
import numpy as np
from scipy.sparse.csgraph import laplacian
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import cluster_graph as cg
from cluster_fixed_k import purity

rows, feats, n, y = cg.rows, cg.feats, cg.n, cg.y
def eigengap(A, kmin=4, kmax=20):
    L = laplacian(A, normed=True)
    w = np.sort(np.linalg.eigvalsh(L))[:kmax + 1]
    gaps = np.diff(w)[kmin - 1:kmax]
    return kmin + int(np.argmax(gaps)), w
def sil_k(D, A, kmin=4, kmax=20):
    best = (-2, None)
    for k in range(kmin, kmax + 1):
        lab = SpectralClustering(k, affinity="precomputed", random_state=0, assign_labels="cluster_qr").fit_predict(A)
        s = silhouette_score(D, lab, metric="precomputed")
        if s > best[0]: best = (s, k)
    return best[1]
print("k chosen by each rule on the full subset and on four 80% subsamples")
for name, idxs in [("all", np.arange(n))] + [(f"sub{i}", s) for i, s in enumerate(cg.subs)]:
    fs = [feats[i] for i in idxs]; D = cg.distance(fs); A = cg.affinity(D)
    ke, _ = eigengap(A); ks = sil_k(D, A)
    print(f"  {name:<5} eigengap -> {ke:>2} | silhouette -> {ks:>2}")
print("\nspectral-kNN quality by k (purity on labelled, stability over 4 subsamples)")
print(f"{'k':>3} {'purity':>7} {'NMI':>6} {'stab':>6} {'dup-tog':>8} {'sizes max/min':>14}")
lm = y != ""
for k in (4, 6, 8, 10, 12, 16, 20):
    lab = cg.fit("spectral-kNN", feats, k)
    runs = [(idx, cg.fit("spectral-kNN", [feats[i] for i in idx], k, seed=s)) for s, idx in enumerate(cg.subs)]
    aris = []
    for i in range(4):
        for j in range(i + 1, 4):
            (ia, la), (ib, lb) = runs[i], runs[j]
            c = np.intersect1d(ia, ib); pa = dict(zip(ia, la)); pb = dict(zip(ib, lb))
            aris.append(adjusted_rand_score([pa[t] for t in c], [pb[t] for t in c]))
    sz = np.bincount(lab)
    tog = np.mean([lab[a] == lab[b] for a, b in cg.pairs])
    print(f"{k:>3} {purity(lab, lm):7.3f} {normalized_mutual_info_score(y[lm], lab[lm]):6.3f} {np.mean(aris):6.2f} {tog:8.2f} {sz.max():>8}/{sz.min():<4}", flush=True)
