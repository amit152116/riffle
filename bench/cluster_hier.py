"""Do deterministic methods (Ward hierarchical, GMM) stabilise EffNet clusters?"""
import sys, warnings
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
sys.path.insert(0, "bench")
import cluster_eval as ce
from cluster_fixed_k import rep, purity, rows, feats, y, src, pairs, subs, n
warnings.filterwarnings("ignore")

def fit(method, X, k, seed=0):
    if method == "ward": return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    if method == "gmm": return GaussianMixture(n_components=k, covariance_type="diag", n_init=5, random_state=seed).fit_predict(X)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)

head = f"{'rep+method':<20}{'k':>3} |{'purity':>7}{'NMI':>7}{'srcNMI':>8} |{'dup-tog':>8}{'stab-ARI':>9}"
print("\n" + head + "\n" + "-" * len(head))
for k in (6, 10):
    for name, method in (("prod", "kmeans"), ("prod", "ward"), ("effnet", "kmeans"), ("effnet", "ward"), ("effnet", "gmm"), ("effnet-sty", "ward")):
        X = rep(name, feats); lab = fit(method, X, k); lm = y != ""
        together = np.mean([lab[a] == lab[b] for a, b in pairs])
        runs = [(idx, fit(method, rep(name, [feats[i] for i in idx]), k, seed=s)) for s, idx in enumerate(subs)]
        aris = []
        for i in range(4):
            for j in range(i + 1, 4):
                (ia, la), (ib, lb) = runs[i], runs[j]
                c = np.intersect1d(ia, ib); pa = dict(zip(ia, la)); pb = dict(zip(ib, lb))
                aris.append(adjusted_rand_score([pa[t] for t in c], [pb[t] for t in c]))
        print(f"{name+'+'+method:<20}{k:>3} |{purity(lab, lm):7.3f}{normalized_mutual_info_score(y[lm], lab[lm]):7.3f}"
              f"{normalized_mutual_info_score(src, lab):8.3f} |{together:8.2f}{np.mean(aris):9.2f}", flush=True)
    print()
