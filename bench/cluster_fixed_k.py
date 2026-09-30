"""Fair clustering comparison: same k for every representation (k-means)."""
import sys, warnings
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "bench")
import cluster_eval as ce
warnings.filterwarnings("ignore")

rows, feats = ce.load()
n = len(rows)
y = np.array([r["label"] or "" for r in rows]); src = np.array([r["source"] for r in rows])
groups = {}
for i, r in enumerate(rows):
    if "dup" in r["roles"] and r["group"] is not None: groups.setdefault(r["group"], []).append(i)
pairs = [(a, b) for g in groups.values() for i, a in enumerate(g) for b in g[i + 1:]]

def rep(name, fs):
    if name == "prod": return ce.features_prod(fs)
    if name == "mfcc-z":
        X = np.hstack([np.stack([f["mfcc_mean"] for f in fs])[:, 1:], np.stack([f["mfcc_std"] for f in fs])[:, 1:]])
        return StandardScaler().fit_transform(X)
    def pca(key, dims=32):
        E = np.stack([f[key] for f in fs]).astype(float)
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        return StandardScaler().fit_transform(PCA(dims, random_state=0).fit_transform(E))
    return {"effnet": lambda: pca("effnet_emb"), "effnet-sty": lambda: pca("effnet_sty"),
            "musicnn": lambda: pca("musicnn_emb")}[name]()

def purity(lab, mask):
    tot = 0
    for k in set(lab[mask]): tot += max((y[mask & (lab == k)] == c).sum() for c in ("punjabi", "hindi"))
    return tot / mask.sum()

rng = np.random.default_rng(1)
subs = [np.sort(rng.choice(n, int(0.8 * n), replace=False)) for _ in range(4)]

def main():
    head = f"{'rep':<11}{'k':>3} |{'purity':>7}{'NMI':>7}{'ARI':>7}{'srcNMI':>8} |{'dup-tog':>8}{'stab-ARI':>9}"
    print(head + "\n" + "-" * len(head))
    for k in (4, 8, 12):
        for name in ("prod", "mfcc-z", "effnet", "effnet-sty", "musicnn"):
            X = rep(name, feats)
            lab = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
            lm = y != ""
            together = np.mean([lab[a] == lab[b] for a, b in pairs])
            runs = [(idx, KMeans(n_clusters=k, n_init=10, random_state=s).fit_predict(rep(name, [feats[i] for i in idx]))) for s, idx in enumerate(subs)]
            aris = []
            for i in range(4):
                for j in range(i + 1, 4):
                    (ia, la), (ib, lb) = runs[i], runs[j]
                    c = np.intersect1d(ia, ib); pa = dict(zip(ia, la)); pb = dict(zip(ib, lb))
                    aris.append(adjusted_rand_score([pa[t] for t in c], [pb[t] for t in c]))
            print(f"{name:<11}{k:>3} |{purity(lab, lm):7.3f}{normalized_mutual_info_score(y[lm], lab[lm]):7.3f}"
                  f"{adjusted_rand_score(y[lm], lab[lm]):7.3f}{normalized_mutual_info_score(src, lab):8.3f} |"
                  f"{together:8.2f}{np.mean(aris):9.2f}", flush=True)
        print()

if __name__ == "__main__":
    main()
