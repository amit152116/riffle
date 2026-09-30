"""Similarity benchmark on the cached subset (read-only).

Representations compared (all give an n x n distance matrix, lower = closer)
  prod         riffle's combined score (MFCC-mean cosine + BPM + key + energy)
  prod-rank    same components, each converted to a percentile rank first
  mfcc-cos     cosine on raw MFCC means only
  mfcc-z       z-scored MFCC mean+std, MFCC0 dropped, Euclidean
  effnet       Discogs-EffNet embedding, cosine
  effnet-pca   EffNet, PCA-whitened to 64 dims, cosine
  effnet-sty   EffNet's 400 Discogs style probabilities, cosine
  musicnn      musicnn embedding, cosine
  hybrid       EffNet + riffle's BPM/key/energy components, percentile-ranked

Metrics
  dup       for tracks of known same-recording groups: how highly the other
            copy ranks among ALL other tracks (hit@1, hit@5, MRR)
  edit      same, for same-song pairs that differ by intro/outro
  label     precision@10 of same folder-label neighbours (chance = 0.5),
            excluding same-artist neighbours
  xsrc      the same, but neighbours restricted to the OTHER source (YouTube
            rip vs older library file): does style survive a change of source?
  srcbias   share of 10-NN with the same source as the query (chance = 0.5)
  hubness   skew of 10-occurrence counts, share never seen in any 10-NN
            list, and the largest count (mean would be 10)
Each is reported raw and after mutual proximity (Schnitzer et al.).

    python bench/evaluate.py
"""
import json
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm, skew

from riffle import similarity

BENCH = Path(os.path.expanduser("~/riffle-bench"))
DB = os.path.expanduser("~/riffle-mymusic.sqlite")
K = 10


# ---------------------------------------------------------------- loading ---
def load():
    subset = json.load(open(BENCH / "subset.json"))
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    artist = {r[0]: (r[1] or "").strip().lower() or None for r in conn.execute(
        "SELECT id, tag_artist FROM track")}
    rows, feats = [], []
    for t in subset:
        f = BENCH / "cache" / f"{t['id']}.npz"
        if not f.exists():
            continue
        with np.load(f, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        if "mfcc_mean" not in d or "effnet_emb" not in d:
            continue
        t["artist"] = artist.get(t["id"])
        rows.append(t)
        feats.append(d)
    return rows, feats


# -------------------------------------------------------------- distances ---
def cosine_dist(X):
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return 1.0 - X @ X.T


def euclid_dist(X):
    sq = (X ** 2).sum(1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))


def pct_rank(D):
    """Percentile rank of each off-diagonal distance within the whole matrix."""
    n = len(D)
    iu = np.triu_indices(n, 1)
    vals = D[iu]
    ranks = vals.argsort().argsort() / (len(vals) - 1)
    R = np.zeros_like(D)
    R[iu] = ranks
    return R + R.T


def production_components(feats):
    n = len(feats)
    bpm = np.array([float(f["bpm"]) for f in feats])
    valid = bpm[bpm > 0]
    q25, q75 = np.percentile(valid, [25, 75])
    scale = max(float(q75 - q25), 10.0)              # compute_bpm_spread()
    C = {k: np.zeros((n, n)) for k in ("mfcc", "bpm", "key", "energy")}
    fd = [{"bpm": float(f["bpm"]), "key_name": str(f["key_name"]),
           "scale": str(f["scale"]), "energy": float(f["energy"]),
           "mfcc_mean": f["mfcc_mean"]} for f in feats]
    for i in range(n):
        for j in range(i + 1, n):
            c = similarity.score_components(fd[i], fd[j], bpm_scale=scale)
            for k, name in (("mfcc", "mfcc_norm"), ("bpm", "bpm_norm"),
                            ("key", "key_norm"), ("energy", "energy_norm")):
                v = c[name]
                C[k][i, j] = C[k][j, i] = 0.0 if v is None else v
    return C, fd, scale


def combine(components, weights):
    return sum(weights[k] * components[k] for k in weights)


def build_matrices(feats):
    M = {}
    stack = lambda key: np.stack([np.asarray(f[key], dtype=np.float64) for f in feats])
    C, fd, _ = production_components(feats)
    W = similarity.DEFAULT_WEIGHTS
    M["prod"] = combine(C, W)
    Cr = {k: pct_rank(v) for k, v in C.items()}
    M["prod-rank"] = combine(Cr, W)
    M["mfcc-cos"] = cosine_dist(stack("mfcc_mean"))
    Xm = np.hstack([stack("mfcc_mean")[:, 1:], stack("mfcc_std")[:, 1:]])
    Xm = (Xm - Xm.mean(0)) / (Xm.std(0) + 1e-9)
    M["mfcc-z"] = euclid_dist(Xm)
    E = stack("effnet_emb")
    M["effnet"] = cosine_dist(E)
    from sklearn.decomposition import PCA
    P = PCA(n_components=min(64, len(E) - 1), whiten=True, random_state=0).fit_transform(E)
    M["effnet-pca"] = cosine_dist(P)
    M["effnet-sty"] = cosine_dist(stack("effnet_sty"))
    M["musicnn"] = cosine_dist(stack("musicnn_emb"))
    Ce = {"mfcc": pct_rank(M["effnet"]), "bpm": Cr["bpm"], "key": Cr["key"],
          "energy": Cr["energy"]}
    M["hybrid"] = combine(Ce, W)
    return M


def mutual_proximity(D):
    """Gaussian mutual proximity distance (Schnitzer et al. 2012)."""
    n = len(D)
    off = ~np.eye(n, dtype=bool)
    mu = np.array([D[i][off[i]].mean() for i in range(n)])
    sd = np.array([D[i][off[i]].std() + 1e-12 for i in range(n)])
    # P(X_i > d_ij), with X_i ~ N(mu_i, sd_i)
    P = 1.0 - norm.cdf((D - mu[:, None]) / sd[:, None])
    MP = P * P.T
    out = 1.0 - MP
    np.fill_diagonal(out, 0.0)
    return out


# ---------------------------------------------------------------- metrics ---
def ranks_of_positives(D, rows, role_prefix):
    groups = defaultdict(list)
    for i, t in enumerate(rows):
        if t["group"] is not None and any(r.startswith(role_prefix) for r in t["roles"]):
            groups[t["group"]].append(i)
    res = []
    for g, idx in groups.items():
        if len(idx) < 2:
            continue
        for i in idx:
            d = D[i].copy(); d[i] = np.inf
            order = np.argsort(d)
            pos = [j for j in idx if j != i]
            best = min(int(np.where(order == j)[0][0]) for j in pos) + 1
            cross = any(rows[j]["source"] != rows[i]["source"] for j in pos)
            res.append((best, cross))
    return res


def summarize_ranks(res, cross_only=False):
    r = [b for b, c in res if (c or not cross_only)]
    if not r:
        return "   -   "
    r = np.array(r)
    return f"{np.mean(r <= 1):.2f}/{np.mean(r <= 5):.2f}/{np.mean(1 / r):.2f}"


def label_metrics(D, rows):
    idx = [i for i, t in enumerate(rows) if t["label"] and
           ({"punjabi", "hindi"} & set(t["roles"]))]
    lab = np.array([rows[i]["label"] for i in idx])
    src = np.array([rows[i]["source"] for i in idx])
    art = [rows[i]["artist"] for i in idx]
    m = len(idx)
    Dl = D[np.ix_(idx, idx)].copy()
    np.fill_diagonal(Dl, np.inf)
    for a in range(m):                                # artist filter
        if art[a]:
            for b in range(m):
                if b != a and art[b] == art[a]:
                    Dl[a, b] = np.inf
    prec, xprec, bias, nn_sets = [], [], [], []
    for a in range(m):
        order = np.argsort(Dl[a])
        nn = [b for b in order if np.isfinite(Dl[a, b])][:K]
        prec.append(np.mean(lab[nn] == lab[a]))
        bias.append(np.mean(src[nn] == src[a]))
        other = [b for b in order if np.isfinite(Dl[a, b]) and src[b] != src[a]][:K]
        xprec.append(np.mean(lab[other] == lab[a]))
        nn_sets.append(nn)
    counts = np.zeros(m)
    for nn in nn_sets:
        for b in nn:
            counts[b] += 1
    return (np.mean(prec), np.mean(xprec), np.mean(bias),
            float(skew(counts)), float(np.mean(counts == 0)), int(counts.max()))


def report():
    rows, feats = load()
    print(f"tracks with baseline+embedding features: {len(rows)}")
    if len(rows) < 100:
        sys.exit("not enough cached tracks yet")
    roles = defaultdict(int)
    for t in rows:
        for r in t["roles"]:
            roles[r] += 1
    print("roles:", dict(roles))
    M = build_matrices(feats)
    head = (f"{'representation':<12} {'mode':<4} | {'dup h1/h5/MRR':<15}{'dup x-src':<15}"
            f"{'edit h1/h5/MRR':<16}| {'label@10':>8}{'xsrc@10':>9}{'srcbias':>9} |"
            f"{'skew':>7}{'never%':>8}{'max':>5}")
    print("\n" + head + "\n" + "-" * len(head))
    for name, D in M.items():
        for mode, Dm in (("raw", D), ("MP", mutual_proximity(D))):
            dup = ranks_of_positives(Dm, rows, "dup")
            edit = ranks_of_positives(Dm, rows, "edit")
            lp, xp, sb, sk, nv, mx = label_metrics(Dm, rows)
            print(f"{name:<12} {mode:<4} | {summarize_ranks(dup):<15}"
                  f"{summarize_ranks(dup, True):<15}{summarize_ranks(edit):<16}|"
                  f"{lp:8.3f}{xp:9.3f}{sb:9.3f} |{sk:7.2f}{100*nv:8.1f}{mx:5d}")


if __name__ == "__main__":
    report()
