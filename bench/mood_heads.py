"""Do Essentia's Discogs-EffNet classification heads add usable signal?

Runs each head on the cached EffNet embeddings (no audio decoding) and checks,
read-only, against what riffle already measures. There is no mood ground truth
in the benchmark subset (its labels are hindi/punjabi), so this measures
  - output spread: a head that is ~constant is useless,
  - agreement with existing features (Spearman): sanity, not proof,
  - redundancy: how much of each head is predictable from existing features
    (5-fold ridge R^2); low R^2 = new information (or noise; not separable
    without ground truth),
  - the tracks at each end of each head, for a human to judge.

    python bench/mood_heads.py HEADS_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict

BENCH = Path("~/riffle-bench").expanduser()
# (head, index of the positive class in the head's `classes` list is read from json)
HEADS = ["mood_happy", "mood_sad", "mood_aggressive", "mood_relaxed",
         "mood_party", "danceability", "voice_instrumental", "tonal_atonal",
         "mood_acoustic", "mood_electronic"]
POSITIVE = {"mood_happy": "happy", "mood_sad": "sad",
            "mood_aggressive": "aggressive", "mood_relaxed": "relaxed",
            "mood_party": "party", "danceability": "danceable",
            "voice_instrumental": "voice", "tonal_atonal": "tonal",
            "mood_acoustic": "acoustic", "mood_electronic": "electronic"}
EXISTING = ["bpm", "energy", "loudness_lufs", "danceability", "spectral_centroid",
            "onset_rate", "dynamic_complexity", "dissonance", "zcr", "key_strength"]


def load():
    subset = json.load(open(BENCH / "subset.json"))
    rows, embs, feats = [], [], []
    for t in subset:
        f = BENCH / "cache" / f"{t['id']}.npz"
        if not f.exists():
            continue
        with np.load(f, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        if "effnet_emb" not in d or "energy" not in d:
            continue
        rows.append(t)
        embs.append(d["effnet_emb"])
        f2 = {k: float(d[k]) for k in EXISTING}
        f2["major"] = 1.0 if str(d["scale"]) == "major" else 0.0
        feats.append(f2)
    return rows, np.stack(embs).astype(np.float32), feats


def run_head(heads_dir, name, embs):
    import essentia.standard as es
    meta = json.load(open(heads_dir / f"{name}-discogs-effnet-1.json"))
    pos = meta["classes"].index(POSITIVE[name])
    model = es.TensorflowPredict2D(
        graphFilename=str(heads_dir / f"{name}-discogs-effnet-1.pb"),
        input="model/Placeholder", output="model/Softmax")
    # Essentia's predictor takes a (batch, dim) matrix.
    return np.asarray(model(embs))[:, pos]


def main():
    heads_dir = Path(sys.argv[1])
    rows, embs, feats = load()
    print(f"{len(rows)} tracks with embeddings and features\n")
    P = {h: run_head(heads_dir, h, embs) for h in HEADS}

    names = list(feats[0])
    F = np.array([[f[k] for k in names] for f in feats])

    print(f"{'head':20} {'mean':>5} {'std':>5} {'p10':>5} {'p90':>5} "
          f"{'R2|existing':>11}")
    Fz = (F - F.mean(0)) / (F.std(0) + 1e-9)
    for h in HEADS:
        p = P[h]
        pred = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 3, 12)), Fz, p, cv=5)
        r2 = 1 - ((p - pred) ** 2).sum() / ((p - p.mean()) ** 2).sum()
        print(f"{h:20} {p.mean():5.2f} {p.std():5.2f} {np.percentile(p,10):5.2f} "
              f"{np.percentile(p,90):5.2f} {r2:11.2f}")

    print("\nSpearman vs existing features")
    print(f"{'head':20} " + " ".join(f"{n[:9]:>9}" for n in names))
    for h in HEADS:
        print(f"{h:20} " + " ".join(
            f"{spearmanr(P[h], F[:, i])[0]:9.2f}" for i in range(len(names))))

    print("\nSpearman between heads (|rho| > 0.6 only)")
    for i, a in enumerate(HEADS):
        for b in HEADS[i + 1:]:
            r = spearmanr(P[a], P[b])[0]
            if abs(r) > 0.6:
                print(f"  {a} ~ {b}: {r:.2f}")

    print("\nBy language label (mean of head output)")
    for lab in ("hindi", "punjabi"):
        idx = [i for i, t in enumerate(rows) if t.get("label") == lab]
        print(f"  {lab:8} n={len(idx)} " + " ".join(
            f"{h.split('_')[-1][:6]}={P[h][idx].mean():.2f}" for h in HEADS))

    print("\nExtremes (file name) for eyeballing")
    for h in HEADS:
        order = np.argsort(P[h])
        def nm(i): return f"{Path(rows[i]['path']).stem[:38]} ({P[h][i]:.2f})"
        print(f"{h}\n  low : " + " | ".join(nm(i) for i in order[:4]) +
              "\n  high: " + " | ".join(nm(i) for i in order[-4:]))


if __name__ == "__main__":
    main()
