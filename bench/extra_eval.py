"""Do the candidate extra features (bench/extra_features.py) earn a place?

Three read-only checks on the benchmark subset. There is no mood ground truth
except the Y/N marks in docs/benchmarks/mood-heads-judging*.md, so:

  1. stability   for known same-recording pairs (a YouTube rip and a library
                 file), how far apart the two copies' values are relative to
                 random pairs. Lower = the feature describes the song, not the
                 file. Source-dependent features (e.g. treble share on a
                 lossy rip) fail here.
  2. redundancy  5-fold ridge R^2 of the feature from the production features.
                 High = nothing new.
  3. judged      AUC of each feature against the hand-marked Y/N labels, and
                 cross-validated AUC of production features vs production +
                 extras (logistic, repeated CV). The labelled tracks were
                 picked by head score, so this favours features that track
                 the heads; treat it as a lower-quality check, n is small.

    python bench/extra_eval.py
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

BENCH = Path("~/riffle-bench").expanduser()
DOCS = Path(__file__).resolve().parent.parent / "docs" / "benchmarks"
PROD = ["bpm", "energy", "loudness_lufs", "danceability", "spectral_centroid",
        "onset_rate", "dynamic_complexity", "dissonance", "zcr", "key_strength"]
EXTRA = ["rolloff85", "rolloff95", "flux_mean", "flux_std", "contrast_mean",
         "bass", "lowmid", "mid", "highmid", "treble", "rms_mean", "rms_std",
         "dyn_range", "chroma_entropy"]
HEADS = ["mood_happy", "mood_sad", "mood_relaxed", "mood_party",
         "danceability", "mood_electronic"]


def load():
    subset = json.load(open(BENCH / "subset.json"))
    rows, X = [], []
    for t in subset:
        a, b = BENCH / "cache" / f"{t['id']}.npz", BENCH / "extra" / f"{t['id']}.npz"
        if not (a.exists() and b.exists()):
            continue
        za, zb = np.load(a), np.load(b)
        if "energy" not in za.files:
            continue
        rows.append(t)
        X.append([float(za[k]) for k in PROD] + [float(zb[k]) for k in EXTRA])
    return rows, np.array(X)


def stability(rows, Z, names):
    groups = defaultdict(list)
    for i, t in enumerate(rows):
        if t.get("group") is not None and "dup" in (t.get("roles") or []):
            groups[t["group"]].append(i)
    pairs = [(g[0], j) for g in groups.values() for j in g[1:]
             if rows[g[0]].get("source") != rows[j].get("source")]
    rng = np.random.default_rng(0)
    rand = rng.integers(0, len(rows), size=(2000, 2))
    print(f"\n1. stability: {len(pairs)} cross-source copy pairs "
          f"(ratio = median |diff| copies / median |diff| random; lower is better)")
    out = {}
    for k, n in enumerate(names):
        d = np.median([abs(Z[a, k] - Z[b, k]) for a, b in pairs])
        r = np.median(np.abs(Z[rand[:, 0], k] - Z[rand[:, 1], k]))
        out[n] = d / r
    for n, v in sorted(out.items(), key=lambda kv: kv[1]):
        print(f"   {n:20} {v:5.2f}{'   (extra)' if n in EXTRA else ''}")


def redundancy(Z, names):
    print("\n2. redundancy: R^2 of each extra from the production features (5-fold ridge)")
    P = Z[:, :len(PROD)]
    for n in EXTRA:
        y = Z[:, names.index(n)]
        pred = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 3, 12)), P, y, cv=5)
        print(f"   {n:20} {1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum():5.2f}")


def parse_sheets():
    lab = defaultdict(dict)                      # head -> (stem, source) -> 0/1
    for f in sorted(DOCS.glob("mood-heads-judging*.md")):
        head = None
        for line in f.read_text().splitlines():
            if line.startswith("## "):
                head = line[3:].split(" ")[0].strip()
            elif re.match(r"\|\s*\d\.\d\d", line) and head in HEADS:
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                call = cells[-1].upper()
                m = re.match(r"(.*)\s\[(\w+)\]$", cells[1])
                if call in ("Y", "N") and m:
                    lab[head][(m.group(1), m.group(2))] = 1 if call == "Y" else 0
    return lab


def judged(rows, Z, names):
    idx = {(Path(t["path"]).stem, t.get("source")): i for i, t in enumerate(rows)}
    lab = parse_sheets()
    print("\n3. judged labels (Y=1). single-feature AUC, |AUC-0.5| shown as strength")
    for head in HEADS:
        pts = [(idx[k], v) for k, v in lab[head].items() if k in idx]
        y = np.array([v for _, v in pts]); ii = [i for i, _ in pts]
        if len(set(y)) < 2 or min(y.sum(), len(y) - y.sum()) < 4:
            print(f"\n   {head}: n={len(y)} (Y={y.sum()}) too unbalanced, skipped")
            continue
        A = Z[ii]
        print(f"\n   {head}: n={len(y)} (Y={y.sum()}, N={len(y) - y.sum()})")
        auc = {n: roc_auc_score(y, A[:, k]) for k, n in enumerate(names)}
        top = sorted(auc.items(), key=lambda kv: -abs(kv[1] - 0.5))[:5]
        print("     best single features: " + ", ".join(
            f"{n}{'*' if n in EXTRA else ''} {a:.2f}" for n, a in top))
        cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=25, random_state=0)
        res = {}
        for label, cols in (("production", range(len(PROD))),
                            ("prod+extras", range(len(names)))):
            aucs = []
            for tr, te in cv.split(A, y):
                m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=500))
                m.fit(A[tr][:, list(cols)], y[tr])
                aucs.append(roc_auc_score(y[te], m.predict_proba(A[te][:, list(cols)])[:, 1]))
            res[label] = np.mean(aucs)
        print(f"     CV AUC: production {res['production']:.2f} -> "
              f"prod+extras {res['prod+extras']:.2f}")


def main():
    rows, Z = load()
    names = PROD + EXTRA
    print(f"{len(rows)} tracks with production + extra features")
    stability(rows, Z, names)
    redundancy(Z, names)
    judged(rows, Z, names)


if __name__ == "__main__":
    sys.exit(main())
