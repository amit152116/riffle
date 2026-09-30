"""Do the low-redundancy extra features improve the similarity benchmark?

Reuses bench/evaluate.py (same subset, same metrics) and adds one component
built from the extras (z-scored Euclidean, then percentile-ranked like the
other hybrid components) at several weights. Controls:
  extras-only     the component by itself
  shuffled        the same component with rows permuted (noise of equal size);
                  any gain that a shuffled column also gets is not real
Baselines are the MFCC-based production score ("prod"), the EffNet cosine
("effnet") and EffNet + BPM/key/energy ("hybrid"), as in bench/evaluate.py.

    python bench/extra_sim_eval.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev                                       # noqa: E402

BENCH = ev.BENCH
LOW_REDUNDANCY = ["dyn_range", "bass", "lowmid", "mid", "chroma_entropy",
                  "contrast_mean"]


def extras_matrix(rows):
    X = []
    for t in rows:
        with np.load(BENCH / "extra" / f"{t['id']}.npz") as z:
            X.append([float(z[k]) for k in LOW_REDUNDANCY])
    X = np.array(X)
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


def main():
    rows, feats = ev.load()
    keep = [i for i, t in enumerate(rows) if (BENCH / "extra" / f"{t['id']}.npz").exists()]
    rows, feats = [rows[i] for i in keep], [feats[i] for i in keep]
    print(f"{len(rows)} tracks")
    M = ev.build_matrices(feats)
    X = extras_matrix(rows)
    Dx = ev.pct_rank(ev.euclid_dist(X))
    rng = np.random.default_rng(0)
    Dshuf = ev.pct_rank(ev.euclid_dist(X[rng.permutation(len(X))]))

    C, _, _ = ev.production_components(feats)
    Cr = {k: ev.pct_rank(v) for k, v in C.items()}
    W = ev.similarity.DEFAULT_WEIGHTS
    Ce = {"mfcc": ev.pct_rank(M["effnet"]), "bpm": Cr["bpm"], "key": Cr["key"],
          "energy": Cr["energy"]}

    reps = {"prod": M["prod-rank"], "effnet": M["effnet"], "hybrid": M["hybrid"],
            "extras-only": Dx, "shuffled-only": Dshuf}
    for w in (0.1, 0.2, 0.4):
        reps[f"hybrid+ex w={w}"] = ev.combine(Ce, W) + w * Dx
        reps[f"hybrid+shuf w={w}"] = ev.combine(Ce, W) + w * Dshuf
        reps[f"effnet+ex w={w}"] = ev.pct_rank(M["effnet"]) + w * Dx
        reps[f"effnet+shuf w={w}"] = ev.pct_rank(M["effnet"]) + w * Dshuf
    reps["prod+ex w=0.2"] = ev.combine(Cr, W) + 0.2 * Dx

    head = (f"{'representation':<20} {'mode':<4} | {'dup h1/h5/MRR':<15}{'dup x-src':<15}"
            f"{'edit h1/h5/MRR':<16}| {'label@10':>8}{'xsrc@10':>9}{'srcbias':>9}")
    print("\n" + head + "\n" + "-" * len(head))
    for name, D in reps.items():
        for mode, Dm in (("raw", D), ("MP", ev.mutual_proximity(D))):
            dup = ev.ranks_of_positives(Dm, rows, "dup")
            edit = ev.ranks_of_positives(Dm, rows, "edit")
            lp, xp, sb, *_ = ev.label_metrics(Dm, rows)
            print(f"{name:<20} {mode:<4} | {ev.summarize_ranks(dup):<15}"
                  f"{ev.summarize_ranks(dup, True):<15}{ev.summarize_ranks(edit):<16}|"
                  f"{lp:8.3f}{xp:9.3f}{sb:9.3f}")


if __name__ == "__main__":
    main()
