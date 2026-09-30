"""Does combining embeddings beat Discogs-EffNet alone? (same metrics as evaluate.py)"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import evaluate as ev

rows, feats = ev.load()
stack = lambda k: np.stack([np.asarray(f[k], dtype=np.float64) for f in feats])
unit = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
cos = lambda X: ev.cosine_dist(X)
eff, mus, sty = cos(stack("effnet_emb")), cos(stack("musicnn_emb")), cos(stack("effnet_sty"))
r = ev.pct_rank
M = {
    "effnet": eff,
    "musicnn": mus,
    "eff+mus (concat)": cos(np.hstack([unit(stack("effnet_emb")), unit(stack("musicnn_emb"))])),
    "eff+mus (rank avg)": (r(eff) + r(mus)) / 2,
    "eff+mus+sty (rank avg)": (r(eff) + r(mus) + r(sty)) / 3,
    "0.7 eff + 0.3 mus": 0.7 * r(eff) + 0.3 * r(mus),
    "eff+sty (rank avg)": (r(eff) + r(sty)) / 2,
}
head = (f"{'representation':<24}{'mode':<4}| {'dup h1/h5/MRR':<15}{'edit h1/h5/MRR':<16}| "
        f"{'label@10':>8}{'xsrc@10':>9}{'srcbias':>9} |{'skew':>6}{'never%':>8}")
print(f"tracks: {len(rows)}\n" + head + "\n" + "-" * len(head))
for name, D in M.items():
    for mode, Dm in (("raw", D), ("MP", ev.mutual_proximity(D))):
        dup = ev.ranks_of_positives(Dm, rows, "dup"); edit = ev.ranks_of_positives(Dm, rows, "edit")
        lp, xp, sb, sk, nv, mx = ev.label_metrics(Dm, rows)
        print(f"{name:<24}{mode:<4}| {ev.summarize_ranks(dup):<15}{ev.summarize_ranks(edit):<16}| "
              f"{lp:8.3f}{xp:9.3f}{sb:9.3f} |{sk:6.2f}{100*nv:8.1f}")
