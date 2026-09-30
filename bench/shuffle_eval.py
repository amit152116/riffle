"""Shuffle benchmark on the cached subset (read-only).

Algorithms (each builds 20-track playlists from 100 random seeds)
  prod            riffle's current walk on the current distance: best of the
                  seed track's stored top-20 neighbours after penalties
                  (recent artist +0.5, BPM jump +0.3, compatible key -0.2),
                  random fallback when the neighbours run out
  emb-naive       the same walk and penalties on the new embedding distance
  emb-scaled      same walk, penalties rescaled to the embedding distance's range
  emb-mmr         retrieve top-30 neighbours, re-rank with maximal marginal
                  relevance against the last 5 tracks, then the scaled rules
Judges (independent of the distance each method used)
  coverage        share of the subset appearing in at least one playlist
  fallback        share of steps that had to pick a random track
  coherence       share of labelled tracks matching the seed's folder label
  smooth          mean transition distance on the embedding distance (lower
                  = smoother; favours embedding-based methods, read with care)
  variety         mean pairwise distance inside a playlist (higher = more varied)
  key / bpm       share of key-compatible transitions; mean octave-aware BPM jump
  artist          share of tracks repeating an artist from the last 3

    python bench/shuffle_eval.py
"""
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import evaluate as ev
from riffle import similarity

N_SEEDS, LENGTH, TOP_K, MMR_POOL, MMR_LAMBDA = 100, 20, 20, 30, 0.7


def octave_jump(a, b):
    if a <= 0 or b <= 0:
        return None
    return min(abs(a - b), abs(2 * a - b), abs(a - 2 * b))


class Library:
    def __init__(self, rows, feats):
        self.rows = rows
        self.n = len(rows)
        M = ev.build_matrices(feats)
        self.D_prod = M["prod"]
        self.D_emb = ev.mutual_proximity(M["effnet"])         # judge and new distance
        self.bpm = np.array([float(f["bpm"]) for f in feats])
        self.key = [(str(f["key_name"]), str(f["scale"])) for f in feats]
        self.artist = [r["artist"] for r in rows]
        self.label = [r["label"] for r in rows]
        self.group = [r["group"] for r in rows]
        valid = self.bpm[self.bpm > 0]
        q25, q75 = np.percentile(valid, [25, 75])
        self.bpm_thr = max(float(q75 - q25), 10.0) * 0.25     # production rule
        self.same_group = {}
        for i, g in enumerate(self.group):
            if g is not None:
                self.same_group.setdefault(g, []).append(i)

    def excluded_by(self, i):
        g = self.group[i]
        return set(self.same_group.get(g, [])) if g is not None else set()

    def key_compat(self, a, b):
        return similarity.key_distance(*self.key[a], *self.key[b]) <= 1

    def rule_penalty(self, cur, cand, recent, scale):
        pen = 0.0
        arts = [self.artist[r] for r in recent[-3:]]
        if self.artist[cand] and self.artist[cand] in arts:
            pen += 0.5 * scale
        j = octave_jump(self.bpm[cur], self.bpm[cand])
        if j is not None and j > self.bpm_thr:      # octave-aware: 70 -> 140 is not a jump
            pen += float(os.environ.get('BPM_PEN', '0.3')) * scale
        jump = octave_jump(self.bpm[cur], self.bpm[cand])
        if jump is not None:
            pen += float(os.environ.get('BPM_CONT', '0')) * min(jump / (2 * self.bpm_thr), 1.0) * scale
        if self.key_compat(cur, cand):
            pen -= 0.2 * scale
        return pen


def walk(lib, seed, algo, rng):
    D = lib.D_prod if algo == "prod" else lib.D_emb
    scale = (1.0 if algo.startswith("emb-p") else float(algo.split("-s")[1])
             if algo.startswith("emb-s") and algo != "emb-scaled"
             else {"prod": 1.0, "emb-naive": 1.0, "emb-scaled": 0.2, "emb-mmr": 0.2}[algo])
    playlist, used, fallbacks = [seed], {seed} | lib.excluded_by(seed), 0
    for _ in range(LENGTH - 1):
        cur = playlist[-1]
        order = [j for j in np.argsort(D[cur]) if j != cur]
        pool_size = (int(algo.split("-p")[1]) if algo.startswith("emb-p")
                     else MMR_POOL if algo == "emb-mmr" else TOP_K)
        pool = order[:pool_size]
        best, best_score = None, np.inf
        recent = playlist[-5:]
        for c in pool:
            if c in used:
                continue
            if algo == "emb-mmr":
                sim_cur = 1.0 - D[cur, c]
                sim_recent = max(1.0 - D[r, c] for r in recent)
                score = -(MMR_LAMBDA * sim_cur - (1 - MMR_LAMBDA) * sim_recent)
            else:
                score = D[cur, c]
            score += lib.rule_penalty(cur, c, playlist, scale)
            if score < best_score:
                best, best_score = c, score
        if best is None:
            fallbacks += 1
            left = [j for j in range(lib.n) if j not in used]
            best = int(rng.choice(left))
        playlist.append(best)
        used.add(best); used |= lib.excluded_by(best)
    return playlist, fallbacks


def judge(lib, plays, fallbacks):
    De = lib.D_emb
    seen = Counter(t for p in plays for t in p)
    coh, smooth, variety, keyok, bpm, art = [], [], [], [], [], []
    for p in plays:
        lab = lib.label[p[0]]
        if lab:
            others = [lib.label[t] for t in p[1:] if lib.label[t]]
            if others:
                coh.append(np.mean([l == lab for l in others]))
        smooth += [De[a, b] for a, b in zip(p, p[1:])]
        idx = np.array(p)
        sub = De[np.ix_(idx, idx)]
        variety.append(sub[np.triu_indices(len(p), 1)].mean())
        keyok += [lib.key_compat(a, b) for a, b in zip(p, p[1:])]
        for a, b in zip(p, p[1:]):
            j = octave_jump(lib.bpm[a], lib.bpm[b])
            if j is not None:
                bpm.append(j)
        for i, t in enumerate(p[1:], start=1):
            arts = [lib.artist[x] for x in p[max(0, i - 3):i]]
            if lib.artist[t]:
                art.append(lib.artist[t] in arts)
    steps = len(plays) * (LENGTH - 1)
    return (len(seen) / lib.n, fallbacks / steps, float(np.mean(coh)), float(np.mean(smooth)),
            float(np.mean(variety)), float(np.mean(keyok)), float(np.mean(bpm)),
            float(np.mean(art)) if art else float("nan"))


def main():
    rows, feats = ev.load()
    lib = Library(rows, feats)
    print(f"tracks: {lib.n}; seeds: {N_SEEDS}; playlist length: {LENGTH}")
    rng = np.random.default_rng(3)
    seeds = rng.choice(lib.n, N_SEEDS, replace=False)
    head = (f"{'algorithm':<12}|{'coverage':>9}{'fallback':>9}{'coherence':>10}|{'smooth':>8}"
            f"{'variety':>8}|{'keyOK':>7}{'bpmJump':>8}{'artRep':>7}")
    print(head + "\n" + "-" * len(head))
    algos = sys.argv[1:] or ["prod", "emb-naive", "emb-scaled", "emb-mmr"]
    for algo in algos:
        plays, fb = [], 0
        r2 = np.random.default_rng(11)
        for s in seeds:
            p, f = walk(lib, int(s), algo, r2)
            plays.append(p); fb += f
        c, f, co, sm, va, ko, bj, ar = judge(lib, plays, fb)
        print(f"{algo:<12}|{c:9.2f}{f:9.2f}{co:10.3f}|{sm:8.3f}{va:8.3f}|{ko:7.2f}{bj:8.1f}{ar:7.2f}")


if __name__ == "__main__":
    main()
