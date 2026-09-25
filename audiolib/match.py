"""Candidate generation and pairwise alignment.

Constants follow Chromaprint's own FingerprintMatcher (src/fingerprint_matcher
.cpp): the alignment key is the top 12 bits, the match threshold is 10.0, the
bit-error series is Gaussian-smoothed at sigma 8.0, and segments are cut at
gradient peaks above 0.15 and merged when their scores differ by less than 0.7.

ACOUSTID_MAX_BIT_ERROR and ACOUSTID_MAX_ALIGN_OFFSET appear in that file but
are NOT used by Match(). They belong to AcoustID's server query path. Imposing
a +/-120 alignment window here would discard exactly the long-offset matches
this tool exists to find.

Upstream adds rand() jitter to every Hamming distance. That is deliberately
not reproduced: reports must be reproducible. Ties break by index order.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

DEFAULT_MATCH_CONFIG = {
    "align_bits": 12,
    "k_cap": 200,
    "k_cap_fraction": 0.02,
    "m_cap": 8,
    "match_threshold": 10.0,
    "sigma": 8.0,
    "gradient_peak": 0.15,
    "merge_delta": 0.7,
    "min_peak_vote_ratio": 0.05,
    "tier1_min_coverage": 0.85,
    "tier1_max_bit_error": 6.0,
    "tier1_min_overlap_seconds": 20.0,
    "tier2_min_overlap_seconds": 15.0,
}


def effective_k_cap(config: dict, n_fingerprints: int) -> int:
    """Stop-key cap, bounded both absolutely and as a share of the corpus.

    The absolute term bounds cost, since O(n^2) pair emission depends on the
    count. The fractional term bounds informativeness: a key present in
    several percent of the library distinguishes nothing, and a purely
    absolute cap grows more permissive as the library shrinks. The floor of
    two is the smallest posting list that can yield a pair at all.
    """
    fractional = int(config["k_cap_fraction"] * n_fingerprints)
    return max(2, min(config["k_cap"], fractional))


def candidate_key(items: np.ndarray, align_bits: int) -> np.ndarray:
    """Upstream's ALIGN_STRIP: keep the top `align_bits` bits."""
    return (np.asarray(items, dtype=np.uint32) >> (32 - align_bits)).astype(
        np.uint32
    )


def build_postings(fps: dict[int, np.ndarray], config: dict) -> dict:
    """Key -> [(content_id, position)], with both caps applied.

    `m_cap` bounds how many positions of one key are carried per fingerprint,
    which stops a repeated key producing a cartesian blow-up inside a pair.
    `k_cap` drops keys that appear in too many fingerprints, which is what
    silence and fades produce; pair emission is O(n^2) in a posting list.
    """
    align_bits = config["align_bits"]
    m_cap = config["m_cap"]
    k_cap = effective_k_cap(config, len(fps))

    postings: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for content_id in sorted(fps):
        keys = candidate_key(fps[content_id], align_bits)
        counts: dict[int, int] = defaultdict(int)
        for pos, key in enumerate(keys.tolist()):
            if counts[key] >= m_cap:
                continue
            counts[key] += 1
            postings[key].append((content_id, pos))

    return {
        key: entries
        for key, entries in postings.items()
        if len({cid for cid, _ in entries}) <= k_cap
    }


def candidate_pairs(postings: dict) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for entries in postings.values():
        ids = sorted({cid for cid, _ in entries})
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                pairs.add((a, b))
    return pairs


def offset_histogram(fp_a: np.ndarray, fp_b: np.ndarray,
                     align_bits: int) -> tuple[np.ndarray, int]:
    """Histogram of position deltas over keys the two fingerprints share.

    Bin index is `pos_a - pos_b + len(fp_b)`, so the shift to subtract to get
    a signed offset is `len(fp_b)`.
    """
    n_a, n_b = len(fp_a), len(fp_b)
    hist = np.zeros(n_a + n_b + 1, dtype=np.int64)
    if n_a == 0 or n_b == 0:
        return hist, n_b

    keys_a = candidate_key(fp_a, align_bits)
    keys_b = candidate_key(fp_b, align_bits)

    by_key_b: dict[int, list[int]] = defaultdict(list)
    for pos, key in enumerate(keys_b.tolist()):
        by_key_b[key].append(pos)

    for pos_a, key in enumerate(keys_a.tolist()):
        for pos_b in by_key_b.get(key, ()):
            hist[pos_a - pos_b + n_b] += 1
    return hist, n_b


def best_alignment(hist: np.ndarray, shift: int) -> tuple[int, int, int]:
    """Highest local-maximum bin with more than one vote.

    Ties are resolved by the lowest index, scanning low to high, so the result
    does not depend on iteration order or on any random jitter.
    """
    total = int(hist.sum())
    best_index = -1
    best_count = 0
    for i in range(len(hist)):
        count = int(hist[i])
        if count <= 1:
            continue
        left_ok = hist[i - 1] <= count if i > 0 else True
        right_ok = hist[i + 1] <= count if i < len(hist) - 1 else True
        if left_ok and right_ok and count > best_count:
            best_count = count
            best_index = i
    if best_index < 0:
        return 0, 0, 0
    return best_index - shift, best_count, total
