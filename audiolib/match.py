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
