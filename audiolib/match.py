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
from dataclasses import dataclass

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


_POPCOUNT = np.array(
    [bin(i).count("1") for i in range(256)], dtype=np.uint8
)


@dataclass(frozen=True)
class Segment:
    pos_a: int
    pos_b: int
    length: int
    score: float


def hamming_series(fp_a: np.ndarray, fp_b: np.ndarray,
                   offset: int) -> np.ndarray:
    """Per-item differing-bit counts over the region the offset overlaps.

    `offset` is `pos_a - pos_b`: item `i` of the overlap is `fp_a[start_a + i]`
    against `fp_b[start_b + i]`.
    """
    start_a = offset if offset > 0 else 0
    start_b = -offset if offset < 0 else 0
    size = min(len(fp_a) - start_a, len(fp_b) - start_b)
    if size <= 0:
        return np.zeros(0, dtype=np.int32)
    xor = (fp_a[start_a:start_a + size] ^ fp_b[start_b:start_b + size])
    as_bytes = np.ascontiguousarray(xor, dtype="<u4").view(np.uint8)
    return _POPCOUNT[as_bytes].reshape(-1, 4).sum(axis=1).astype(np.int32)


def gaussian_smooth(series: np.ndarray, sigma: float) -> np.ndarray:
    """Truncated Gaussian convolution.

    Upstream approximates the Gaussian with three box-filter passes. A direct
    kernel is used here: the difference is immaterial at sigma 8, and every
    threshold downstream is calibrated rather than inherited verbatim.
    """
    if len(series) == 0:
        return series.astype(float)
    radius = max(1, int(round(4 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    padded = np.pad(series.astype(float), radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def segments(fp_a: np.ndarray, fp_b: np.ndarray, offset: int,
             config: dict) -> list[Segment]:
    """Cut the overlap into segments and keep the ones that match.

    No random jitter is added to the bit counts. Gradient-peak ties resolve by
    index order, so repeated runs return identical segments.
    """
    raw = hamming_series(fp_a, fp_b, offset)
    size = len(raw)
    if size == 0:
        return []

    start_a = offset if offset > 0 else 0
    start_b = -offset if offset < 0 else 0

    smoothed = gaussian_smooth(raw, config["sigma"])
    gradient = np.abs(np.gradient(smoothed))

    boundaries: list[int] = []
    threshold = config["gradient_peak"]
    for i in range(1, size - 1):
        g = gradient[i]
        if g > threshold and g >= gradient[i - 1] and g >= gradient[i + 1]:
            if not boundaries or boundaries[-1] + 1 < i:
                boundaries.append(i)
    boundaries.append(size)

    kept: list[Segment] = []
    begin = 0
    for end in boundaries:
        length = end - begin
        if length <= 0:
            continue
        score = float(raw[begin:end].mean())
        if score < config["match_threshold"]:
            if kept and abs(kept[-1].score - score) < config["merge_delta"] \
                    and kept[-1].pos_a + kept[-1].length == start_a + begin:
                prev = kept.pop()
                total = prev.length + length
                merged_score = (
                    prev.score * prev.length + score * length
                ) / total
                kept.append(Segment(prev.pos_a, prev.pos_b, total, merged_score))
            else:
                kept.append(
                    Segment(start_a + begin, start_b + begin, length, score)
                )
        begin = end
    return kept


@dataclass(frozen=True)
class PairEvidence:
    best_offset: int
    peak_votes: int
    peak_vote_ratio: float
    matched_span_items: int
    matched_span_seconds: float
    coverage_a: float
    coverage_b: float
    mean_bit_error: float
    segment_count: int
    tier: int


TIER_NONE = 0


def classify(fields: dict, config: dict) -> int:
    """Tier from evidence. Tier 0 is assigned by identity, never here."""
    if fields["matched_span_items"] == 0:
        return TIER_NONE
    if fields["peak_vote_ratio"] < config["min_peak_vote_ratio"]:
        return TIER_NONE

    cov_min = min(fields["coverage_a"], fields["coverage_b"])
    cov_max = max(fields["coverage_a"], fields["coverage_b"])
    seconds = fields["matched_span_seconds"]

    if (cov_min >= config["tier1_min_coverage"]
            and fields["mean_bit_error"] <= config["tier1_max_bit_error"]
            and seconds >= config["tier1_min_overlap_seconds"]):
        return 1

    if cov_max >= config["tier1_min_coverage"] \
            and seconds >= config["tier2_min_overlap_seconds"]:
        return 2
    if cov_min > 0.0 and seconds >= config["tier2_min_overlap_seconds"]:
        return 2
    return TIER_NONE


def compare(fp_a: np.ndarray, fp_b: np.ndarray, config: dict,
            item_seconds: float) -> PairEvidence:
    n_a, n_b = len(fp_a), len(fp_b)
    hist, shift = offset_histogram(fp_a, fp_b, config["align_bits"])
    offset, peak_votes, total_votes = best_alignment(hist, shift)

    segs = segments(fp_a, fp_b, offset, config) if peak_votes else []
    span = sum(s.length for s in segs)
    if span:
        mean_bit_error = sum(s.score * s.length for s in segs) / span
    else:
        mean_bit_error = 0.0

    fields = {
        "best_offset": offset,
        "peak_votes": peak_votes,
        "peak_vote_ratio": (peak_votes / total_votes) if total_votes else 0.0,
        "matched_span_items": span,
        "matched_span_seconds": span * item_seconds,
        "coverage_a": (span / n_a) if n_a else 0.0,
        "coverage_b": (span / n_b) if n_b else 0.0,
        "mean_bit_error": mean_bit_error,
        "segment_count": len(segs),
    }
    fields["tier"] = classify(fields, config)
    return PairEvidence(**fields)
