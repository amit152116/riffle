import numpy as np

from audiolib import match

ITEM = 0.1238
CFG = match.DEFAULT_MATCH_CONFIG


def _rand(n, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2 ** 32, size=n, dtype=np.uint64).astype(np.uint32)


def test_identical_is_tier_1():
    fp = _rand(400, 10)
    ev = match.compare(fp, fp, CFG, ITEM)
    assert ev.tier == 1
    assert ev.coverage_a == 1.0
    assert ev.coverage_b == 1.0
    assert ev.mean_bit_error == 0.0


def test_unrelated_is_tier_none():
    ev = match.compare(_rand(400, 11), _rand(400, 12), CFG, ITEM)
    assert ev.tier == match.TIER_NONE
    assert ev.matched_span_items == 0


def test_clip_inside_longer_track_is_tier_2():
    long = _rand(800, 13)
    clip = long[200:400].copy()
    ev = match.compare(long, clip, CFG, ITEM)
    assert ev.tier == 2
    assert ev.coverage_b > 0.9
    assert ev.coverage_a < 0.4


def test_empty_fingerprint_does_not_divide_by_zero():
    # Review Focus 1
    empty = np.array([], dtype=np.uint32)
    ev = match.compare(_rand(100, 14), empty, CFG, ITEM)
    assert ev.coverage_a == 0.0
    assert ev.coverage_b == 0.0
    assert ev.peak_vote_ratio == 0.0
    assert ev.matched_span_seconds == 0.0


def test_two_item_fingerprints_do_not_crash():
    # Review Focus 1: a sub-second file yields a handful of items.
    tiny = np.array([0x12345678, 0x9ABCDEF0], dtype=np.uint32)
    ev = match.compare(tiny, tiny, CFG, ITEM)
    assert ev.matched_span_seconds >= 0.0
    assert 0.0 <= ev.peak_vote_ratio <= 1.0


def test_tier_1_requires_the_absolute_overlap_floor():
    # Review Focus 1: full coverage on both sides, but far too short.
    short = _rand(30, 15)  # ~3.7 seconds at 0.1238 s/item
    ev = match.compare(short, short, CFG, ITEM)
    assert ev.coverage_a == 1.0
    assert ev.tier != 1


def test_compare_is_deterministic():
    a, b = _rand(400, 16), _rand(400, 16)
    results = {match.compare(a, b, CFG, ITEM) for _ in range(10)}
    assert len(results) == 1


def test_tier_none_does_not_collide_with_tier_0():
    # Review finding I8: TIER_NONE was 0, the same value as tier 0
    # ("identical encoded audio stream"). A verified non-match (from
    # group.py's direct chain verification, stored as a real pair.tier
    # value) and a genuine tier-0 identity classification are conceptually
    # unrelated but were numerically indistinguishable in that column.
    assert match.TIER_NONE != 0
