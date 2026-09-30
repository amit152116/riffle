"""offset_histogram/best_alignment are the hot path of `match`.

Profiled on a 1,800-track library, 70% of compare() time was the Python
double loop in offset_histogram and another 24% the Python scan in
best_alignment. The reference implementations below are the original loops;
the vectorised versions must return exactly the same values.
"""
import time
from collections import defaultdict

import numpy as np
import pytest

from riffle import match


def _reference_offset_histogram(fp_a, fp_b, align_bits):
    n_a, n_b = len(fp_a), len(fp_b)
    hist = np.zeros(n_a + n_b + 1, dtype=np.int64)
    if n_a == 0 or n_b == 0:
        return hist, n_b
    keys_a = match.candidate_key(fp_a, align_bits)
    keys_b = match.candidate_key(fp_b, align_bits)
    by_key_b = defaultdict(list)
    for pos, key in enumerate(keys_b.tolist()):
        by_key_b[key].append(pos)
    for pos_a, key in enumerate(keys_a.tolist()):
        for pos_b in by_key_b.get(key, ()):
            hist[pos_a - pos_b + n_b] += 1
    return hist, n_b


def _reference_best_alignment(hist, shift):
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


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("bits", [4, 8, 12, 24])
def test_offset_histogram_matches_the_reference(seed, bits):
    rng = np.random.default_rng(seed)
    fp_a = rng.integers(0, 2**32, rng.integers(1, 300), dtype=np.uint32)
    fp_b = rng.integers(0, 2**32, rng.integers(1, 300), dtype=np.uint32)
    got, shift = match.offset_histogram(fp_a, fp_b, bits)
    want, want_shift = _reference_offset_histogram(fp_a, fp_b, bits)
    assert shift == want_shift
    assert np.array_equal(got, want)


def test_offset_histogram_handles_empty_input():
    empty = np.array([], dtype=np.uint32)
    other = np.array([1, 2, 3], dtype=np.uint32)
    for a, b in ((empty, other), (other, empty), (empty, empty)):
        got, shift = match.offset_histogram(a, b, 12)
        want, want_shift = _reference_offset_histogram(a, b, 12)
        assert shift == want_shift
        assert np.array_equal(got, want)


@pytest.mark.parametrize("seed", range(20))
def test_best_alignment_matches_the_reference(seed):
    rng = np.random.default_rng(seed)
    # Small counts force plateaus and ties, which the lowest-index rule
    # must resolve identically.
    hist = rng.integers(0, 4, rng.integers(1, 60)).astype(np.int64)
    shift = int(rng.integers(0, 30))
    assert match.best_alignment(hist, shift) == \
        _reference_best_alignment(hist, shift)


def test_best_alignment_finds_nothing_without_a_repeated_bin():
    hist = np.array([0, 1, 0, 1, 0], dtype=np.int64)
    assert match.best_alignment(hist, 2) == (0, 0, 0)


def test_hot_path_is_fast_on_library_sized_fingerprints():
    # Real fingerprints repeat values heavily (sustained notes, silence), so
    # one pair yields tens of thousands of key hits. Drawing 2,500 items
    # from a pool of 40 values reproduces that: ~150,000 hits per pair. The
    # Python loops took ~30 ms per pair; vectorised it is ~1 ms. The
    # threshold sits well between the two to stay stable on a slow machine.
    rng = np.random.default_rng(0)
    pool = rng.integers(0, 2**32, 40, dtype=np.uint32)
    fps = [pool[rng.integers(0, 40, 2500)] for _ in range(2)]
    start = time.perf_counter()
    for _ in range(30):
        hist, shift = match.offset_histogram(fps[0], fps[1], 24)
        match.best_alignment(hist, shift)
    assert time.perf_counter() - start < 0.4
