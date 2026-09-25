import numpy as np

from riffle import match


def _fp(values):
    return np.array([v << 20 for v in values], dtype=np.uint32)


def test_identical_fingerprints_align_at_zero():
    fp = _fp([1, 2, 3, 4, 5, 6, 7, 8])
    hist, shift = match.offset_histogram(fp, fp, 12)
    offset, peak, total = match.best_alignment(hist, shift)
    assert offset == 0
    assert peak >= 8


def test_shifted_fingerprint_reports_the_shift():
    base = _fp([10, 11, 12, 13, 14, 15, 16, 17])
    shifted = _fp([90, 91, 92, 10, 11, 12, 13, 14, 15, 16, 17])
    hist, shift = match.offset_histogram(base, shifted, 12)
    offset, peak, total = match.best_alignment(hist, shift)
    # base position i corresponds to shifted position i + 3
    assert offset == -3
    assert peak >= 8


def test_unrelated_fingerprints_produce_no_strong_peak():
    a = _fp(list(range(100, 140)))
    b = _fp(list(range(500, 540)))
    hist, shift = match.offset_histogram(a, b, 12)
    offset, peak, total = match.best_alignment(hist, shift)
    assert peak == 0
    assert total == 0


def test_empty_fingerprint_is_handled():
    a = _fp([1, 2, 3])
    b = np.array([], dtype=np.uint32)
    hist, shift = match.offset_histogram(a, b, 12)
    offset, peak, total = match.best_alignment(hist, shift)
    assert (offset, peak, total) == (0, 0, 0)


def test_peak_selection_is_deterministic_under_ties():
    # Two equal-height peaks: the lower index must win, every time.
    hist = np.array([0, 5, 0, 5, 0], dtype=np.int64)
    results = {match.best_alignment(hist, 2) for _ in range(20)}
    assert len(results) == 1
    offset, peak, _ = results.pop()
    assert (offset, peak) == (-1, 5)
