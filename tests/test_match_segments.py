import numpy as np

from audiolib import match


def test_hamming_series_counts_differing_bits():
    a = np.array([0b0000, 0b1111], dtype=np.uint32)
    b = np.array([0b0001, 0b1111], dtype=np.uint32)
    series = match.hamming_series(a, b, 0)
    assert list(series) == [1, 0]


def test_hamming_series_respects_offset():
    a = np.array([9, 1, 2, 3], dtype=np.uint32)
    b = np.array([1, 2, 3], dtype=np.uint32)
    # a position 1 lines up with b position 0, so offset is +1
    assert list(match.hamming_series(a, b, 1)) == [0, 0, 0]


def test_no_overlap_gives_empty_series():
    a = np.array([1, 2], dtype=np.uint32)
    b = np.array([1, 2], dtype=np.uint32)
    assert len(match.hamming_series(a, b, 99)) == 0


def test_gaussian_smooth_preserves_length_and_flattens_noise():
    series = np.zeros(200)
    series[100] = 32.0
    out = match.gaussian_smooth(series, 8.0)
    assert len(out) == 200
    assert out[100] < 32.0
    assert out[90] > 0.0


def test_identical_audio_yields_one_full_length_segment():
    rng = np.random.default_rng(0)
    fp = rng.integers(0, 2 ** 32, size=400, dtype=np.uint64).astype(np.uint32)
    segs = match.segments(fp, fp, 0, match.DEFAULT_MATCH_CONFIG)
    assert len(segs) == 1
    assert segs[0].length == 400
    assert segs[0].score == 0.0


def test_unrelated_audio_yields_no_segments():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 2 ** 32, size=400, dtype=np.uint64).astype(np.uint32)
    b = rng.integers(0, 2 ** 32, size=400, dtype=np.uint64).astype(np.uint32)
    segs = match.segments(a, b, 0, match.DEFAULT_MATCH_CONFIG)
    assert sum(s.length for s in segs) == 0


def test_partial_match_yields_a_partial_segment():
    rng = np.random.default_rng(2)
    shared = rng.integers(0, 2 ** 32, size=200, dtype=np.uint64).astype(np.uint32)
    noise_a = rng.integers(0, 2 ** 32, size=200, dtype=np.uint64).astype(np.uint32)
    noise_b = rng.integers(0, 2 ** 32, size=200, dtype=np.uint64).astype(np.uint32)
    a = np.concatenate([shared, noise_a])
    b = np.concatenate([shared, noise_b])
    segs = match.segments(a, b, 0, match.DEFAULT_MATCH_CONFIG)
    matched = sum(s.length for s in segs)
    assert 100 < matched < 320


def test_segments_are_deterministic():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 2 ** 32, size=300, dtype=np.uint64).astype(np.uint32)
    b = a.copy()
    b[150:] = rng.integers(0, 2 ** 32, size=150, dtype=np.uint64).astype(np.uint32)
    runs = [match.segments(a, b, 0, match.DEFAULT_MATCH_CONFIG) for _ in range(5)]
    assert all(r == runs[0] for r in runs)
