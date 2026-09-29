import numpy as np

from riffle import match


def test_candidate_key_takes_the_top_bits():
    items = np.array([0xFFF00000, 0x00100000, 0x000FFFFF], dtype=np.uint32)
    keys = match.candidate_key(items, 12)
    assert list(keys) == [0xFFF, 0x001, 0x000]


def test_postings_group_equal_keys():
    fps = {
        1: np.array([0x00100000, 0x00200000], dtype=np.uint32),
        2: np.array([0x00100000], dtype=np.uint32),
    }
    cfg = dict(match.DEFAULT_MATCH_CONFIG, align_bits=12)
    postings = match.build_postings(fps, cfg)
    assert sorted(postings[0x001]) == [(1, 0), (2, 0)]


def test_stop_key_cap_drops_ubiquitous_keys():
    cfg = dict(match.DEFAULT_MATCH_CONFIG, align_bits=12, k_cap=3)
    fps = {i: np.array([0x00100000], dtype=np.uint32) for i in range(1, 6)}
    postings = match.build_postings(fps, cfg)
    assert 0x001 not in postings


def test_effective_k_cap_scales_with_the_corpus():
    cfg = dict(match.DEFAULT_MATCH_CONFIG, k_cap=200, k_cap_fraction=0.02)
    # Large corpus: the absolute cap binds.
    assert match.effective_k_cap(cfg, 50_000) == 200
    # Target scale: the fraction binds, so the cap tracks the library.
    assert match.effective_k_cap(cfg, 5_000) == 100
    assert match.effective_k_cap(cfg, 2_000) == 40
    # Tiny corpus: floor of 20, not the fraction's near-zero value. A lower
    # floor would drop any key shared by 3+ fingerprints outright, breaking
    # detection of a realistic N-way duplicate cluster at small scale.
    assert match.effective_k_cap(cfg, 3) == 20


def test_occurrence_cap_limits_positions_within_one_fingerprint():
    cfg = dict(match.DEFAULT_MATCH_CONFIG, align_bits=12, m_cap=2)
    fps = {
        1: np.array([0x00100000] * 10, dtype=np.uint32),
        2: np.array([0x00100000] * 10, dtype=np.uint32),
    }
    postings = match.build_postings(fps, cfg)
    per_content = {}
    for cid, pos in postings[0x001]:
        per_content.setdefault(cid, []).append(pos)
    assert per_content[1] == [0, 1]
    assert per_content[2] == [0, 1]


def test_candidate_pairs_are_ordered_and_unique():
    postings = {0x001: [(5, 0), (2, 0), (9, 3)]}
    pairs = match.candidate_pairs(postings)
    assert pairs == {(2, 5), (2, 9), (5, 9)}


def test_no_self_pairs():
    postings = {0x001: [(3, 0), (3, 7)]}
    assert match.candidate_pairs(postings) == set()


def test_min_shared_discards_chance_collisions():
    # Pair (1, 2) shares 3 keys; (1, 3) shares one, as a chance collision.
    postings = {
        0x001: [(1, 0), (2, 0)],
        0x002: [(1, 1), (2, 1)],
        0x003: [(1, 2), (2, 2), (3, 0)],
    }
    assert match.candidate_pairs(postings) == {(1, 2), (1, 3), (2, 3)}
    assert match.candidate_pairs(postings, min_shared=3) == {(1, 2)}


def test_default_config_finds_duplicates_in_a_library_sized_corpus():
    # Regression: with 12-bit keys every key exceeded k_cap at ~1800 tracks,
    # so no candidates were emitted and only exact duplicates were found.
    rng = np.random.default_rng(0)
    n = 1500
    fps = {i: rng.integers(0, 2**32, 600, dtype=np.uint32)
           for i in range(n)}
    dup = fps[0].copy()
    flip = rng.integers(0, 32, dup.size)  # one flipped bit per item
    fps[n] = dup ^ (np.uint32(1) << flip.astype(np.uint32))
    cfg = match.DEFAULT_MATCH_CONFIG
    pairs = match.candidate_pairs(match.build_postings(fps, cfg),
                                  cfg["min_shared_keys"])
    assert (0, n) in pairs
    assert len(pairs) < 50
