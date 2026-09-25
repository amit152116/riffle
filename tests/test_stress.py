import resource
import time

from audiolib import fingerprint, match, matchrun, scan, store
from tests.fixtures import concat, make_silence, make_tone


def test_common_keys_do_not_explode_the_candidate_set(tmp_path):
    """The caps exist for exactly this input: many files sharing silence."""
    lib = tmp_path / "lib"
    silence = make_silence(tmp_path / "sil.flac", seconds=20.0)
    for i in range(12):
        body = make_tone(tmp_path / f"body{i}.flac", seconds=20.0,
                         freq=300 + 40 * i)
        concat(lib / f"track{i}.flac", silence, body, silence)

    conn = store.connect(tmp_path / "db.sqlite")
    scan.scan(conn, [lib])
    fingerprint.fingerprint_pending(conn)

    fps = matchrun.load_fingerprints(conn)
    # Pin the caps rather than inheriting them: with only 12 fixtures the
    # fractional term would floor to 2 and the test would be exercising the
    # floor instead of the caps it exists to check.
    cfg = dict(match.DEFAULT_MATCH_CONFIG, k_cap=4, k_cap_fraction=1.0)
    capped = match.build_postings(fps, cfg)
    uncapped = match.build_postings(
        fps, dict(cfg, k_cap=10 ** 9, k_cap_fraction=10 ** 9, m_cap=10 ** 9))

    assert sum(len(v) for v in capped.values()) < \
           sum(len(v) for v in uncapped.values())
    assert len(match.candidate_pairs(capped)) <= \
           len(match.candidate_pairs(uncapped))


def test_match_run_stays_within_time_and_memory(tmp_path):
    lib = tmp_path / "lib"
    for i in range(12):
        make_tone(lib / f"t{i}.flac", seconds=30.0, freq=250 + 50 * i)

    conn = store.connect(tmp_path / "db.sqlite")
    scan.scan(conn, [lib])
    fingerprint.fingerprint_pending(conn)

    start = time.monotonic()
    matchrun.run_match(conn)
    elapsed = time.monotonic() - start
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    assert elapsed < 60.0
    assert peak_mb < 1024
