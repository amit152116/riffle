import json
import time

import numpy as np
import pytest

from audiolib import enrich, fingerprint, store


def _content_with_fp(conn, tmp_path, n_items=2000, cid=1, file_seconds=5.0):
    """A content row with a synthetic canonical fingerprint.

    Also backed by a real audio file with a `track` row: every real
    audio_content always comes from a scanned file, and lookup_fingerprint's
    fallback to a dedicated acoustid_lookup artifact needs a real path to run
    fpcalc against whenever the (here, fake) canonical array is longer than
    the 120s lookup window. The file's actual audio has no relationship to
    the synthetic fingerprint bytes; most callers only exercise caching, rate
    limiting and duration-passing, not fingerprint content, so `file_seconds`
    defaults to just enough for fpcalc to succeed (empty below ~3s). A test
    that checks the truncated item count against a ~120s expectation must
    pass a `file_seconds` long enough for that expectation to mean anything.
    """
    from tests.fixtures import make_tone

    conn.execute("INSERT INTO audio_content (id, audio_hash, hash_method, "
                 " duration) VALUES (?,?, 'streamhash', 300.0)", (cid, f"h{cid}"))
    rng = np.random.default_rng(cid)
    raw = rng.integers(0, 2 ** 32, size=n_items, dtype=np.uint64).astype(np.uint32)
    conn.execute(
        "INSERT INTO fingerprint (audio_content_id, analyzer, "
        " analyzer_version, config_hash, purpose, algorithm, fp_raw, fp_length) "
        "VALUES (?, 'chromaprint', 'v', 'c', 'canonical', 2, ?, ?)",
        (cid, store.pack_fingerprint(raw), len(raw)))

    p = make_tone(tmp_path / f"content{cid}.flac", seconds=file_seconds)
    st = p.stat()
    conn.execute(
        "INSERT INTO track (id, path, size, mtime, audio_content_id, present) "
        "VALUES (?,?,?,?,?,1)", (cid, str(p), st.st_size, st.st_mtime, cid))
    return raw


def test_rate_limiter_spaces_calls():
    slept = []
    rl = enrich.RateLimiter(3.0, sleeper=slept.append, clock=iter([0.0, 0.0, 0.0]).__next__)
    rl.wait()
    rl.wait()
    assert slept and slept[0] > 0


def test_lookup_key_covers_the_whole_request():
    a = enrich.lookup_key(2, "AQAA", 300.0, "recordings")
    assert a == enrich.lookup_key(2, "AQAA", 300.0, "recordings")
    assert a != enrich.lookup_key(2, "AQAA", 301.0, "recordings")
    assert a != enrich.lookup_key(2, "AQAB", 300.0, "recordings")
    assert a != enrich.lookup_key(2, "AQAA", 300.0, "recordings+releasegroups")


def test_lookup_fingerprint_truncates_to_120_seconds(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    # The backing file must itself run well past 120s, or the real fpcalc
    # fallback (needed because slicing isn't prefix-compatible -- see
    # test_prefix_compatibility_check) has nothing to truncate.
    _content_with_fp(conn, tmp_path, n_items=2000, file_seconds=150.0)
    encoded, n = enrich.lookup_fingerprint(conn, 1, fingerprint.DEFAULT_CONFIG)
    expected = int(round(120.0 / fingerprint.item_duration_seconds()))
    assert abs(n - expected) <= 10
    assert isinstance(encoded, str) and encoded


def test_lookup_fingerprint_keeps_short_fingerprints_whole(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _content_with_fp(conn, tmp_path, n_items=100)
    _, n = enrich.lookup_fingerprint(conn, 1, fingerprint.DEFAULT_CONFIG)
    assert n == 100


def test_enrich_caches_and_does_not_refetch(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _content_with_fp(conn, tmp_path)
    calls = []

    def fake_client(apikey, fp, duration, meta):
        calls.append((fp, duration))
        return {"status": "ok", "results": []}

    enrich.enrich(conn, "key", client=fake_client, sleeper=lambda _: None)
    enrich.enrich(conn, "key", client=fake_client, sleeper=lambda _: None)
    assert len(calls) == 1
    assert conn.execute(
        "SELECT count(*) c FROM acoustid_cache").fetchone()["c"] == 1


def test_enrich_sends_the_full_duration_not_the_truncated_one(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _content_with_fp(conn, tmp_path)
    seen = {}

    def fake_client(apikey, fp, duration, meta):
        seen["duration"] = duration
        return {"status": "ok", "results": []}

    enrich.enrich(conn, "key", client=fake_client, sleeper=lambda _: None)
    assert seen["duration"] == 300.0


def test_enrich_failure_does_not_raise(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _content_with_fp(conn, tmp_path)

    def broken_client(*_args, **_kwargs):
        raise RuntimeError("network down")

    result = enrich.enrich(conn, "key", client=broken_client,
                           sleeper=lambda _: None)
    assert result["failed"] == 1
    assert result["looked_up"] == 0


def test_prefix_compatibility_check(tmp_path):
    # Documented assumption under test: slicing the canonical array and
    # re-encoding should equal a plain default fpcalc run on the same file.
    from tests.fixtures import make_tone

    p = make_tone(tmp_path / "a.flac", seconds=200.0)
    full = fingerprint.fingerprint_file(p).raw
    n = int(round(120.0 / fingerprint.item_duration_seconds()))
    derived = enrich.encode(full[:n], 2)
    default = fingerprint.fingerprint_file(
        p, dict(fingerprint.DEFAULT_CONFIG, length=120))
    native = enrich.encode(default.raw, 2)
    if derived != native:
        pytest.skip(
            "fingerprints are not prefix-compatible; "
            "enrich must store a dedicated acoustid_lookup artifact"
        )


def test_lookup_fingerprint_falls_back_to_a_dedicated_artifact(tmp_path):
    # Confirmed on this build: slicing the canonical array and re-encoding is
    # NOT prefix-compatible with a plain `fpcalc -length 120` run (see
    # test_prefix_compatibility_check, which skips here). lookup_fingerprint
    # must therefore create and reuse a dedicated purpose='acoustid_lookup'
    # artifact rather than trust the derived slice, without ever touching the
    # canonical full-track fingerprint.
    from audiolib import scan
    from tests.fixtures import make_tone

    lib = tmp_path / "lib"
    p = make_tone(lib / "a.flac", seconds=200.0)
    conn = store.connect(tmp_path / "db.sqlite")
    scan.scan(conn, [lib])
    fingerprint.fingerprint_pending(conn)

    content_id = conn.execute(
        "SELECT audio_content_id FROM track LIMIT 1").fetchone()[0]
    canonical_before = conn.execute(
        "SELECT fp_raw FROM fingerprint WHERE purpose = 'canonical'"
    ).fetchone()["fp_raw"]

    native = enrich.encode(
        fingerprint.fingerprint_file(
            p, dict(fingerprint.DEFAULT_CONFIG, length=120)).raw,
        2,
    )

    encoded, _ = enrich.lookup_fingerprint(conn, content_id,
                                           fingerprint.DEFAULT_CONFIG)
    assert encoded == native

    lookup_rows = conn.execute(
        "SELECT count(*) c FROM fingerprint WHERE purpose = 'acoustid_lookup'"
    ).fetchone()["c"]
    assert lookup_rows == 1

    # The canonical row is untouched.
    canonical_after = conn.execute(
        "SELECT fp_raw FROM fingerprint WHERE purpose = 'canonical'"
    ).fetchone()["fp_raw"]
    assert canonical_after == canonical_before

    # Reused, not recreated, on a second call.
    enrich.lookup_fingerprint(conn, content_id, fingerprint.DEFAULT_CONFIG)
    lookup_rows_again = conn.execute(
        "SELECT count(*) c FROM fingerprint WHERE purpose = 'acoustid_lookup'"
    ).fetchone()["c"]
    assert lookup_rows_again == 1
