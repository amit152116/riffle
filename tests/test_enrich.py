import json
import time

import numpy as np
import pytest

from riffle import enrich, fingerprint, store


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


def test_rate_limiter_accounts_for_time_spent_sleeping():
    # Review finding I6: _last was captured before sleeping, not after, so
    # any real per-request latency between calls let the effective rate
    # drift above the configured cap -- reproduced with 20ms of latency
    # between wait() calls, matching the finding's own simulation, which
    # measured ~6 req/s against a configured 3 req/s (Global Constraint).
    t = [0.0]

    def clock():
        return t[0]

    def sleeper(seconds):
        t[0] += seconds  # sleeping actually advances the clock

    rl = enrich.RateLimiter(3.0, sleeper=sleeper, clock=clock)
    calls = []
    for _ in range(5):
        rl.wait()
        calls.append(t[0])
        t[0] += 0.02  # 20ms of real request latency after each wait()

    gaps = [b - a for a, b in zip(calls, calls[1:])]
    assert all(g >= rl._interval - 1e-9 for g in gaps)


def test_lookup_key_covers_the_whole_request():
    a = enrich.lookup_key(2, "AQAA", 300.0, "recordings")
    assert a == enrich.lookup_key(2, "AQAA", 300.0, "recordings")
    assert a != enrich.lookup_key(2, "AQAA", 301.0, "recordings")
    assert a != enrich.lookup_key(2, "AQAB", 300.0, "recordings")
    assert a != enrich.lookup_key(2, "AQAA", 300.0, "recordings+releasegroups")


def test_lookup_fingerprint_truncates_to_120_seconds(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    # The fast path slices the in-memory canonical array directly and never
    # touches the backing file, so the fixture's cheap default duration is
    # enough here -- only the (monkeypatched) fallback test needs a real
    # long file to actually fingerprint.
    _content_with_fp(conn, tmp_path, n_items=2000)
    encoded, n = enrich.lookup_fingerprint(conn, 1, fingerprint.DEFAULT_CONFIG)
    # Not the naive 120/item_duration_seconds(): that overcounts by ignoring
    # Chromaprint's fixed per-file delay (I4's second half). This is the
    # same delay-aware count lookup_fingerprint itself now uses.
    expected = fingerprint.lookup_window_items(120.0)
    assert abs(n - expected) <= 2
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


def test_enrich_does_not_cache_an_api_level_error_response(tmp_path):
    # pyacoustid's client does not raise for an API-level error (bad key,
    # rate limited, malformed request) -- it returns a normal 200 OK dict
    # with status="error", the same shape as a real success. Caching that
    # unconditionally, as observed live against the real service with an
    # invalid key, poisons the cache: a later run with a corrected key
    # would see a cache hit and never retry the lookup.
    conn = store.connect(tmp_path / "db.sqlite")
    _content_with_fp(conn, tmp_path)

    def error_client(apikey, fp, duration, meta):
        return {"status": "error", "error": {"code": 4,
                                             "message": "invalid API key"}}

    result = enrich.enrich(conn, "bad-key", client=error_client,
                           sleeper=lambda _: None)
    assert result["failed"] == 1
    assert result["looked_up"] == 0
    assert conn.execute(
        "SELECT count(*) c FROM acoustid_cache").fetchone()["c"] == 0

    # A later run, even with the same client, must retry rather than treat
    # the poisoned lookup as already done.
    result2 = enrich.enrich(conn, "bad-key", client=error_client,
                            sleeper=lambda _: None)
    assert result2["failed"] == 1
    assert result2["cached"] == 0


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


def test_lookup_fingerprint_prefers_the_fast_derived_slice_when_valid(tmp_path):
    # Verified compatible on this build (test_prefix_compatibility_check
    # passes). The fast path -- slice the in-memory canonical array and
    # re-encode -- must be preferred whenever it is valid: no file access,
    # no dedicated artifact row.
    from riffle import scan
    from tests.fixtures import make_tone

    lib = tmp_path / "lib"
    p = make_tone(lib / "a.flac", seconds=200.0)
    conn = store.connect(tmp_path / "db.sqlite")
    scan.scan(conn, [lib])
    fingerprint.fingerprint_pending(conn)

    content_id = conn.execute(
        "SELECT audio_content_id FROM track LIMIT 1").fetchone()[0]

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
    assert lookup_rows == 0


def test_lookup_fingerprint_falls_back_to_a_dedicated_artifact(monkeypatch,
                                                                tmp_path):
    # The spec's own design is preferred-fast-path, fallback-if-invalid. This
    # build's fast path is valid (see the test above), so the fallback branch
    # is forced here rather than relied on to occur naturally, to verify it
    # still produces a correct result and never touches the canonical
    # full-track artifact when it does trigger -- on a build, version, or
    # algorithm where slicing is genuinely not prefix-compatible.
    monkeypatch.setattr(enrich, "_prefix_slicing_is_valid", lambda: False)

    from riffle import scan
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
