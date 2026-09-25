import numpy as np
import pytest

from audiolib import fingerprint, scan, store
from tests.fixtures import make_tone, transcode


def test_fpcalc_version_is_reported():
    v = fingerprint.fpcalc_version()
    assert v and any(ch.isdigit() for ch in v)


def test_fingerprint_shape_matches_expectation(tmp_path):
    # Guards the parser against a change in fpcalc's output format.
    p = make_tone(tmp_path / "a.flac", seconds=20.0)
    res = fingerprint.fingerprint_file(p)
    assert isinstance(res.raw, np.ndarray)
    assert res.raw.dtype == np.uint32
    assert len(res.raw) > 50
    assert 19.0 < res.duration < 21.0


def test_full_length_is_used_not_the_120s_default(tmp_path):
    long = make_tone(tmp_path / "long.flac", seconds=150.0)
    res = fingerprint.fingerprint_file(long)
    # At ~8 items/sec, 120s would cap near 970 items. Full length must exceed it.
    assert len(res.raw) > 1100


def test_item_duration_is_plausible():
    d = fingerprint.item_duration_seconds()
    assert 0.05 < d < 0.5


def test_transcode_fingerprints_similarly(tmp_path):
    a = make_tone(tmp_path / "a.flac", seconds=20.0)
    b = transcode(a, tmp_path / "a.mp3", codec="libmp3lame", bitrate="128k")
    fa = fingerprint.fingerprint_file(a).raw
    fb = fingerprint.fingerprint_file(b).raw
    n = min(len(fa), len(fb))
    # Same audio, different encoding: most top-12-bit keys should survive.
    same = np.sum((fa[:n] >> 20) == (fb[:n] >> 20))
    assert same / n > 0.5


def test_config_hash_is_stable_and_order_independent():
    a = fingerprint.config_hash({"length": 0, "algorithm": 2})
    b = fingerprint.config_hash({"algorithm": 2, "length": 0})
    assert a == b
    assert a != fingerprint.config_hash({"length": 120, "algorithm": 2})


def test_zero_duration_input_raises(tmp_path):
    # Review Focus 5: a stream with no usable duration must fail loudly,
    # not produce a fingerprint with duration 0 that later divides by zero.
    p = tmp_path / "empty.flac"
    p.write_bytes(b"")
    with pytest.raises(fingerprint.FingerprintError):
        fingerprint.fingerprint_file(p)


def test_fingerprint_pending_writes_one_artifact_per_content(tmp_path):
    lib = tmp_path / "lib"
    make_tone(lib / "a.flac", seconds=15.0)
    make_tone(lib / "b.flac", seconds=15.0, freq=1200)
    conn = store.connect(tmp_path / "db.sqlite")
    scan.scan(conn, [lib])
    written = fingerprint.fingerprint_pending(conn)
    assert written == 2
    # Idempotent: a second call writes nothing new.
    assert fingerprint.fingerprint_pending(conn) == 0
    row = conn.execute("SELECT * FROM fingerprint LIMIT 1").fetchone()
    assert row["analyzer"] == "chromaprint"
    assert row["purpose"] == "canonical"
    assert len(row["fp_raw"]) == row["fp_length"] * 4
