"""Effective bandwidth: the highest frequency a lossy file actually keeps.

A "320 kbps" mp3 made from a 128 kbps source still cuts off near 16 kHz, so
nominal bitrate alone cannot say which copy is better. The estimator looks
for the low-pass *cliff*; a file with no cliff reports `cutoff_hz=None`,
which is distinct from `None` (unmeasurable) returned for the whole result.
"""
import subprocess
import wave

import numpy as np

from riffle import bandwidth, fingerprint, scan, store

SR = 44100


def _noise_wav(path, seconds=12.0, lowpass_hz=None, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(SR * seconds))
    if lowpass_hz is not None:
        spec = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(len(x), 1 / SR)
        spec[freqs > lowpass_hz] = 0  # brickwall
        x = np.fft.irfft(spec, n=len(x))
    x = x / np.abs(x).max() * 0.5
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    return path


def _soft_cliff_wav(path, seconds=12.0, start_hz=15000, width_hz=2500,
                    fall_db=40.0, seed=0):
    """Noise that rolls off by `fall_db` over `width_hz`, then stays low.

    LAME's soft low-pass looks like this; its steepest 1 kHz drop is well
    under a brickwall's.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(SR * seconds))
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    frac = np.clip((freqs - start_hz) / width_hz, 0, 1)
    spec *= 10 ** (-fall_db * frac / 20)
    x = np.fft.irfft(spec, n=len(x))
    x = x / np.abs(x).max() * 0.5
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    return path


def _tilted_noise_wav(path, seconds=12.0, tilt=1.0, seed=0):
    """Noise whose power falls as 1/f**(2*tilt): music-like, but no cliff."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(SR * seconds))
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    spec *= 1.0 / np.maximum(freqs, 100.0) ** tilt
    x = np.fft.irfft(spec, n=len(x))
    x = x / np.abs(x).max() * 0.5
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    return path


def _encode(src, dst, bitrate):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(src), "-ac", "2",  # stereo, as real rips are:
                    # a mono stream gets twice the bits per channel and so
                    # a much higher low-pass than its nominal bitrate implies
                    "-c:a", "libmp3lame", "-b:a", bitrate,
                    str(dst)], check=True)
    return dst


def test_finds_a_brickwall_cutoff(tmp_path):
    wav = _noise_wav(tmp_path / "lp16.wav", lowpass_hz=16000)
    result = bandwidth.measure_cutoff(wav, duration=12.0)
    assert result is not None
    assert result.cutoff_hz is not None
    assert 15000 <= result.cutoff_hz <= 17000
    assert result.cliff_db >= 20


def test_full_band_signal_has_no_cliff(tmp_path):
    wav = _noise_wav(tmp_path / "full.wav")
    result = bandwidth.measure_cutoff(wav, duration=12.0)
    assert result is not None
    assert result.cutoff_hz is None


def test_low_bitrate_mp3_cuts_off_below_a_high_bitrate_one(tmp_path):
    wav = _noise_wav(tmp_path / "src.wav", seconds=20.0)
    low = bandwidth.measure_cutoff(_encode(wav, tmp_path / "128.mp3", "128k"),
                                   duration=20.0)
    high = bandwidth.measure_cutoff(_encode(wav, tmp_path / "320.mp3", "320k"),
                                    duration=20.0)
    assert low.cutoff_hz is not None and low.cutoff_hz < 19000
    assert high.cutoff_hz is None or high.cutoff_hz > low.cutoff_hz + 1500


def test_a_320k_transcode_of_a_16k_source_is_still_seen_as_16k(tmp_path):
    src = _noise_wav(tmp_path / "src16.wav", seconds=20.0, lowpass_hz=16000)
    result = bandwidth.measure_cutoff(
        _encode(src, tmp_path / "up320.mp3", "320k"), duration=20.0)
    assert result.cutoff_hz is not None
    assert 15000 <= result.cutoff_hz <= 17000


def test_unreadable_or_too_short_audio_is_unmeasurable(tmp_path):
    empty = tmp_path / "empty.mp3"
    empty.write_bytes(b"")
    assert bandwidth.measure_cutoff(empty, duration=None) is None
    tiny = _noise_wav(tmp_path / "tiny.wav", seconds=0.05)
    assert bandwidth.measure_cutoff(tiny, duration=0.05) is None


def _library_with_one_track(tmp_path):
    lib = tmp_path / "lib"
    _noise_wav(lib / "a.wav", seconds=12.0, lowpass_hz=16000)
    conn = store.connect(tmp_path / "db.sqlite")
    scan.scan(conn, [lib])
    return conn


def test_scan_stores_a_measurement_once_per_content(tmp_path):
    conn = _library_with_one_track(tmp_path)
    summary = bandwidth.bandwidth_scan(conn)
    assert summary == {"measured": 1, "unmeasurable": 0, "cached": 0}
    row = conn.execute("SELECT * FROM audio_bandwidth").fetchone()
    assert 15000 <= row["cutoff_hz"] <= 17000
    assert row["measured_at"]
    # A second pass measures nothing new.
    assert bandwidth.bandwidth_scan(conn) == \
        {"measured": 0, "unmeasurable": 0, "cached": 1}


def test_a_full_band_file_is_stored_as_measured_with_no_cutoff(tmp_path):
    lib = tmp_path / "lib"
    _noise_wav(lib / "full.wav", seconds=12.0)
    conn = store.connect(tmp_path / "db.sqlite")
    scan.scan(conn, [lib])
    bandwidth.bandwidth_scan(conn)
    row = conn.execute("SELECT * FROM audio_bandwidth").fetchone()
    assert row is not None          # measured ...
    assert row["cutoff_hz"] is None  # ... and found to have no cliff


def test_finds_cutoffs_well_below_16_khz(tmp_path):
    # Low-bitrate or low-sample-rate files cut off near 9-11 kHz. Reading
    # those as "no cliff" would rank them as the *widest* copy.
    for hz in (9000, 11000):
        wav = _noise_wav(tmp_path / f"lp{hz}.wav", lowpass_hz=hz)
        result = bandwidth.measure_cutoff(wav, duration=12.0)
        assert result.cutoff_hz is not None, hz
        assert hz - 1000 <= result.cutoff_hz <= hz + 1000, (hz, result)


def test_a_22khz_sample_rate_mp3_is_seen_as_limited_to_11_khz(tmp_path):
    src = _noise_wav(tmp_path / "src.wav", seconds=20.0)
    low = tmp_path / "sr22.mp3"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(src), "-ac", "2", "-ar", "22050",
                    "-c:a", "libmp3lame", "-b:a", "64k", str(low)], check=True)
    result = bandwidth.measure_cutoff(low, duration=20.0)
    assert result.cutoff_hz is not None
    assert result.cutoff_hz <= 12000


def test_natural_spectral_tilt_is_not_mistaken_for_a_cliff(tmp_path):
    for tilt in (0.5, 1.0):  # -3 and -6 dB per octave
        wav = _tilted_noise_wav(tmp_path / f"tilt{tilt}.wav", tilt=tilt)
        result = bandwidth.measure_cutoff(wav, duration=12.0)
        assert result.cutoff_hz is None, (tilt, result)


def test_remeasure_replaces_existing_measurements(tmp_path):
    conn = _library_with_one_track(tmp_path)
    bandwidth.bandwidth_scan(conn)
    conn.execute("UPDATE audio_bandwidth SET cutoff_hz = 99999, cliff_db = 1")
    summary = bandwidth.bandwidth_scan(conn, remeasure=True)
    assert summary["measured"] == 1
    row = conn.execute("SELECT cutoff_hz FROM audio_bandwidth").fetchone()
    assert 15000 <= row["cutoff_hz"] <= 17000
    assert conn.execute("SELECT count(*) c FROM audio_bandwidth"
                        ).fetchone()["c"] == 1


def test_remeasure_leaves_absent_files_measurements_alone(tmp_path):
    conn = _library_with_one_track(tmp_path)
    bandwidth.bandwidth_scan(conn)
    conn.execute("UPDATE track SET present = 0")
    bandwidth.bandwidth_scan(conn, remeasure=True)
    assert conn.execute("SELECT count(*) c FROM audio_bandwidth"
                        ).fetchone()["c"] == 1


def test_a_soft_rolloff_is_still_a_cliff(tmp_path):
    # 40 dB over 2.5 kHz is ~16 dB in the steepest kHz: below a brickwall,
    # but the file clearly has nothing above ~17 kHz.
    wav = _soft_cliff_wav(tmp_path / "soft.wav")
    result = bandwidth.measure_cutoff(wav, duration=12.0)
    assert result.cutoff_hz is not None
    assert 15500 <= result.cutoff_hz <= 17500, result


def _many_files(tmp_path, n=4):
    lib = tmp_path / "lib"
    for i in range(n):
        _noise_wav(lib / f"f{i}.wav", seconds=10.0,
                   lowpass_hz=12000 + 1000 * i, seed=i)
    conn = store.connect(tmp_path / "db.sqlite")
    scan.scan(conn, [lib])
    return conn


def test_parallel_workers_give_the_same_measurements(tmp_path):
    def snapshot(conn):
        return {r["audio_content_id"]: (r["cutoff_hz"], round(r["cliff_db"], 6))
                for r in conn.execute("SELECT * FROM audio_bandwidth")}
    serial = _many_files(tmp_path / "s")
    assert bandwidth.bandwidth_scan(serial, workers=1)["measured"] == 4
    parallel = _many_files(tmp_path / "p")
    assert bandwidth.bandwidth_scan(parallel, workers=4)["measured"] == 4
    assert snapshot(serial) == snapshot(parallel)
