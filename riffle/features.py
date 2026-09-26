"""Audio feature extraction via Essentia DSP algorithms."""
from __future__ import annotations

import hashlib
import json

import numpy as np

EXTRACTION_CONFIG = {
    "sample_rate": 44100,
    "frame_size": 2048,
    "hop_size": 1024,
    "window_type": "hann",
    "mfcc_bands": 13,
    "rhythm_method": "multifeature",
}


def extraction_config_hash() -> str:
    blob = json.dumps(EXTRACTION_CONFIG, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _correct_bpm(bpm: float) -> float:
    if bpm < 60.0:
        return bpm * 2.0
    if bpm > 200.0:
        return bpm / 2.0
    return bpm


def _essentia_version() -> str:
    import essentia
    return essentia.__version__


def extract_file(path) -> dict:
    from pathlib import Path
    import essentia.standard as es

    path = Path(path)
    sr = EXTRACTION_CONFIG["sample_rate"]
    audio = es.MonoLoader(filename=str(path), sampleRate=sr)()

    bpm, ticks, confidence, _, _ = es.RhythmExtractor2013(
        method=EXTRACTION_CONFIG["rhythm_method"]
    )(audio)
    bpm = _correct_bpm(float(bpm))

    key, scale, strength = es.KeyExtractor()(audio)

    stereo_audio = es.StereoMuxer()(audio, audio)
    loudness = es.LoudnessEBUR128(sampleRate=sr)(stereo_audio)
    lufs = float(loudness[2])  # integrated loudness

    dance, _ = es.Danceability()(audio)

    raw_energy = float(es.Energy()(audio))
    n_samples = len(audio)
    max_energy = float(n_samples)  # max energy = sum of 1.0^2 * n
    energy = raw_energy / max_energy if max_energy > 0 else 0.0
    energy = min(energy, 1.0)

    sc = float(es.SpectralCentroidTime()(audio))

    _, onset_rate = es.OnsetRate()(audio)

    dc, _ = es.DynamicComplexity()(audio)

    fs = EXTRACTION_CONFIG["frame_size"]
    hs = EXTRACTION_CONFIG["hop_size"]
    mfcc_frames = []
    dissonance_frames = []
    for frame in es.FrameGenerator(audio, frameSize=fs, hopSize=hs):
        windowed = es.Windowing(type=EXTRACTION_CONFIG["window_type"])(frame)
        spectrum = es.Spectrum()(windowed)
        _, mfcc_coeffs = es.MFCC(numberCoefficients=EXTRACTION_CONFIG["mfcc_bands"])(spectrum)
        mfcc_frames.append(mfcc_coeffs)
        freqs, mags = es.SpectralPeaks()(spectrum)
        if len(freqs) >= 2:
            dissonance_frames.append(float(es.Dissonance()(freqs, mags)))

    mfcc_mean = np.mean(mfcc_frames, axis=0).astype(np.float64) if mfcc_frames else np.zeros(13)

    zcr = float(es.ZeroCrossingRate()(audio))

    avg_dissonance = float(np.mean(dissonance_frames)) if dissonance_frames else 0.0

    return {
        "bpm": bpm,
        "bpm_confidence": float(confidence),
        "key_name": str(key),
        "scale": str(scale),
        "key_strength": float(strength),
        "loudness_lufs": lufs,
        "danceability": float(dance),
        "energy": energy,
        "spectral_centroid": sc,
        "onset_rate": float(onset_rate),
        "dynamic_complexity": float(dc),
        "dissonance": avg_dissonance,
        "zcr": zcr,
        "mfcc_mean": mfcc_mean,
        "extractor_version": _essentia_version(),
        "config_hash": extraction_config_hash(),
    }


from datetime import UTC, datetime
from pathlib import Path

from riffle import store


def feature_scan(conn, limit: int | None = None) -> dict:
    query = (
        "SELECT t.id AS track_id, t.path, t.audio_content_id "
        "FROM track t "
        "LEFT JOIN audio_features af ON af.audio_content_id = t.audio_content_id "
        "WHERE t.present = 1 AND af.id IS NULL "
        "ORDER BY t.id"
    )
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    rows = conn.execute(query).fetchall()

    already = conn.execute(
        "SELECT count(*) c FROM track t "
        "JOIN audio_features af ON af.audio_content_id = t.audio_content_id "
        "WHERE t.present = 1"
    ).fetchone()["c"]

    analyzed = failed = 0
    for row in rows:
        path = Path(row["path"])
        if not path.exists():
            failed += 1
            continue
        try:
            result = extract_file(path)
        except Exception:
            failed += 1
            continue

        conn.execute(
            "INSERT INTO audio_features (audio_content_id, bpm, bpm_confidence, "
            "key_name, scale, key_strength, loudness_lufs, danceability, energy, "
            "spectral_centroid, onset_rate, dynamic_complexity, dissonance, zcr, "
            "mfcc_mean, extractor_version, config_hash, analyzed_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(audio_content_id) DO UPDATE SET "
            "bpm=excluded.bpm, bpm_confidence=excluded.bpm_confidence, "
            "key_name=excluded.key_name, scale=excluded.scale, "
            "key_strength=excluded.key_strength, loudness_lufs=excluded.loudness_lufs, "
            "danceability=excluded.danceability, energy=excluded.energy, "
            "spectral_centroid=excluded.spectral_centroid, onset_rate=excluded.onset_rate, "
            "dynamic_complexity=excluded.dynamic_complexity, dissonance=excluded.dissonance, "
            "zcr=excluded.zcr, mfcc_mean=excluded.mfcc_mean, "
            "extractor_version=excluded.extractor_version, config_hash=excluded.config_hash, "
            "analyzed_at=excluded.analyzed_at",
            (row["audio_content_id"], result["bpm"], result["bpm_confidence"],
             result["key_name"], result["scale"], result["key_strength"],
             result["loudness_lufs"], result["danceability"], result["energy"],
             result["spectral_centroid"], result["onset_rate"],
             result["dynamic_complexity"], result["dissonance"], result["zcr"],
             store.pack_mfcc(result["mfcc_mean"]),
             result["extractor_version"], result["config_hash"],
             datetime.now(UTC).isoformat()),
        )
        analyzed += 1

    return {"analyzed": analyzed, "failed": failed, "cached": already}


def render_features(conn) -> str:
    total = conn.execute("SELECT count(*) c FROM audio_features").fetchone()["c"]

    lines = ["Audio Features Report", "=" * 40, ""]
    lines.append(f"Analyzed:            {total}")

    if total == 0:
        return "\n".join(lines)

    bpm_buckets = [(60, 80), (80, 100), (100, 120), (120, 140), (140, 160), (160, 200)]
    lines.append("")
    lines.append("BPM Distribution")
    lines.append("-" * 20)
    for lo, hi in bpm_buckets:
        n = conn.execute(
            "SELECT count(*) c FROM audio_features WHERE bpm >= ? AND bpm < ?",
            (lo, hi)
        ).fetchone()["c"]
        lines.append(f"  {lo:>3}-{hi:<3}  {n:>5}")
    over = conn.execute(
        "SELECT count(*) c FROM audio_features WHERE bpm >= 200"
    ).fetchone()["c"]
    lines.append(f"  200+     {over:>5}")

    lines.append("")
    lines.append("Key Distribution (top 5)")
    lines.append("-" * 20)
    for r in conn.execute(
        "SELECT key_name, scale, count(*) c FROM audio_features "
        "WHERE key_name IS NOT NULL "
        "GROUP BY key_name, scale ORDER BY c DESC LIMIT 5"
    ).fetchall():
        lines.append(f"  {r['key_name']} {r['scale']:<6} {r['c']:>5}")

    loudness = conn.execute(
        "SELECT min(loudness_lufs) mn, max(loudness_lufs) mx, avg(loudness_lufs) av "
        "FROM audio_features WHERE loudness_lufs IS NOT NULL"
    ).fetchone()
    if loudness["mn"] is not None:
        lines.append("")
        lines.append("Loudness (LUFS)")
        lines.append("-" * 20)
        lines.append(f"  Min: {loudness['mn']:.1f}")
        lines.append(f"  Max: {loudness['mx']:.1f}")
        lines.append(f"  Avg: {loudness['av']:.1f}")

    return "\n".join(lines)
