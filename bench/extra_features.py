"""Candidate extra DSP features for the benchmark subset, cached per track.

From a 60 s middle excerpt at 44.1 kHz (frame 2048, hop 1024, like production):
  rolloff85, rolloff95   spectral rolloff (Hz), mean over frames
  flux_mean, flux_std    spectral flux
  contrast_mean          spectral contrast, mean over bands and frames
  bass, lowmid, mid, highmid, treble   energy share in 20-250-500-2k-4k-20k Hz
  rms_mean, rms_std      frame RMS
  dyn_range              p95 - p5 of frame RMS in dB
  chroma_entropy         entropy of the mean HPCP (12 pitch classes)
Results go to ~/riffle-bench/extra/<track_id>.npz (resumable). Read-only on
the library and its database.

    python bench/extra_features.py --workers 3
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

BENCH = Path(os.path.expanduser("~/riffle-bench"))
OUT = BENCH / "extra"
SR, FS, HS, SECONDS = 44100, 2048, 1024, 60
BANDS = [20, 250, 500, 2000, 4000, 20000]
NAMES = ["bass", "lowmid", "mid", "highmid", "treble"]


def one(job):
    tid, path = job
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    out = OUT / f"{tid}.npz"
    if out.exists():
        return tid, "cached", None
    try:
        import essentia.standard as es
        audio = es.MonoLoader(filename=str(path), sampleRate=SR)()
        n = SECONDS * SR
        if len(audio) > n:
            s = (len(audio) - n) // 2
            audio = audio[s:s + n]
        w, sp = es.Windowing(type="hann"), es.Spectrum(size=FS)
        r85, r95 = es.RollOff(cutoff=0.85, sampleRate=SR), es.RollOff(cutoff=0.95, sampleRate=SR)
        flux = es.Flux()
        contrast = es.SpectralContrast(frameSize=FS, sampleRate=SR)
        fb = es.FrequencyBands(frequencyBands=BANDS, sampleRate=SR)
        rms = es.RMS()
        spp, hpcp = es.SpectralPeaks(sampleRate=SR, maxPeaks=60), es.HPCP(sampleRate=SR)
        R85, R95, FL, CT, BD, RM, HP = [], [], [], [], [], [], []
        for fr in es.FrameGenerator(audio, frameSize=FS, hopSize=HS):
            spec = sp(w(fr))
            R85.append(r85(spec)); R95.append(r95(spec)); FL.append(flux(spec))
            c, _ = contrast(spec); CT.append(np.mean(c))
            BD.append(fb(spec)); RM.append(rms(fr))
            f, m = spp(spec)
            HP.append(hpcp(f, m))
        RM = np.array(RM); db = 20 * np.log10(np.maximum(RM, 1e-6))
        bd = np.array(BD).sum(0); bd = bd / (bd.sum() + 1e-12)
        h = np.mean(HP, axis=0); h = h / (h.sum() + 1e-12)
        d = {"rolloff85": np.mean(R85), "rolloff95": np.mean(R95),
             "flux_mean": np.mean(FL), "flux_std": np.std(FL),
             "contrast_mean": np.mean(CT), "rms_mean": RM.mean(), "rms_std": RM.std(),
             "dyn_range": np.percentile(db, 95) - np.percentile(db, 5),
             "chroma_entropy": float(-(h * np.log2(h + 1e-12)).sum())}
        d.update({k: float(v) for k, v in zip(NAMES, bd)})
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez(out, **{k: float(v) for k, v in d.items()})
        return tid, "ok", None
    except Exception as exc:  # noqa: BLE001
        return tid, "error", f"{type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    subset = json.load(open(BENCH / "subset.json"))
    jobs = [(t["id"], t["path"]) for t in subset if Path(t["path"]).exists()]
    done = errs = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(one, j) for j in jobs]):
            tid, st, msg = f.result(); done += 1
            if st == "error":
                errs += 1; print(f"error {tid}: {msg}", flush=True)
            if done % 25 == 0:
                print(f"{done}/{len(jobs)}", flush=True)
    print(f"done {done}, errors {errs}")


if __name__ == "__main__":
    sys.exit(main())
