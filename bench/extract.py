"""Extract benchmark features for the subset, cached per track (resumable).

Stages
  baseline  the production features (riffle.features) plus MFCC std, on the
            full track. Needs plain Essentia only.
  embed     Discogs-EffNet and musicnn embeddings (+ style/tag probabilities)
            from a 60 s middle excerpt at 16 kHz. Needs essentia-tensorflow and
            the model files in ~/riffle-models.

Results go to ~/riffle-bench/cache/<track_id>.npz. The library database is
never touched.

    python bench/extract.py baseline --workers 4
    python bench/extract.py embed --workers 3
"""
import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

BENCH = Path(os.path.expanduser("~/riffle-bench"))
CACHE = BENCH / "cache"
MODELS = Path(os.path.expanduser("~/riffle-models"))
EXCERPT_SECONDS = 60


def _worker_env():
    # One thread per worker: several workers each spawning many threads
    # oversubscribes the CPU and the memory.
    for k in ("OMP_NUM_THREADS", "TF_NUM_INTEROP_THREADS",
              "TF_NUM_INTRAOP_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def baseline_one(job):
    tid, path = job
    _worker_env()
    out = CACHE / f"{tid}.npz"
    have = dict(np_load(out)) if out.exists() else {}
    if "mfcc_mean" in have:
        return tid, "cached", None
    try:
        import numpy as np
        import essentia.standard as es
        from riffle import features

        res = features.extract_file(path)
        # MFCC std needs the frames again; decode once more at the same rate.
        audio = es.MonoLoader(filename=str(path), sampleRate=44100)()
        w, sp = es.Windowing(type="hann"), es.Spectrum()
        mf = es.MFCC(numberCoefficients=13)
        frames = [mf(sp(w(fr)))[1] for fr in
                  es.FrameGenerator(audio, frameSize=2048, hopSize=1024)]
        res["mfcc_std"] = np.std(frames, axis=0).astype(np.float64)
        for k, v in list(res.items()):
            if isinstance(v, str):
                res[k] = np.array(v)
        have.update(res)
        np.savez(out, **have)
        return tid, "ok", None
    except Exception as exc:  # noqa: BLE001
        return tid, "error", f"{type(exc).__name__}: {exc}"


def np_load(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


_MODELS = {}


def _models():
    if not _MODELS:
        import essentia.standard as es
        _MODELS["eff_emb"] = es.TensorflowPredictEffnetDiscogs(
            graphFilename=str(MODELS / "discogs-effnet-bs64-1.pb"),
            output="PartitionedCall:1")
        _MODELS["eff_sty"] = es.TensorflowPredictEffnetDiscogs(
            graphFilename=str(MODELS / "discogs-effnet-bs64-1.pb"),
            output="PartitionedCall:0")
        _MODELS["mus_emb"] = es.TensorflowPredictMusiCNN(
            graphFilename=str(MODELS / "msd-musicnn-1.pb"),
            output="model/dense/BiasAdd")
        _MODELS["mus_tag"] = es.TensorflowPredictMusiCNN(
            graphFilename=str(MODELS / "msd-musicnn-1.pb"),
            output="model/Sigmoid")
    return _MODELS


def embed_one(job):
    tid, path = job
    _worker_env()
    out = CACHE / f"{tid}.npz"
    have = np_load(out) if out.exists() else {}
    if "effnet_emb" in have:
        return tid, "cached", None
    try:
        import numpy as np
        import essentia.standard as es

        audio = es.MonoLoader(filename=str(path), sampleRate=16000,
                              resampleQuality=4)()
        n = EXCERPT_SECONDS * 16000
        if len(audio) > n:
            start = (len(audio) - n) // 2
            audio = audio[start:start + n]
        m = _models()
        have["effnet_emb"] = np.mean(m["eff_emb"](audio), axis=0)
        have["effnet_sty"] = np.mean(m["eff_sty"](audio), axis=0)
        have["musicnn_emb"] = np.mean(m["mus_emb"](audio), axis=0)
        have["musicnn_tag"] = np.mean(m["mus_tag"](audio), axis=0)
        np.savez(out, **have)
        return tid, "ok", None
    except Exception as exc:  # noqa: BLE001
        return tid, "error", f"{type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["baseline", "embed"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    subset = json.load(open(BENCH / "subset.json"))
    jobs = [(t["id"], t["path"]) for t in subset]
    if a.limit:
        jobs = jobs[:a.limit]
    fn = baseline_one if a.stage == "baseline" else embed_one
    ctx = multiprocessing.get_context("spawn")
    t0 = time.time(); done = ok = err = cached = 0
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as pool:
        futs = [pool.submit(fn, j) for j in jobs]
        for f in as_completed(futs):
            tid, status, msg = f.result(); done += 1
            ok += status == "ok"; cached += status == "cached"; err += status == "error"
            if status == "error":
                print(f"  ERROR track {tid}: {msg}", flush=True)
            if done % 20 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}  ok={ok} cached={cached} err={err}  "
                      f"{time.time()-t0:.0f}s", flush=True)
    print(f"{a.stage}: ok={ok} cached={cached} errors={err} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
