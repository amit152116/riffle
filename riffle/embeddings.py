"""Audio embeddings from Discogs-EffNet (Essentia + TensorFlow).

The embedding is what similarity search runs on. On a 416-track benchmark it
found a known duplicate at rank 1 for 99% of tracks and put same-style
tracks in 81% of the top 10, against 85% and 56% for the MFCC-based score
it replaces (see riffle/similarity.py).

A 60 s excerpt from the middle of each track is embedded, not the whole
file: it keeps extraction to a few seconds per track and avoids intros and
outros, which are the parts that differ between copies of one song.
"""
from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from riffle import similarity

MODEL_FILE = "discogs-effnet-bs64-1.pb"
MODEL_URL = ("https://essentia.upf.edu/models/feature-extractors/"
             "discogs-effnet/" + MODEL_FILE)
EMBEDDING_OUTPUT = "PartitionedCall:1"     # 1280-d embedding (":0" is styles)
SAMPLE_RATE = 16000
EXCERPT_SECONDS = 60

_PREDICTORS: dict[str, object] = {}


def default_model_dir() -> Path:
    return Path(os.environ.get("RIFFLE_MODEL_DIR", "~/riffle-models")).expanduser()


def _require_model(model_dir) -> Path:
    model = Path(model_dir) / MODEL_FILE
    if not model.exists():
        raise FileNotFoundError(
            f"Model file {MODEL_FILE} not found in {model_dir}. "
            f"Download it from {MODEL_URL} (or set RIFFLE_MODEL_DIR).")
    return model


def _check_available() -> None:
    try:
        import essentia.standard as es
    except ImportError as exc:
        raise ImportError(
            "Essentia is not installed. Install it with: "
            "pip install essentia-tensorflow") from exc
    if not hasattr(es, "TensorflowPredictEffnetDiscogs"):
        raise ImportError(
            "This Essentia build has no TensorFlow support. Replace it with: "
            "pip install essentia-tensorflow")


def _predict(audio: np.ndarray, model_dir) -> np.ndarray:
    import essentia.standard as es

    model = str(_require_model(model_dir))
    if model not in _PREDICTORS:
        _PREDICTORS[model] = es.TensorflowPredictEffnetDiscogs(
            graphFilename=model, output=EMBEDDING_OUTPUT)
    patches = _PREDICTORS[model](audio)            # (patches, 1280)
    return np.mean(patches, axis=0).astype(np.float32)


def embed_file(path, model_dir=None) -> np.ndarray:
    """The track's 1280-d embedding, averaged over its middle 60 s."""
    import essentia.standard as es

    model_dir = model_dir or default_model_dir()
    _require_model(model_dir)
    audio = es.MonoLoader(filename=str(path), sampleRate=SAMPLE_RATE,
                          resampleQuality=4)()
    n = EXCERPT_SECONDS * SAMPLE_RATE
    if len(audio) > n:
        start = (len(audio) - n) // 2
        audio = audio[start:start + n]
    return _predict(audio, model_dir)


def _embed_or_error(path: str, model_dir: str):
    """Worker entry point: (vector, None) or (None, message)."""
    # One thread per worker: several workers each running a multi-threaded
    # TensorFlow session oversubscribe the CPU and exhaust memory.
    for k in ("OMP_NUM_THREADS", "TF_NUM_INTEROP_THREADS", "TF_NUM_INTRAOP_THREADS"):
        os.environ[k] = "1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    try:
        return embed_file(path, model_dir), None
    except Exception as exc:  # noqa: BLE001 - any decode/inference failure
        return None, f"{type(exc).__name__}: {exc}"


def embedding_scan(conn, limit: int | None = None, workers: int = 1,
                   model_dir=None, on_progress=None) -> dict:
    """Embed present audio that has no embedding yet (identical audio once)."""
    _check_available()
    model_dir = str(model_dir or default_model_dir())
    _require_model(model_dir)

    query = (
        "SELECT MIN(t.id) AS track_id, t.path AS path, "
        "       t.audio_content_id AS content_id "
        "FROM track t LEFT JOIN audio_embedding e "
        "  ON e.audio_content_id = t.audio_content_id AND e.model = ? "
        "WHERE t.present = 1 AND e.audio_content_id IS NULL "
        "GROUP BY t.audio_content_id ORDER BY MIN(t.id)"
    )
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    rows = conn.execute(query, (similarity.EMBEDDING_MODEL,)).fetchall()
    cached = conn.execute(
        "SELECT count(DISTINCT e.audio_content_id) AS n FROM audio_embedding e "
        "JOIN track t ON t.audio_content_id = e.audio_content_id AND t.present = 1 "
        "WHERE e.model = ?", (similarity.EMBEDDING_MODEL,)).fetchone()["n"]

    embedded = failed = 0
    todo = []
    for row in rows:
        if Path(row["path"]).exists():
            todo.append(row)
        else:
            failed += 1

    def record(row, outcome):
        nonlocal embedded, failed
        vector, error = outcome
        if error is not None:
            failed += 1
        else:
            conn.execute(
                "INSERT OR REPLACE INTO audio_embedding (audio_content_id, model, "
                "dim, vector, computed_at) VALUES (?,?,?,?,?)",
                (row["content_id"], similarity.EMBEDDING_MODEL, len(vector),
                 np.asarray(vector, dtype="<f4").tobytes(),
                 datetime.now(UTC).isoformat()))
            embedded += 1
        if on_progress is not None:
            on_progress(embedded + failed, len(rows))

    if workers <= 1:
        for row in todo:
            record(row, _embed_or_error(row["path"], model_dir))
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = {pool.submit(_embed_or_error, r["path"], model_dir): r
                       for r in todo}
            for fut in as_completed(futures):
                record(futures[fut], fut.result())

    return {"embedded": embedded, "failed": failed, "cached": cached}
