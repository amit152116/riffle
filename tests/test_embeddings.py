"""Audio embedding extraction (Discogs-EffNet via Essentia + TensorFlow).

Needs essentia-tensorflow and the model file; skipped otherwise.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from riffle import store
from tests.fixtures import make_tone
from tests.test_features import _db_with_track

MODEL_DIR = Path(os.environ.get("RIFFLE_MODEL_DIR", "~/riffle-models")).expanduser()


def _available() -> bool:
    try:
        import essentia.standard as es
        return (hasattr(es, "TensorflowPredictEffnetDiscogs")
                and (MODEL_DIR / "discogs-effnet-bs64-1.pb").exists())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _available(),
    reason="needs essentia-tensorflow and discogs-effnet-bs64-1.pb in ~/riffle-models")


def test_embed_file_returns_a_finite_1280_vector(tmp_path):
    from riffle import embeddings
    p = make_tone(tmp_path / "a.flac", seconds=8.0, volume=0.5)
    v = embeddings.embed_file(p, MODEL_DIR)
    assert v.shape == (1280,) and v.dtype == np.float32
    assert np.isfinite(v).all()
    assert float(np.abs(v).sum()) > 0


def test_embedding_is_deterministic(tmp_path):
    from riffle import embeddings
    p = make_tone(tmp_path / "a.flac", seconds=8.0, volume=0.5)
    assert np.array_equal(embeddings.embed_file(p, MODEL_DIR),
                          embeddings.embed_file(p, MODEL_DIR))


def test_different_audio_gives_different_embeddings(tmp_path):
    from riffle import embeddings
    a = embeddings.embed_file(make_tone(tmp_path / "a.flac", seconds=8.0, freq=300), MODEL_DIR)
    b = embeddings.embed_file(make_tone(tmp_path / "b.flac", seconds=8.0, freq=1900), MODEL_DIR)
    assert not np.allclose(a, b)


def test_long_tracks_use_a_middle_excerpt_not_the_whole_file(tmp_path, monkeypatch):
    from riffle import embeddings
    seen = {}
    real = embeddings._predict

    def spy(audio, model_dir):
        seen["seconds"] = len(audio) / 16000
        return real(audio, model_dir)

    monkeypatch.setattr(embeddings, "_predict", spy)
    p = make_tone(tmp_path / "long.flac", seconds=90.0, volume=0.5)
    embeddings.embed_file(p, MODEL_DIR)
    assert seen["seconds"] == pytest.approx(embeddings.EXCERPT_SECONDS, abs=0.5)


def test_scan_stores_one_embedding_per_audio_and_is_idempotent(tmp_path):
    from riffle import embeddings
    conn = store.connect(tmp_path / "db.sqlite")
    first = make_tone(tmp_path / "a.flac", seconds=8.0, volume=0.5)
    _db_with_track(conn, tmp_path, first, cid=1)
    copy = tmp_path / "copy.flac"
    copy.write_bytes(first.read_bytes())
    st = copy.stat()
    conn.execute("INSERT INTO track (id, path, size, mtime, audio_content_id, "
                 "bitrate, present) VALUES (2, ?, ?, ?, 1, 1000000, 1)",
                 (str(copy), st.st_size, st.st_mtime))
    r = embeddings.embedding_scan(conn, model_dir=MODEL_DIR)
    assert r["embedded"] == 1 and r["failed"] == 0
    row = conn.execute("SELECT * FROM audio_embedding").fetchone()
    assert row["dim"] == 1280 and len(row["vector"]) == 1280 * 4
    again = embeddings.embedding_scan(conn, model_dir=MODEL_DIR)
    assert again["embedded"] == 0 and again["cached"] == 1


def test_scan_with_workers_matches_serial(tmp_path):
    from riffle import embeddings

    def build(name):
        conn = store.connect(tmp_path / f"{name}.sqlite")
        for i, f in enumerate((330, 440, 550), start=1):
            p = make_tone(tmp_path / f"{name}_{i}.flac", seconds=8.0, freq=f, volume=0.5)
            _db_with_track(conn, tmp_path, p, cid=i)
        return conn

    def snap(conn):
        return {r["audio_content_id"]: np.frombuffer(r["vector"], dtype="<f4")
                for r in conn.execute("SELECT * FROM audio_embedding")}

    a, b = build("s"), build("p")
    assert embeddings.embedding_scan(a, model_dir=MODEL_DIR, workers=1)["embedded"] == 3
    assert embeddings.embedding_scan(b, model_dir=MODEL_DIR, workers=3)["embedded"] == 3
    # Not bit-identical: one thread against several rounds float32 sums
    # differently (measured ~3e-7 relative), which is irrelevant to similarity.
    sa, sb = snap(a), snap(b)
    assert sa.keys() == sb.keys()
    for cid in sa:
        assert np.allclose(sa[cid], sb[cid], rtol=1e-4, atol=1e-6)


def test_scan_counts_a_missing_file_as_failed(tmp_path):
    from riffle import embeddings
    conn = store.connect(tmp_path / "db.sqlite")
    conn.execute("INSERT INTO audio_content (id, audio_hash, hash_method) "
                 "VALUES (1, 'h', 'streamhash')")
    conn.execute("INSERT INTO track (id, path, size, mtime, audio_content_id, present) "
                 "VALUES (1, '/nonexistent/x.mp3', 1, 1.0, 1, 1)")
    r = embeddings.embedding_scan(conn, model_dir=MODEL_DIR)
    assert r["embedded"] == 0 and r["failed"] == 1


def test_missing_model_gives_a_clear_error(tmp_path):
    from riffle import embeddings
    p = make_tone(tmp_path / "a.flac", seconds=4.0)
    with pytest.raises(FileNotFoundError, match="discogs-effnet"):
        embeddings.embed_file(p, tmp_path / "no-models-here")


def test_cli_embed_reports_counts(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from riffle.cli import app
    monkeypatch.setenv("RIFFLE_MODEL_DIR", str(MODEL_DIR))
    p = make_tone(tmp_path / "a.flac", seconds=6.0, volume=0.5)
    conn = store.connect(tmp_path / "db.sqlite")
    _db_with_track(conn, tmp_path, p)
    conn.close()
    r = CliRunner().invoke(app, ["--db", str(tmp_path / "db.sqlite"), "embed", "--as-json"])
    assert r.exit_code == 0, r.stdout
    assert json.loads(r.stdout)["embedded"] == 1
