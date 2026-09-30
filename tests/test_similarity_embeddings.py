"""Embedding-based similarity index (with mutual proximity).

Benchmarked on 416 real tracks: Discogs-EffNet embeddings found a known
duplicate at rank 1 for 99% of tracks (current MFCC-based score: 85%) and put
same-style neighbours in 81% of the top 10 (current: 56%, chance 50%).
"""
import json

import numpy as np
import pytest

from riffle import similarity, store

MODEL = similarity.EMBEDDING_MODEL


def _library(tmp_path, vectors, present=None):
    """vectors: list of 1-D arrays; track i+1 gets vectors[i]."""
    conn = store.connect(tmp_path / "db.sqlite")
    for i, v in enumerate(vectors, start=1):
        conn.execute("INSERT INTO audio_content (id, audio_hash, hash_method) "
                     "VALUES (?,?, 'streamhash')", (i, f"h{i}"))
        conn.execute("INSERT INTO track (id, path, audio_content_id, present) "
                     "VALUES (?,?,?,?)",
                     (i, f"/x/{i}.flac", i,
                      1 if present is None or i in present else 0))
        conn.execute("INSERT INTO audio_embedding (audio_content_id, model, dim, "
                     "vector, computed_at) VALUES (?,?,?,?, 'now')",
                     (i, MODEL, len(v), np.asarray(v, "<f4").tobytes()))
    return conn


def _two_clusters(rng, per=8, dim=32):
    a, b = rng.normal(size=dim), rng.normal(size=dim)
    vecs = [a + 0.15 * rng.normal(size=dim) for _ in range(per)]
    vecs += [b + 0.15 * rng.normal(size=dim) for _ in range(per)]
    return vecs


def test_neighbours_come_from_the_same_cluster(tmp_path):
    conn = _library(tmp_path, _two_clusters(np.random.default_rng(0)))
    similarity.build_embedding_similarity(conn, top_k=5)
    for tid in range(1, 17):
        rows = conn.execute("SELECT neighbor_id FROM track_similarity "
                            "WHERE track_id = ?", (tid,)).fetchall()
        same_side = {n["neighbor_id"] <= 8 for n in rows}
        assert same_side == {tid <= 8}, tid


def test_stores_top_k_rows_and_the_method_hash(tmp_path):
    conn = _library(tmp_path, _two_clusters(np.random.default_rng(1)))
    result = similarity.build_embedding_similarity(conn, top_k=4)
    assert result["tracks"] == 16
    counts = {r["c"] for r in conn.execute(
        "SELECT count(*) c FROM track_similarity GROUP BY track_id")}
    assert counts == {4}
    hashes = {r["config_hash"] for r in conn.execute(
        "SELECT DISTINCT config_hash FROM track_similarity")}
    assert hashes == {similarity.embedding_config_hash()}
    assert similarity.embedding_config_hash() != similarity.similarity_config_hash()


def test_scores_are_between_0_and_1_and_best_first(tmp_path):
    conn = _library(tmp_path, _two_clusters(np.random.default_rng(2)))
    similarity.build_embedding_similarity(conn, top_k=6)
    for r in conn.execute("SELECT combined_score FROM track_similarity"):
        assert 0.0 <= r["combined_score"] <= 1.0
    near = conn.execute("SELECT combined_score FROM track_similarity "
                        "WHERE track_id = 1 ORDER BY combined_score").fetchall()
    assert [r["combined_score"] for r in near] == sorted(
        r["combined_score"] for r in near)


def test_absent_tracks_are_not_indexed_or_suggested(tmp_path):
    conn = _library(tmp_path, _two_clusters(np.random.default_rng(3)),
                    present=set(range(1, 16)))  # track 16 quarantined
    similarity.build_embedding_similarity(conn, top_k=5)
    assert conn.execute("SELECT count(*) c FROM track_similarity "
                        "WHERE track_id = 16 OR neighbor_id = 16"
                        ).fetchone()["c"] == 0


def test_rebuilding_replaces_the_previous_index(tmp_path):
    conn = _library(tmp_path, _two_clusters(np.random.default_rng(4)))
    similarity.build_embedding_similarity(conn, top_k=5)
    conn.execute("INSERT OR REPLACE INTO track_similarity (track_id, neighbor_id, "
                 "mfcc_norm, combined_score, config_hash) VALUES (1, 2, 0.1, 0.1, 'old-method')")
    similarity.build_embedding_similarity(conn, top_k=3)
    assert conn.execute("SELECT count(*) c FROM track_similarity WHERE config_hash = 'old-method'"
                        ).fetchone()["c"] == 0
    assert {r["c"] for r in conn.execute(
        "SELECT count(*) c FROM track_similarity GROUP BY track_id")} == {3}


def test_mutual_proximity_is_symmetric_bounded_with_zero_diagonal():
    rng = np.random.default_rng(5)
    D = similarity.cosine_distance_matrix(rng.normal(size=(30, 16)))
    mp = similarity.mutual_proximity(D)
    assert np.allclose(mp, mp.T)
    assert np.allclose(np.diag(mp), 0.0)
    assert mp.min() >= 0.0 and mp.max() <= 1.0


def test_mutual_proximity_reduces_hubs():
    # In high dimensions a point near the middle of the cloud is close to
    # almost everything: a "hub" that would be suggested over and over.
    rng = np.random.default_rng(0)
    cloud = rng.normal(size=(120, 64))
    X = np.vstack([cloud, cloud.mean(axis=0, keepdims=True)])   # last = hub

    def max_occurrences(D, k=10):
        D = D.copy(); np.fill_diagonal(D, np.inf)
        counts = np.bincount(np.argsort(D, axis=1)[:, :k].ravel(), minlength=len(D))
        return counts.max()

    D = similarity.cosine_distance_matrix(X)
    assert max_occurrences(similarity.mutual_proximity(D)) < max_occurrences(D)


def test_build_index_uses_embeddings_when_available(tmp_path):
    conn = _library(tmp_path, _two_clusters(np.random.default_rng(6)))
    result = similarity.build_index(conn, method="auto", top_k=4)
    assert result["method"] == "embedding"
    assert conn.execute("SELECT count(*) c FROM track_similarity WHERE config_hash = ?",
                        (similarity.embedding_config_hash(),)).fetchone()["c"] > 0


def test_build_index_falls_back_to_mfcc_without_embeddings(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    result = similarity.build_index(conn, method="auto", top_k=4)
    assert result["method"] == "mfcc"


def test_embedding_method_without_embeddings_is_an_error(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    with pytest.raises(ValueError, match="embed"):
        similarity.build_index(conn, method="embedding", top_k=4)


def test_cli_build_index_reports_the_method(tmp_path):
    from typer.testing import CliRunner
    from riffle.cli import app
    conn = _library(tmp_path, _two_clusters(np.random.default_rng(7)))
    conn.close()
    result = CliRunner().invoke(app, ["--db", str(tmp_path / "db.sqlite"),
                                      "build-index", "--as-json"])
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["method"] == "embedding"


def test_embedding_index_stores_a_deep_neighbour_list_by_default(tmp_path):
    # Shuffle re-ranks a pool of ~80 neighbours, so the default must keep more.
    rng = np.random.default_rng(8)
    conn = _library(tmp_path, [rng.normal(size=8) for _ in range(130)])
    similarity.build_index(conn, method="embedding")
    assert {r["c"] for r in conn.execute(
        "SELECT count(*) c FROM track_similarity GROUP BY track_id")} == {100}
