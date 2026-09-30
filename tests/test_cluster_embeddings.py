"""Clustering on embeddings: spectral clustering of the mutual-proximity graph.

Benchmarked on 416 real tracks against k-means on riffle's own features at
the same k: label purity 0.82 vs 0.73, NMI 0.22 vs 0.11, known duplicates
kept together 1.00 vs 0.77, stability across subsamples 0.63 vs 0.63 (and 0.66
vs 0.45 at k=10); k-means on the embeddings themselves was far less stable
(0.17-0.21). Cluster count is not auto-detected: eigengap and silhouette
chose wildly different k on subsamples, and quality is flat across k.
"""
import json
import warnings

import numpy as np
import pytest

from riffle import cluster, similarity, store

warnings.filterwarnings("ignore")
MODEL = similarity.EMBEDDING_MODEL


def _blobs(rng, groups=3, per=12, dim=32, spread=0.15):
    centres = [rng.normal(size=dim) * 3 for _ in range(groups)]
    vecs, truth = [], []
    for g, c in enumerate(centres):
        for _ in range(per):
            vecs.append(c + spread * rng.normal(size=dim)); truth.append(g)
    return vecs, truth, centres


def _library(tmp_path, vectors, present=None):
    conn = store.connect(tmp_path / "db.sqlite")
    for i, v in enumerate(vectors, start=1):
        _add(conn, i, v, present=(present is None or i in present))
    return conn


def _add(conn, i, v, present=True):
    conn.execute("INSERT INTO audio_content (id, audio_hash, hash_method, duration) "
                 "VALUES (?,?, 'streamhash', 200.0)", (i, f"h{i}"))
    conn.execute("INSERT INTO track (id, path, size, mtime, audio_content_id, "
                 "tag_artist, tag_title, present) VALUES (?,?,1,1.0,?,?,?,?)",
                 (i, f"/m/{i}.mp3", i, f"A{i}", f"T{i}", 1 if present else 0))
    conn.execute("INSERT INTO audio_embedding (audio_content_id, model, dim, vector, "
                 "computed_at) VALUES (?,?,?,?, 'now')",
                 (i, MODEL, len(v), np.asarray(v, "<f4").tobytes()))


def _assignments(conn, run_id):
    return {r["audio_content_id"]: r["cluster_id"] for r in conn.execute(
        "SELECT * FROM cluster_assignment WHERE run_id = ?", (run_id,))}


def _agreement(assign, truth):
    """1.0 when clusters are exactly the generating groups (up to naming)."""
    from sklearn.metrics import adjusted_rand_score
    ids = sorted(assign)
    return adjusted_rand_score([truth[i - 1] for i in ids], [assign[i] for i in ids])


def test_recovers_well_separated_groups(tmp_path):
    vecs, truth, _ = _blobs(np.random.default_rng(0))
    conn = _library(tmp_path, vecs)
    result = cluster.cluster_tracks(conn, n_clusters=3)
    assert result["method"] == "embedding" and result["n_clusters"] == 3
    assert _agreement(_assignments(conn, result["run_id"]), truth) == pytest.approx(1.0)


def test_every_present_content_is_assigned_and_absent_ones_are_not(tmp_path):
    vecs, _, _ = _blobs(np.random.default_rng(1))
    conn = _library(tmp_path, vecs, present=set(range(1, 36)))   # 36 quarantined
    result = cluster.cluster_tracks(conn, n_clusters=3)
    assigned = _assignments(conn, result["run_id"])
    assert set(assigned) == set(range(1, 36))
    assert sum(result["sizes"]) == 35


def test_run_records_the_method_and_centroids_and_distances(tmp_path):
    vecs, _, _ = _blobs(np.random.default_rng(2))
    conn = _library(tmp_path, vecs)
    result = cluster.cluster_tracks(conn, n_clusters=3)
    meta = json.loads(conn.execute("SELECT scaler_params FROM cluster_run WHERE id = ?",
                                   (result["run_id"],)).fetchone()["scaler_params"])
    assert meta["method"] == "spectral-knn" and meta["model"] == MODEL
    cents = conn.execute("SELECT * FROM cluster_centroid WHERE run_id = ?",
                         (result["run_id"],)).fetchall()
    assert len(cents) == 3 and sum(c["n_tracks"] for c in cents) == 36
    dists = [r["distance_to_centroid"] for r in conn.execute(
        "SELECT distance_to_centroid FROM cluster_assignment WHERE run_id = ?",
        (result["run_id"],))]
    assert all(0.0 <= d <= 2.0 for d in dists)


def test_default_cluster_count_is_eight(tmp_path):
    vecs, _, _ = _blobs(np.random.default_rng(3), groups=8, per=8)
    conn = _library(tmp_path, vecs)
    assert cluster.EMBEDDING_CLUSTERS == 8
    assert cluster.cluster_tracks(conn)["n_clusters"] == 8


def test_results_are_deterministic(tmp_path):
    vecs, _, _ = _blobs(np.random.default_rng(4))
    a = _library(tmp_path / "a", vecs); b = _library(tmp_path / "b", vecs)
    ra = cluster.cluster_tracks(a, n_clusters=3); rb = cluster.cluster_tracks(b, n_clusters=3)
    assert _assignments(a, ra["run_id"]) == _assignments(b, rb["run_id"])


def test_too_many_clusters_for_the_library_is_a_clear_error(tmp_path):
    vecs, _, _ = _blobs(np.random.default_rng(5), groups=2, per=10)
    conn = _library(tmp_path, vecs)
    with pytest.raises(ValueError, match="only 20"):
        cluster.cluster_tracks(conn, n_clusters=40)


def test_a_tiny_library_becomes_one_cluster(tmp_path):
    vecs, _, _ = _blobs(np.random.default_rng(6), groups=1, per=6)
    conn = _library(tmp_path, vecs)
    result = cluster.cluster_tracks(conn)
    assert result["n_clusters"] == 1 and result["sizes"] == [6]


def test_new_tracks_join_the_nearest_existing_cluster(tmp_path):
    rng = np.random.default_rng(7)
    vecs, truth, centres = _blobs(rng)
    conn = _library(tmp_path, vecs)
    first = cluster.cluster_tracks(conn, n_clusters=3)
    before = _assignments(conn, first["run_id"])
    for j, g in enumerate((2, 0), start=37):            # two new songs
        _add(conn, j, centres[g] + 0.15 * rng.normal(size=32))
        truth.append(g)
    second = cluster.cluster_tracks(conn, n_clusters=3)
    assert second["run_id"] == first["run_id"]           # incremental, no reshuffle
    after = _assignments(conn, second["run_id"])
    assert {k: after[k] for k in before} == before
    assert _agreement(after, truth) == pytest.approx(1.0)


def test_rebuild_keeps_cluster_ids_stable(tmp_path):
    rng = np.random.default_rng(8)
    vecs, truth, centres = _blobs(rng)
    conn = _library(tmp_path, vecs)
    first = cluster.cluster_tracks(conn, n_clusters=3)
    ids_before = _assignments(conn, first["run_id"])
    for j in range(37, 43):
        _add(conn, j, centres[j % 3] + 0.15 * rng.normal(size=32))
    second = cluster.rebuild_clusters(conn, n_clusters=3)
    assert second["run_id"] != first["run_id"]
    ids_after = _assignments(conn, second["run_id"])
    assert {k: ids_after[k] for k in ids_before} == ids_before


def test_a_different_cluster_count_starts_a_new_run(tmp_path):
    vecs, _, _ = _blobs(np.random.default_rng(9), groups=4, per=10)
    conn = _library(tmp_path, vecs)
    a = cluster.cluster_tracks(conn, n_clusters=4)
    b = cluster.cluster_tracks(conn, n_clusters=2)
    assert b["run_id"] != a["run_id"] and b["n_clusters"] == 2


def test_legacy_features_are_used_when_there_are_no_embeddings(tmp_path):
    from tests.test_cluster import _seed_features
    conn = store.connect(tmp_path / "db.sqlite")
    _seed_features(conn, n=30)
    result = cluster.cluster_tracks(conn, n_clusters=3)
    assert result.get("method", "kmeans") == "kmeans"


def test_method_can_be_forced(tmp_path):
    vecs, _, _ = _blobs(np.random.default_rng(10))
    conn = _library(tmp_path, vecs)
    with pytest.raises(ValueError):
        cluster.cluster_tracks(conn, n_clusters=3, method="nonsense")


def test_cli_cluster_reports_the_method(tmp_path):
    from typer.testing import CliRunner
    from riffle.cli import app
    vecs, _, _ = _blobs(np.random.default_rng(11))
    conn = _library(tmp_path, vecs)
    conn.close()
    r = CliRunner().invoke(app, ["--db", str(tmp_path / "db.sqlite"), "cluster",
                                 "--n", "3", "--as-json"])
    assert r.exit_code == 0, r.stdout
    data = json.loads(r.stdout)
    assert data["method"] == "embedding" and data["n_clusters"] == 3


def test_neighbour_graph_is_symmetric_with_bounded_weights():
    rng = np.random.default_rng(12)
    D = similarity.mutual_proximity(similarity.cosine_distance_matrix(rng.normal(size=(30, 16))))
    A = similarity.knn_affinity(D, k=5)
    assert np.allclose(A, A.T) and np.allclose(np.diag(A), 0)
    assert A.min() >= 0 and A.max() <= 1
    assert (A > 0).sum(axis=1).min() >= 5        # every node keeps its 5 neighbours


def test_graph_neighbours_grow_with_the_library_up_to_a_cap():
    # 10 neighbours suited the 416-track benchmark, but on ~1,800 tracks it left
    # tiny groups (3 tracks) breaking off as clusters; 20 gave balanced
    # clusters, higher purity (0.80 vs 0.78 at k=8) and better stability at k=12.
    assert cluster._graph_neighbours(36) == 10
    assert cluster._graph_neighbours(416) == 10
    assert cluster._graph_neighbours(1000) == 20
    assert cluster._graph_neighbours(1839) == 20
    assert cluster._graph_neighbours(50_000) == 20


def test_run_records_the_neighbours_it_used(tmp_path):
    vecs, _, _ = _blobs(np.random.default_rng(13))
    conn = _library(tmp_path, vecs)
    run_id = cluster.cluster_tracks(conn, n_clusters=3)["run_id"]
    meta = json.loads(conn.execute("SELECT scaler_params FROM cluster_run WHERE id = ?",
                                   (run_id,)).fetchone()["scaler_params"])
    assert meta["knn"] == cluster._graph_neighbours(36)
