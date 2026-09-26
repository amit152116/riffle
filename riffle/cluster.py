"""K-means clustering on audio feature vectors."""
from __future__ import annotations

import pickle
from datetime import UTC, datetime

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from riffle import store

FEATURE_COLUMNS = ["bpm", "energy", "danceability", "loudness_lufs",
                   "spectral_centroid", "onset_rate"]


def load_feature_matrix(conn) -> tuple[np.ndarray, list[int]]:
    rows = conn.execute(
        "SELECT audio_content_id, bpm, energy, danceability, loudness_lufs, "
        "spectral_centroid, onset_rate, mfcc_mean "
        "FROM audio_features "
        "WHERE bpm IS NOT NULL AND energy IS NOT NULL AND mfcc_mean IS NOT NULL"
    ).fetchall()

    content_ids = []
    vectors = []
    for r in rows:
        scalar_feats = [r["bpm"], r["energy"], r["danceability"],
                        r["loudness_lufs"], r["spectral_centroid"], r["onset_rate"]]
        mfcc = store.unpack_mfcc(r["mfcc_mean"])
        vectors.append(scalar_feats + list(mfcc))
        content_ids.append(r["audio_content_id"])

    if not vectors:
        return np.empty((0, 19)), []
    return np.array(vectors), content_ids


def _latest_run(conn):
    return conn.execute(
        "SELECT id, n_clusters FROM cluster_run ORDER BY id DESC LIMIT 1"
    ).fetchone()


def cluster_tracks(conn, n_clusters: int | None = None) -> dict:
    matrix, content_ids = load_feature_matrix(conn)
    n_tracks = len(content_ids)

    if n_tracks == 0:
        return {"run_id": None, "n_clusters": 0, "sizes": [], "labels": []}

    if n_tracks < 15:
        run_id = _create_single_cluster(conn, matrix, content_ids)
        return {"run_id": run_id, "n_clusters": 1, "sizes": [n_tracks], "labels": ["All"]}

    latest = _latest_run(conn)
    needs_full_cluster = (
        latest is None
        or latest["n_clusters"] == 1
        or (n_clusters is not None and n_clusters != latest["n_clusters"])
    )
    if not needs_full_cluster:
        return _assign_new_to_existing(conn, matrix, content_ids, latest)

    return _full_cluster(conn, matrix, content_ids, n_clusters)


def _create_single_cluster(conn, matrix, content_ids) -> int:
    now = datetime.now(UTC).isoformat()
    scaler = StandardScaler()
    scaler.fit(matrix)

    conn.execute(
        "INSERT INTO cluster_run (n_clusters, n_tracks, scaler_params, created_at) "
        "VALUES (?, ?, ?, ?)", (1, len(content_ids), pickle.dumps(scaler), now))
    run_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    centroid = np.mean(matrix, axis=0)
    conn.execute(
        "INSERT INTO cluster_centroid (run_id, cluster_id, label, centroid, n_tracks) "
        "VALUES (?, ?, ?, ?, ?)",
        (run_id, 0, "All", store.pack_mfcc(centroid), len(content_ids)))

    for cid in content_ids:
        conn.execute(
            "INSERT INTO cluster_assignment (run_id, audio_content_id, cluster_id, "
            "distance_to_centroid) VALUES (?, ?, 0, 0.0)", (run_id, cid))

    return run_id


def _full_cluster(conn, matrix, content_ids, n_clusters=None) -> dict:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)

    if n_clusters is None:
        best_k, best_score = 5, -1
        for k in range(5, min(16, len(content_ids))):
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(scaled)
            try:
                score = silhouette_score(scaled, labels)
            except ValueError:
                continue
            if score > best_score:
                best_k, best_score = k, score
            elif score == best_score and k < best_k:
                best_k = k
        n_clusters = best_k

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(scaled)

    now = datetime.now(UTC).isoformat()
    conn.execute(
        "INSERT INTO cluster_run (n_clusters, n_tracks, scaler_params, created_at) "
        "VALUES (?, ?, ?, ?)", (n_clusters, len(content_ids), pickle.dumps(scaler), now))
    run_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    for cid in range(n_clusters):
        mask = labels == cid
        centroid = km.cluster_centers_[cid]
        n_in = int(mask.sum())
        conn.execute(
            "INSERT INTO cluster_centroid (run_id, cluster_id, label, centroid, n_tracks) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_id, cid, None, centroid.tobytes(), n_in))

    for i, content_id in enumerate(content_ids):
        dist = float(np.linalg.norm(scaled[i] - km.cluster_centers_[labels[i]]))
        conn.execute(
            "INSERT INTO cluster_assignment (run_id, audio_content_id, cluster_id, "
            "distance_to_centroid) VALUES (?, ?, ?, ?)",
            (run_id, content_id, int(labels[i]), dist))

    sizes = [int((labels == c).sum()) for c in range(n_clusters)]
    return {"run_id": run_id, "n_clusters": n_clusters, "sizes": sizes, "labels": []}


def _assign_new_to_existing(conn, matrix, content_ids, latest_run) -> dict:
    run_id = latest_run["id"]
    n_clusters = latest_run["n_clusters"]

    assigned_ids = {r[0] for r in conn.execute(
        "SELECT audio_content_id FROM cluster_assignment WHERE run_id = ?", (run_id,)
    ).fetchall()}

    new_mask = [cid not in assigned_ids for cid in content_ids]
    new_indices = [i for i, is_new in enumerate(new_mask) if is_new]

    if not new_indices:
        sizes = []
        for cid in range(n_clusters):
            n = conn.execute(
                "SELECT count(*) c FROM cluster_assignment "
                "WHERE run_id = ? AND cluster_id = ?", (run_id, cid)
            ).fetchone()["c"]
            sizes.append(n)
        return {"run_id": run_id, "n_clusters": n_clusters, "sizes": sizes, "labels": []}

    scaler_blob = conn.execute(
        "SELECT scaler_params FROM cluster_run WHERE id = ?", (run_id,)
    ).fetchone()["scaler_params"]
    scaler = pickle.loads(scaler_blob)

    centroids_rows = conn.execute(
        "SELECT cluster_id, centroid FROM cluster_centroid WHERE run_id = ? "
        "ORDER BY cluster_id", (run_id,)
    ).fetchall()
    centroids = np.array([np.frombuffer(r["centroid"], dtype=np.float64)
                          for r in centroids_rows])

    for idx in new_indices:
        vec = scaler.transform(matrix[idx:idx+1])
        dists = np.linalg.norm(vec - centroids, axis=1)
        closest = int(np.argmin(dists))
        conn.execute(
            "INSERT INTO cluster_assignment (run_id, audio_content_id, cluster_id, "
            "distance_to_centroid) VALUES (?, ?, ?, ?)",
            (run_id, content_ids[idx], closest, float(dists[closest])))

    sizes = []
    for cid in range(n_clusters):
        n = conn.execute(
            "SELECT count(*) c FROM cluster_assignment "
            "WHERE run_id = ? AND cluster_id = ?", (run_id, cid)
        ).fetchone()["c"]
        sizes.append(n)

    return {"run_id": run_id, "n_clusters": n_clusters, "sizes": sizes, "labels": []}


from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def rebuild_clusters(conn, n_clusters: int | None = None) -> dict:
    matrix, content_ids = load_feature_matrix(conn)
    if len(content_ids) < 15:
        return cluster_tracks(conn, n_clusters)

    conn.execute("BEGIN")
    try:
        result = _rebuild_clusters_txn(conn, matrix, content_ids, n_clusters)
    except Exception:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
    return result


def _rebuild_clusters_txn(conn, matrix, content_ids, n_clusters) -> dict:
    old_run = _latest_run(conn)
    old_centroids = None
    if old_run is not None:
        rows = conn.execute(
            "SELECT cluster_id, centroid FROM cluster_centroid WHERE run_id = ? "
            "ORDER BY cluster_id", (old_run["id"],)
        ).fetchall()
        if rows:
            old_centroids = np.array([np.frombuffer(r["centroid"], dtype=np.float64)
                                      for r in rows])

    result = _full_cluster(conn, matrix, content_ids, n_clusters)
    new_run_id = result["run_id"]

    if old_centroids is not None and result["n_clusters"] == len(old_centroids):
        new_rows = conn.execute(
            "SELECT cluster_id, centroid FROM cluster_centroid WHERE run_id = ? "
            "ORDER BY cluster_id", (new_run_id,)
        ).fetchall()
        new_centroids = np.array([np.frombuffer(r["centroid"], dtype=np.float64)
                                  for r in new_rows])

        cost = cdist(old_centroids, new_centroids)
        old_idx, new_idx = linear_sum_assignment(cost)
        mapping = {}
        for o, n in zip(old_idx, new_idx):
            mapping[int(n)] = int(o)
        next_id = max(mapping.values(), default=-1) + 1
        for n in range(result["n_clusters"]):
            if n not in mapping:
                mapping[n] = next_id
                next_id += 1

        for old_cid, new_cid in mapping.items():
            if old_cid != new_cid:
                conn.execute(
                    "UPDATE cluster_centroid SET cluster_id = ? "
                    "WHERE run_id = ? AND cluster_id = ?",
                    (-1000 - old_cid, new_run_id, old_cid))
        for old_cid, new_cid in mapping.items():
            conn.execute(
                "UPDATE cluster_centroid SET cluster_id = ? "
                "WHERE run_id = ? AND cluster_id = ?",
                (new_cid, new_run_id, -1000 - old_cid if old_cid != new_cid else old_cid))

        for old_cid, new_cid in mapping.items():
            if old_cid != new_cid:
                conn.execute(
                    "UPDATE cluster_assignment SET cluster_id = ? "
                    "WHERE run_id = ? AND cluster_id = ?",
                    (-1000 - old_cid, new_run_id, old_cid))
        for old_cid, new_cid in mapping.items():
            conn.execute(
                "UPDATE cluster_assignment SET cluster_id = ? "
                "WHERE run_id = ? AND cluster_id = ?",
                (new_cid, new_run_id, -1000 - old_cid if old_cid != new_cid else old_cid))

    for cid in range(result["n_clusters"]):
        lbl = label_cluster(conn, cid, new_run_id)
        conn.execute(
            "UPDATE cluster_centroid SET label = ? "
            "WHERE run_id = ? AND cluster_id = ?", (lbl, new_run_id, cid))

    return result


def label_cluster(conn, cluster_id: int, run_id: int) -> str:
    rows = conn.execute(
        "SELECT af.bpm, af.energy, af.scale, af.spectral_centroid "
        "FROM cluster_assignment ca "
        "JOIN audio_features af ON af.audio_content_id = ca.audio_content_id "
        "WHERE ca.run_id = ? AND ca.cluster_id = ?", (run_id, cluster_id)
    ).fetchall()

    if not rows:
        return "Empty"

    bpms = [r["bpm"] for r in rows if r["bpm"] is not None]
    energies = [r["energy"] for r in rows if r["energy"] is not None]
    scales = [r["scale"] for r in rows if r["scale"] is not None]
    centroids = [r["spectral_centroid"] for r in rows if r["spectral_centroid"] is not None]

    parts = []
    if bpms:
        med_bpm = sorted(bpms)[len(bpms) // 2]
        parts.append("Slow" if med_bpm < 80 else "Upbeat" if med_bpm > 120 else "Mid-tempo")
    if energies:
        med_e = sorted(energies)[len(energies) // 2]
        parts.append("Calm" if med_e < 0.3 else "Energetic" if med_e > 0.7 else "Moderate")
    if scales:
        major_count = sum(1 for s in scales if s == "major")
        parts.append("Major" if major_count > len(scales) / 2 else "Minor")
    if centroids:
        all_sc = conn.execute(
            "SELECT spectral_centroid FROM audio_features WHERE spectral_centroid IS NOT NULL"
        ).fetchall()
        global_median = sorted(r[0] for r in all_sc)[len(all_sc) // 2]
        med_sc = sorted(centroids)[len(centroids) // 2]
        parts.append("Bright" if med_sc > global_median else "Warm")

    return " ".join(parts) if parts else "Mixed"


def render_clusters(conn) -> str:
    latest = _latest_run(conn)
    if latest is None:
        return "No clustering runs found. Run 'riffle cluster' first."

    run_id = latest["id"]
    lines = ["Cluster Summary", "=" * 40, ""]

    centroids = conn.execute(
        "SELECT cluster_id, label, n_tracks FROM cluster_centroid "
        "WHERE run_id = ? ORDER BY cluster_id", (run_id,)
    ).fetchall()

    for c in centroids:
        label = c["label"] or label_cluster(conn, c["cluster_id"], run_id)
        lines.append(f"Cluster {c['cluster_id']}: {label} ({c['n_tracks']} tracks)")
        examples = conn.execute(
            "SELECT t.tag_artist, t.tag_title FROM cluster_assignment ca "
            "JOIN track t ON t.audio_content_id = ca.audio_content_id "
            "WHERE ca.run_id = ? AND ca.cluster_id = ? LIMIT 3",
            (run_id, c["cluster_id"])
        ).fetchall()
        for ex in examples:
            lines.append(f"  - {ex['tag_artist'] or '?'} — {ex['tag_title'] or '?'}")
        lines.append("")

    return "\n".join(lines)
