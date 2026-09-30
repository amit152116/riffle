"""Similarity metrics: MFCC cosine distance, key distance, normalization."""
from __future__ import annotations

import hashlib
import json

import numpy as np

CIRCLE_OF_FIFTHS = ["C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F"]
_KEY_ALIASES = {"C#": "Db", "D#": "Eb", "G#": "Ab", "A#": "Bb"}
_RELATIVE_MINOR = {"C": "A", "G": "E", "D": "B", "A": "F#", "E": "Db",
                   "B": "Ab", "F#": "Eb", "Db": "Bb", "Ab": "F",
                   "Eb": "C", "Bb": "G", "F": "D"}

DEFAULT_WEIGHTS = {"mfcc": 0.5, "bpm": 0.25, "key": 0.15, "energy": 0.1}


def similarity_config_hash() -> str:
    blob = json.dumps(DEFAULT_WEIGHTS, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def mfcc_cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    cos_sim = np.dot(a, b) / (norm_a * norm_b)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return float(1.0 - cos_sim)


def _normalize_key(key: str) -> str:
    return _KEY_ALIASES.get(key, key)


def key_distance(key_a: str, scale_a: str, key_b: str, scale_b: str) -> int:
    ka = _normalize_key(key_a)
    kb = _normalize_key(key_b)

    if scale_a == "minor":
        ka = {v: k for k, v in _RELATIVE_MINOR.items()}.get(ka, ka)
    if scale_b == "minor":
        kb = {v: k for k, v in _RELATIVE_MINOR.items()}.get(kb, kb)

    if ka not in CIRCLE_OF_FIFTHS or kb not in CIRCLE_OF_FIFTHS:
        return 6

    ia = CIRCLE_OF_FIFTHS.index(ka)
    ib = CIRCLE_OF_FIFTHS.index(kb)
    diff = abs(ia - ib)
    return min(diff, 12 - diff)


def normalize_components(mfcc_dist: float, bpm_diff: float,
                         key_dist: int, energy_diff: float) -> dict:
    return {
        "mfcc": min(mfcc_dist / 2.0, 1.0),
        "bpm": min(abs(bpm_diff) / 60.0, 1.0),
        "key": min(key_dist / 6.0, 1.0),
        "energy": min(abs(energy_diff), 1.0),
    }


def _is_missing_bpm(bpm) -> bool:
    return bpm is None or bpm <= 0


def _is_missing_key(key_name) -> bool:
    return not key_name


def _is_missing_energy(energy) -> bool:
    return energy is None


DEFAULT_BPM_SCALE = 60.0


def score_components(features_a: dict, features_b: dict,
                     weights: dict | None = None,
                     bpm_scale: float = DEFAULT_BPM_SCALE) -> dict:
    """Per-component normalized distances plus the combined score.

    A component whose inputs are missing (BPM is NULL/<=0 -- Essentia writes
    0.0 for beatless/silent tracks, not NULL; key is NULL) is OMITTED rather
    than defaulted to a literal 0 (which would read as a perfect BPM match)
    or a fabricated key like "C major". The remaining weights are
    renormalized so the score stays comparable across tracks with different
    amounts of missing data. MFCC and energy are required by the caller's
    query (mfcc_mean IS NOT NULL) and always contribute.

    `bpm_scale` is the BPM difference that counts as "maximally different"
    -- defaults to a fixed 60.0, but callers indexing a real library should
    pass `compute_bpm_spread(conn)` instead, since what counts as a big BPM
    difference depends on the library's own tempo range.
    """
    w = weights or DEFAULT_WEIGHTS

    mfcc_dist = mfcc_cosine_distance(features_a["mfcc_mean"], features_b["mfcc_mean"])
    mfcc_norm = min(mfcc_dist / 2.0, 1.0)

    bpm_a, bpm_b = features_a.get("bpm"), features_b.get("bpm")
    if _is_missing_bpm(bpm_a) or _is_missing_bpm(bpm_b):
        bpm_norm = None
    else:
        bpm_norm = min(abs(bpm_a - bpm_b) / bpm_scale, 1.0)

    key_a, key_b = features_a.get("key_name"), features_b.get("key_name")
    if _is_missing_key(key_a) or _is_missing_key(key_b):
        key_norm = None
    else:
        kd = key_distance(key_a, features_a.get("scale") or "major",
                          key_b, features_b.get("scale") or "major")
        key_norm = min(kd / 6.0, 1.0)

    energy_a, energy_b = features_a.get("energy"), features_b.get("energy")
    if _is_missing_energy(energy_a) or _is_missing_energy(energy_b):
        energy_norm = None
    else:
        energy_norm = min(abs(energy_a - energy_b), 1.0)

    parts = [("mfcc", mfcc_norm), ("bpm", bpm_norm), ("key", key_norm), ("energy", energy_norm)]
    available = [(name, val) for name, val in parts if val is not None]
    weight_sum = sum(w[name] for name, _ in available) or 1.0
    score = sum(w[name] * val for name, val in available) / weight_sum

    return {"mfcc_norm": mfcc_norm, "bpm_norm": bpm_norm, "key_norm": key_norm,
            "energy_norm": energy_norm, "combined_score": score}


def combined_score(features_a: dict, features_b: dict,
                   weights: dict | None = None,
                   bpm_scale: float = DEFAULT_BPM_SCALE) -> float:
    return score_components(features_a, features_b, weights, bpm_scale)["combined_score"]


def compute_bpm_spread(conn) -> float:
    """The library's own typical BPM variation (interquartile range across
    present tracks with a valid BPM), used in place of a fixed assumption
    about what counts as a 'big' tempo difference. Falls back to
    DEFAULT_BPM_SCALE when there isn't enough data for a stable IQR, and
    floors the result so a library with almost no tempo variation doesn't
    make tiny BPM differences read as maximally different.
    """
    bpms = sorted(
        r[0] for r in conn.execute(
            "SELECT bpm FROM audio_features WHERE bpm IS NOT NULL AND bpm > 0"
        ).fetchall()
    )
    if len(bpms) < 4:
        return DEFAULT_BPM_SCALE
    q25, q75 = np.percentile(bpms, [25, 75])
    iqr = float(q75 - q25)
    return max(iqr, 10.0)


from riffle import store


def _load_all_features(conn) -> dict[int, dict]:
    rows = conn.execute(
        "SELECT t.id AS track_id, af.bpm, af.key_name, af.scale, af.energy, af.mfcc_mean "
        "FROM track t "
        "JOIN audio_features af ON af.audio_content_id = t.audio_content_id "
        "WHERE t.present = 1 AND af.mfcc_mean IS NOT NULL"
    ).fetchall()
    features = {}
    for r in rows:
        features[r["track_id"]] = {
            "bpm": r["bpm"],
            "key_name": r["key_name"],
            "scale": r["scale"],
            "energy": r["energy"],
            "mfcc_mean": store.unpack_mfcc(r["mfcc_mean"]),
        }
    return features


def _insert_similarity_row(conn, track_id: int, neighbor_id: int, comp: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO track_similarity "
        "(track_id, neighbor_id, mfcc_norm, bpm_norm, key_norm, "
        "energy_norm, combined_score, config_hash) VALUES (?,?,?,?,?,?,?,?)",
        (track_id, neighbor_id, comp["mfcc_norm"], comp["bpm_norm"], comp["key_norm"],
         comp["energy_norm"], comp["combined_score"], similarity_config_hash()))


def build_similarity(conn, top_k: int = 20) -> dict:
    all_features = _load_all_features(conn)
    all_ids = sorted(all_features.keys())

    indexed_ids = {r[0] for r in conn.execute(
        "SELECT DISTINCT track_id FROM track_similarity"
    ).fetchall()}

    new_ids = [tid for tid in all_ids if tid not in indexed_ids]
    if not new_ids:
        return {"new_tracks": 0, "pairs_stored": 0, "existing_updated": 0}

    bpm_scale = compute_bpm_spread(conn)
    pairs_stored = 0
    existing_updated = 0

    # Each new track gets its own directed top-K against every other track
    # (including other new ones), independent of any other track's rows.
    for new_id in new_ids:
        fa = all_features[new_id]
        scored = []
        for other_id in all_ids:
            if other_id == new_id:
                continue
            comp = score_components(fa, all_features[other_id], bpm_scale=bpm_scale)
            scored.append((other_id, comp))
        scored.sort(key=lambda x: x[1]["combined_score"])
        for other_id, comp in scored[:top_k]:
            _insert_similarity_row(conn, new_id, other_id, comp)
            pairs_stored += 1

    # Existing tracks: a new track may belong in their top-K. Fill up to
    # top_k unconditionally if they have room; otherwise only displace the
    # current worst neighbor if the new one scores better, evicting it so
    # the row count never exceeds top_k. Directed storage means this only
    # ever touches existing_id's own rows.
    for existing_id in indexed_ids:
        if existing_id not in all_features or existing_id in new_ids:
            continue
        fe = all_features[existing_id]

        current = conn.execute(
            "SELECT neighbor_id, combined_score FROM track_similarity "
            "WHERE track_id = ? ORDER BY combined_score", (existing_id,)
        ).fetchall()
        current_ids = {r["neighbor_id"] for r in current}
        count = len(current)
        worst_score = current[-1]["combined_score"] if current else float("inf")

        for new_id in new_ids:
            if new_id in current_ids:
                continue
            comp = score_components(fe, all_features[new_id], bpm_scale=bpm_scale)
            score = comp["combined_score"]

            if count < top_k:
                _insert_similarity_row(conn, existing_id, new_id, comp)
                count += 1
                current_ids.add(new_id)
                existing_updated += 1
                worst_score = max(worst_score, score) if worst_score != float("inf") else score
            elif score < worst_score:
                conn.execute(
                    "DELETE FROM track_similarity WHERE track_id = ? AND neighbor_id = ("
                    "SELECT neighbor_id FROM track_similarity WHERE track_id = ? "
                    "ORDER BY combined_score DESC LIMIT 1)", (existing_id, existing_id))
                _insert_similarity_row(conn, existing_id, new_id, comp)
                current_ids.add(new_id)
                existing_updated += 1
                worst_row = conn.execute(
                    "SELECT combined_score FROM track_similarity WHERE track_id = ? "
                    "ORDER BY combined_score DESC LIMIT 1", (existing_id,)
                ).fetchone()
                worst_score = worst_row["combined_score"] if worst_row else float("inf")

    return {"new_tracks": len(new_ids), "pairs_stored": pairs_stored,
            "existing_updated": existing_updated}


def full_rebuild(conn, top_k: int = 20) -> dict:
    conn.execute("BEGIN")
    try:
        conn.execute("DELETE FROM track_similarity")
        result = build_similarity(conn, top_k)
    except Exception:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
    return result


# --- Embedding-based index --------------------------------------------------
#
# Benchmarked on 416 real tracks against the MFCC-based score above. The
# Discogs-EffNet embedding found a known duplicate at rank 1 for 99% of
# tracks (MFCC score: 85%) and put same-style tracks in 81% of the top 10
# (MFCC score: 56%, chance 50%). Blending tempo/key/energy into the embedding
# distance made it worse, so those stay out of the similarity score and are
# used as transition rules when building a playlist.

EMBEDDING_MODEL = "discogs-effnet-bs64-1"


def embedding_config_hash() -> str:
    blob = json.dumps({"method": "embedding", "model": EMBEDDING_MODEL,
                       "distance": "cosine",
                       "correction": "mutual-proximity-gaussian"},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def cosine_distance_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    D = np.clip(1.0 - X @ X.T, 0.0, 2.0)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return D


def mutual_proximity(D: np.ndarray) -> np.ndarray:
    """Gaussian mutual proximity (Schnitzer et al., JMLR 2012).

    In high dimensions some tracks ("hubs") sit among the nearest neighbours
    of a large share of the library while others ("anti-hubs") appear in
    nobody's list, so a shuffle over raw distances keeps returning the same
    tracks and never reaches others. Mutual proximity re-expresses each
    distance as the chance that BOTH tracks see the other as unusually
    close, judged against each one's own distance distribution. Returned as
    a distance in [0, 1].
    """
    from scipy.special import ndtr

    n = len(D)
    if n < 3:
        return D / max(float(D.max()), 1e-12)
    off = ~np.eye(n, dtype=bool)
    mu = (D * off).sum(axis=1) / (n - 1)
    sd = np.sqrt((((D - mu[:, None]) ** 2) * off).sum(axis=1) / (n - 1)) + 1e-12
    p_farther = 1.0 - ndtr((D - mu[:, None]) / sd[:, None])   # P(X_i > d_ij)
    out = np.clip(1.0 - p_farther * p_farther.T, 0.0, 1.0)
    np.fill_diagonal(out, 0.0)
    return out


def load_embeddings(conn) -> tuple[list[int], np.ndarray]:
    rows = conn.execute(
        "SELECT t.id AS track_id, e.dim AS dim, e.vector AS vector "
        "FROM track t JOIN audio_embedding e "
        "  ON e.audio_content_id = t.audio_content_id AND e.model = ? "
        "WHERE t.present = 1 ORDER BY t.id", (EMBEDDING_MODEL,)).fetchall()
    ids = [r["track_id"] for r in rows]
    if not rows:
        return ids, np.zeros((0, 0))
    return ids, np.stack([np.frombuffer(r["vector"], dtype="<f4") for r in rows])


EMBEDDING_TOP_K = 100   # shuffle re-ranks a pool of ~80 neighbours


def build_embedding_similarity(conn, top_k: int = EMBEDDING_TOP_K) -> dict:
    """(Re)build every track's top-k neighbours from the embeddings.

    Always a full rebuild: mutual proximity depends on the whole library, so
    adding one track can change other tracks' distances. It is cheap (a
    matrix over the library) and replaces any index built another way.
    """
    ids, X = load_embeddings(conn)
    if not ids:
        raise ValueError("No embeddings found: run `riffle embed` first")
    raw = cosine_distance_matrix(X)
    D = mutual_proximity(raw)
    k = min(top_k, len(ids) - 1)
    config_hash = embedding_config_hash()

    rows = []
    for i, track_id in enumerate(ids):
        order = np.argsort(D[i], kind="stable")
        order = order[order != i][:k]
        for j in order:
            # mfcc_norm is NOT NULL and holds the primary audio distance
            # (here the raw cosine distance); the other components are unused.
            rows.append((track_id, ids[j], float(raw[i, j] / 2.0), None, None,
                         None, float(D[i, j]), config_hash))

    conn.execute("BEGIN")
    try:
        conn.execute("DELETE FROM track_similarity")
        conn.executemany(
            "INSERT INTO track_similarity (track_id, neighbor_id, mfcc_norm, "
            "bpm_norm, key_norm, energy_norm, combined_score, config_hash) "
            "VALUES (?,?,?,?,?,?,?,?)", rows)
    except Exception:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
    return {"tracks": len(ids), "pairs_stored": len(rows)}


def build_index(conn, method: str = "auto", top_k: int | None = None,
                rebuild: bool = False) -> dict:
    """Build the neighbour index with the embedding or the legacy MFCC score.

    "auto" uses embeddings when any exist, else the MFCC-based score.
    """
    has_embeddings = conn.execute(
        "SELECT count(*) c FROM audio_embedding WHERE model = ?",
        (EMBEDDING_MODEL,)).fetchone()["c"] > 0
    if method == "auto":
        method = "embedding" if has_embeddings else "mfcc"
    if method == "embedding":
        if not has_embeddings:
            raise ValueError("No embeddings found: run `riffle embed` first")
        result = build_embedding_similarity(conn, top_k or EMBEDDING_TOP_K)
    elif method == "mfcc":
        k = top_k or 20
        result = full_rebuild(conn, k) if rebuild else build_similarity(conn, k)
    else:
        raise ValueError(f"unknown method {method!r} (auto, embedding, mfcc)")
    result["method"] = method
    return result
