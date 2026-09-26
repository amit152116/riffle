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


def score_components(features_a: dict, features_b: dict,
                     weights: dict | None = None) -> dict:
    """Per-component normalized distances plus the combined score.

    A component whose inputs are missing (BPM is NULL/<=0 -- Essentia writes
    0.0 for beatless/silent tracks, not NULL; key is NULL) is OMITTED rather
    than defaulted to a literal 0 (which would read as a perfect BPM match)
    or a fabricated key like "C major". The remaining weights are
    renormalized so the score stays comparable across tracks with different
    amounts of missing data. MFCC and energy are required by the caller's
    query (mfcc_mean IS NOT NULL) and always contribute.
    """
    w = weights or DEFAULT_WEIGHTS

    mfcc_dist = mfcc_cosine_distance(features_a["mfcc_mean"], features_b["mfcc_mean"])
    mfcc_norm = min(mfcc_dist / 2.0, 1.0)

    bpm_a, bpm_b = features_a.get("bpm"), features_b.get("bpm")
    if _is_missing_bpm(bpm_a) or _is_missing_bpm(bpm_b):
        bpm_norm = None
    else:
        bpm_norm = min(abs(bpm_a - bpm_b) / 60.0, 1.0)

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
                   weights: dict | None = None) -> float:
    return score_components(features_a, features_b, weights)["combined_score"]


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
            comp = score_components(fa, all_features[other_id])
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
            comp = score_components(fe, all_features[new_id])
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
