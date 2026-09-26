"""Similarity metrics: MFCC cosine distance, key distance, normalization."""
from __future__ import annotations

import numpy as np

CIRCLE_OF_FIFTHS = ["C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F"]
_KEY_ALIASES = {"C#": "Db", "D#": "Eb", "G#": "Ab", "A#": "Bb"}
_RELATIVE_MINOR = {"C": "A", "G": "E", "D": "B", "A": "F#", "E": "Db",
                   "B": "Ab", "F#": "Eb", "Db": "Bb", "Ab": "F",
                   "Eb": "C", "Bb": "G", "F": "D"}

DEFAULT_WEIGHTS = {"mfcc": 0.5, "bpm": 0.25, "key": 0.15, "energy": 0.1}


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


def combined_score(features_a: dict, features_b: dict,
                   weights: dict | None = None) -> float:
    w = weights or DEFAULT_WEIGHTS
    mfcc_dist = mfcc_cosine_distance(features_a["mfcc_mean"], features_b["mfcc_mean"])
    bpm_diff = abs((features_a.get("bpm") or 0) - (features_b.get("bpm") or 0))
    kd = key_distance(
        features_a.get("key_name") or "C", features_a.get("scale") or "major",
        features_b.get("key_name") or "C", features_b.get("scale") or "major")
    energy_diff = abs((features_a.get("energy") or 0) - (features_b.get("energy") or 0))

    norm = normalize_components(mfcc_dist, bpm_diff, kd, energy_diff)
    return (w["mfcc"] * norm["mfcc"] + w["bpm"] * norm["bpm"] +
            w["key"] * norm["key"] + w["energy"] * norm["energy"])


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


def build_similarity(conn, top_k: int = 20) -> dict:
    all_features = _load_all_features(conn)
    all_ids = sorted(all_features.keys())

    indexed_ids = set()
    for r in conn.execute(
        "SELECT DISTINCT track_a_id FROM track_similarity"
    ).fetchall():
        indexed_ids.add(r[0])
    for r in conn.execute(
        "SELECT DISTINCT track_b_id FROM track_similarity"
    ).fetchall():
        indexed_ids.add(r[0])

    new_ids = [tid for tid in all_ids if tid not in indexed_ids]
    if not new_ids:
        return {"new_tracks": 0, "pairs_stored": 0, "existing_updated": 0}

    pairs_stored = 0
    existing_updated = 0

    for new_id in new_ids:
        scores = []
        fa = all_features[new_id]
        for other_id in all_ids:
            if other_id == new_id:
                continue
            fb = all_features[other_id]
            score = combined_score(fa, fb)
            norm = normalize_components(
                mfcc_cosine_distance(fa["mfcc_mean"], fb["mfcc_mean"]),
                abs((fa.get("bpm") or 0) - (fb.get("bpm") or 0)),
                key_distance(fa.get("key_name") or "C", fa.get("scale") or "major",
                             fb.get("key_name") or "C", fb.get("scale") or "major"),
                abs((fa.get("energy") or 0) - (fb.get("energy") or 0)),
            )
            scores.append((other_id, score, norm))

        scores.sort(key=lambda x: x[1])
        for other_id, score, norm in scores[:top_k]:
            a, b = min(new_id, other_id), max(new_id, other_id)
            conn.execute(
                "INSERT OR REPLACE INTO track_similarity "
                "(track_a_id, track_b_id, mfcc_norm, bpm_norm, key_norm, "
                "energy_norm, combined_score) VALUES (?,?,?,?,?,?,?)",
                (a, b, norm["mfcc"], norm["bpm"], norm["key"], norm["energy"], score))
            pairs_stored += 1

    for existing_id in indexed_ids:
        if existing_id not in all_features:
            continue
        worst = conn.execute(
            "SELECT combined_score FROM track_similarity "
            "WHERE track_a_id = ? OR track_b_id = ? "
            "ORDER BY combined_score DESC LIMIT 1",
            (existing_id, existing_id)
        ).fetchone()
        if worst is None:
            continue
        worst_score = worst["combined_score"]

        fe = all_features[existing_id]
        for new_id in new_ids:
            fn = all_features[new_id]
            score = combined_score(fe, fn)
            if score < worst_score:
                a, b = min(existing_id, new_id), max(existing_id, new_id)
                existing_check = conn.execute(
                    "SELECT 1 FROM track_similarity WHERE track_a_id = ? AND track_b_id = ?",
                    (a, b)
                ).fetchone()
                if existing_check is None:
                    norm = normalize_components(
                        mfcc_cosine_distance(fe["mfcc_mean"], fn["mfcc_mean"]),
                        abs((fe.get("bpm") or 0) - (fn.get("bpm") or 0)),
                        key_distance(fe.get("key_name") or "C", fe.get("scale") or "major",
                                     fn.get("key_name") or "C", fn.get("scale") or "major"),
                        abs((fe.get("energy") or 0) - (fn.get("energy") or 0)),
                    )
                    conn.execute(
                        "INSERT INTO track_similarity "
                        "(track_a_id, track_b_id, mfcc_norm, bpm_norm, key_norm, "
                        "energy_norm, combined_score) VALUES (?,?,?,?,?,?,?)",
                        (a, b, norm["mfcc"], norm["bpm"], norm["key"], norm["energy"], score))
                    existing_updated += 1

    return {"new_tracks": len(new_ids), "pairs_stored": pairs_stored,
            "existing_updated": existing_updated}


def full_rebuild(conn, top_k: int = 20) -> dict:
    conn.execute("DELETE FROM track_similarity")
    return build_similarity(conn, top_k)
