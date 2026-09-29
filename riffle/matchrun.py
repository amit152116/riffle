"""One matching pass over the library, recorded for reproducibility."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

import riffle
from riffle import fingerprint, match, store


def load_fingerprints(conn) -> dict[int, np.ndarray]:
    rows = conn.execute(
        "SELECT f.audio_content_id AS cid, f.fp_raw, f.fp_length "
        "FROM fingerprint f "
        "WHERE f.purpose = 'canonical' AND EXISTS ("
        "  SELECT 1 FROM track t "
        "  WHERE t.audio_content_id = f.audio_content_id AND t.present = 1)"
    ).fetchall()
    return {
        r["cid"]: store.unpack_fingerprint(r["fp_raw"], r["fp_length"])
        for r in rows
    }


def run_match(conn, config: dict = match.DEFAULT_MATCH_CONFIG) -> int:
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "INSERT INTO match_run (created_at, fingerprint_config, match_config, "
        " software_version, status) VALUES (?,?,?,?,'running')",
        (now,
         json.dumps(fingerprint.DEFAULT_CONFIG, sort_keys=True),
         json.dumps(config, sort_keys=True),
         riffle.__version__),
    )
    run_id = cur.lastrowid

    try:
        conn.execute(
            "INSERT INTO run_track (run_id, track_id, path, size, mtime, "
            " audio_hash, hash_method) "
            "SELECT ?, t.id, t.path, t.size, t.mtime, ac.audio_hash, "
            "       ac.hash_method "
            "FROM track t JOIN audio_content ac ON ac.id = t.audio_content_id "
            "WHERE t.present = 1",
            (run_id,),
        )

        fps = load_fingerprints(conn)
        postings = match.build_postings(fps, config)
        item_seconds = fingerprint.item_duration_seconds()

        for a, b in sorted(match.candidate_pairs(
                postings, config.get("min_shared_keys", 1))):
            ev = match.compare(fps[a], fps[b], config, item_seconds)
            if ev.tier == match.TIER_NONE:
                continue
            conn.execute(
                "INSERT INTO pair (run_id, a_content_id, b_content_id, "
                " best_offset, peak_votes, peak_vote_ratio, "
                " matched_span_items, matched_span_seconds, coverage_a, "
                " coverage_b, mean_bit_error, segment_count, tier, "
                " verified_direct) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,0)",
                (run_id, a, b, ev.best_offset, ev.peak_votes,
                 ev.peak_vote_ratio, ev.matched_span_items,
                 ev.matched_span_seconds, ev.coverage_a, ev.coverage_b,
                 ev.mean_bit_error, ev.segment_count, ev.tier),
            )

        conn.execute("UPDATE match_run SET status='complete' WHERE id=?",
                     (run_id,))
    except Exception:
        conn.execute("UPDATE match_run SET status='failed' WHERE id=?",
                     (run_id,))
        raise
    return run_id
