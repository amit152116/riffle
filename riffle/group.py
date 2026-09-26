"""Content-level grouping, expanded to tracks at the end.

The K and M caps make candidate generation lossy, so a missing edge is not
evidence of a non-match. Every component is re-verified pairwise with the caps
bypassed before clique-versus-chain is decided; otherwise `formed_by_chain`
would fire on generation misses rather than on genuine non-matches.
"""
from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations

from riffle import fingerprint, match, matchrun


def components(edges: set[tuple[int, int]]) -> list[set[int]]:
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)

    out: dict[int, set[int]] = {}
    for node in parent:
        out.setdefault(find(node), set()).add(node)
    return list(out.values())


def _tier0_components(conn) -> list[set[int]]:
    """Content ids reachable from more than one present track."""
    rows = conn.execute(
        "SELECT audio_content_id AS cid, count(*) AS n FROM track "
        "WHERE present = 1 AND audio_content_id IS NOT NULL "
        "GROUP BY audio_content_id HAVING n > 1"
    ).fetchall()
    return [{r["cid"]} for r in rows]


def build_groups(conn, run_id: int, config: dict = match.DEFAULT_MATCH_CONFIG,
                 verifier=None) -> int:
    """Write dup_group, group_content and group_member rows for a run.

    `verifier(a, b) -> tier` overrides direct verification; tests use it to
    construct a chain without needing real audio.
    """
    now = datetime.now(timezone.utc).isoformat()

    # The real verifier's full PairEvidence, keyed by (a, b), when available.
    # A test-injected verifier returns tier alone and never populates this,
    # so those rows keep the tier-only shape the tests construct.
    full_evidence: dict[tuple[int, int], object] = {}

    if verifier is None:
        fps = matchrun.load_fingerprints(conn)
        item_seconds = fingerprint.item_duration_seconds()

        def verifier(a, b):  # noqa: F811 - deliberate local default
            if a not in fps or b not in fps:
                return match.TIER_NONE
            ev = match.compare(fps[a], fps[b], config, item_seconds)
            full_evidence[(a, b)] = ev
            return ev.tier

    edges = {
        (r["a_content_id"], r["b_content_id"])
        for r in conn.execute(
            "SELECT a_content_id, b_content_id FROM pair "
            "WHERE run_id = ? AND tier = 1", (run_id,))
    }

    written = 0
    for comp in components(edges):
        members = sorted(comp)
        verified: dict[tuple[int, int], int] = {}
        for a, b in combinations(members, 2):
            tier = verifier(a, b)
            verified[(a, b)] = tier
            ev = full_evidence.get((a, b))
            if ev is not None:
                # Direct verification with real evidence: store the full
                # picture, not just the tier -- this is the pair a chain
                # group exists to let a person inspect.
                conn.execute(
                    "INSERT INTO pair (run_id, a_content_id, b_content_id, "
                    " best_offset, peak_votes, peak_vote_ratio, "
                    " matched_span_items, matched_span_seconds, coverage_a, "
                    " coverage_b, mean_bit_error, segment_count, tier, "
                    " verified_direct) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,1) "
                    "ON CONFLICT(run_id, a_content_id, b_content_id) "
                    "DO UPDATE SET "
                    "best_offset=excluded.best_offset, "
                    "peak_votes=excluded.peak_votes, "
                    "peak_vote_ratio=excluded.peak_vote_ratio, "
                    "matched_span_items=excluded.matched_span_items, "
                    "matched_span_seconds=excluded.matched_span_seconds, "
                    "coverage_a=excluded.coverage_a, "
                    "coverage_b=excluded.coverage_b, "
                    "mean_bit_error=excluded.mean_bit_error, "
                    "segment_count=excluded.segment_count, "
                    "tier=excluded.tier, verified_direct=1",
                    (run_id, a, b, ev.best_offset, ev.peak_votes,
                     ev.peak_vote_ratio, ev.matched_span_items,
                     ev.matched_span_seconds, ev.coverage_a, ev.coverage_b,
                     ev.mean_bit_error, ev.segment_count, ev.tier),
                )
            else:
                conn.execute(
                    "INSERT INTO pair (run_id, a_content_id, b_content_id, "
                    " tier, verified_direct) VALUES (?,?,?,?,1) "
                    "ON CONFLICT(run_id, a_content_id, b_content_id) "
                    "DO UPDATE SET tier = excluded.tier, verified_direct = 1",
                    (run_id, a, b, tier),
                )

        is_clique = all(t == 1 for t in verified.values())
        tier = 1 if is_clique else 2
        _write_group(conn, run_id, members, tier,
                     formed_by_chain=0 if is_clique else 1, now=now)
        written += 1

    for comp in _tier0_components(conn):
        _write_group(conn, run_id, sorted(comp), tier=0,
                     formed_by_chain=0, now=now)
        written += 1

    return written


def _write_group(conn, run_id: int, content_ids: list[int], tier: int,
                 formed_by_chain: int, now: str) -> int:
    cur = conn.execute(
        "INSERT INTO dup_group (run_id, tier, formed_by_chain, decision) "
        "VALUES (?,?,?, 'proposed')", (run_id, tier, formed_by_chain)
    )
    group_id = cur.lastrowid
    for cid in content_ids:
        conn.execute(
            "INSERT INTO group_content (group_id, audio_content_id) "
            "VALUES (?,?)", (group_id, cid))
        for row in conn.execute(
            "SELECT id FROM track WHERE audio_content_id = ? AND present = 1",
            (cid,),
        ):
            conn.execute(
                "INSERT INTO group_member (group_id, track_id, is_keeper) "
                "VALUES (?,?,0)",
                (group_id, row["id"]))
    return group_id
