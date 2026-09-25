"""Human-readable and JSON reports for a match run."""
from __future__ import annotations


def _escape(path: str) -> str:
    """Paths may contain newlines; line-oriented output must not be forgeable."""
    return path.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")


TIER_NAMES = {0: "identical stream", 1: "same recording", 2: "variant"}


def report_data(conn, run_id: int, tier: int | None = None) -> dict:
    groups = []
    warnings = []

    query = "SELECT * FROM dup_group WHERE run_id = ?"
    params: list = [run_id]
    if tier is not None:
        query += " AND tier = ?"
        params.append(tier)

    for g in conn.execute(query + " ORDER BY id", params):
        members = []
        for m in conn.execute(
            "SELECT gm.track_id, gm.is_keeper, gm.rank_score, t.path, "
            "       t.bitrate, ac.hash_method, ac.duration "
            "FROM group_member gm "
            "JOIN track t ON t.id = gm.track_id "
            "JOIN audio_content ac ON ac.id = gm.audio_content_id "
            "WHERE gm.group_id = ? ORDER BY gm.track_id", (g["id"],)
        ):
            if m["hash_method"] == "whole_file":
                warnings.append(
                    f"{_escape(m['path'])} uses a whole-file identity, "
                    "which does not survive retagging"
                )
            members.append({
                "track_id": m["track_id"],
                "path": m["path"],
                "is_keeper": bool(m["is_keeper"]),
                "bitrate": m["bitrate"],
                "hash_method": m["hash_method"],
                "duration": m["duration"],
                "rank_score": m["rank_score"],
            })

        content_ids = [r["audio_content_id"] for r in conn.execute(
            "SELECT audio_content_id FROM group_content WHERE group_id = ?",
            (g["id"],))]
        evidence = []
        if content_ids:
            placeholders = ",".join("?" * len(content_ids))
            for p in conn.execute(
                f"SELECT * FROM pair WHERE run_id = ? "
                f"AND a_content_id IN ({placeholders}) "
                f"AND b_content_id IN ({placeholders})",
                [run_id, *content_ids, *content_ids],
            ):
                evidence.append({
                    "a": p["a_content_id"], "b": p["b_content_id"],
                    "tier": p["tier"],
                    "best_offset": p["best_offset"],
                    "peak_vote_ratio": p["peak_vote_ratio"],
                    "coverage_a": p["coverage_a"],
                    "coverage_b": p["coverage_b"],
                    "mean_bit_error": p["mean_bit_error"],
                    "matched_span_seconds": p["matched_span_seconds"],
                    "verified_direct": bool(p["verified_direct"]),
                })

        groups.append({
            "group_id": g["id"],
            "tier": g["tier"],
            "tier_name": TIER_NAMES.get(g["tier"], "unknown"),
            "formed_by_chain": bool(g["formed_by_chain"]),
            "decision": g["decision"],
            "members": members,
            "evidence": evidence,
        })

    return {"run_id": run_id, "groups": groups,
            "warnings": sorted(set(warnings))}


def _fmt(value, spec: str) -> str:
    """Format a possibly-NULL evidence field. A verification-only pair
    (direct chain re-check with no captured full evidence) carries only
    tier and verified_direct; every other field is NULL -- and that is
    exactly the kind of group a person most needs to see in the report.
    """
    return "-" if value is None else format(value, spec)


def render_text(data: dict) -> str:
    lines = [f"Match run {data['run_id']}", ""]
    if not data["groups"]:
        lines.append("No groups.")
    for g in data["groups"]:
        chain = " [CHAIN - review only]" if g["formed_by_chain"] else ""
        lines.append(
            f"Group {g['group_id']}  tier {g['tier']} "
            f"({g['tier_name']}){chain}  decision: {g['decision']}"
        )
        for m in g["members"]:
            marker = "KEEP  " if m["is_keeper"] else "loser "
            lines.append(f"  {marker} {_escape(m['path'])}")
        for e in g["evidence"]:
            lines.append(
                f"    {e['a']}~{e['b']} tier {e['tier']} "
                f"cov {_fmt(e['coverage_a'], '.2f')}/"
                f"{_fmt(e['coverage_b'], '.2f')} "
                f"err {_fmt(e['mean_bit_error'], '.2f')} "
                f"span {_fmt(e['matched_span_seconds'], '.1f')}s"
            )
        lines.append("")
    for w in data["warnings"]:
        lines.append(f"warning: {w}")
    return "\n".join(lines)
