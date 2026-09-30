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
            "JOIN audio_content ac ON ac.id = t.audio_content_id "
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


PROBABLE_MIN_COVERAGE = 0.6
PROBABLE_MAX_BIT_ERROR = 4.0
PROBABLE_MIN_SPAN_SECONDS = 90.0


def _content_paths(conn, content_id: int) -> tuple[list[str], bool]:
    """Every file with this audio, at its quarantine path if it was moved."""
    paths: list[str] = []
    quarantined = False
    for r in conn.execute(
        "SELECT t.path AS path, "
        "       (SELECT q.dst_path FROM quarantine_log q "
        "        WHERE q.track_id = t.id AND q.state = 'moved' "
        "        ORDER BY q.id DESC LIMIT 1) AS quarantine_path "
        "FROM track t WHERE t.audio_content_id = ? ORDER BY t.id",
        (content_id,),
    ):
        if r["quarantine_path"]:
            paths.append(r["quarantine_path"])
            quarantined = True
        else:
            paths.append(r["path"])
    return paths, quarantined


def probable_pairs(conn, run_id: int,
                   min_coverage: float = PROBABLE_MIN_COVERAGE,
                   max_bit_error: float = PROBABLE_MAX_BIT_ERROR,
                   min_span_seconds: float = PROBABLE_MIN_SPAN_SECONDS,
                   ) -> list[dict]:
    """Tier-2 pairs that look like one recording with extra intro/outro.

    Tier 1 needs 85% coverage on both files, so a copy carrying a long intro
    or outro falls just short. Requiring both files to be substantially
    covered (not just one) excludes a song inside a long compilation, where
    the song covers most of itself but a sliver of the other file.

    Review only: nothing here forms a group or authorizes quarantine.
    """
    rows = conn.execute(
        "SELECT * FROM pair WHERE run_id = ? AND tier = 2 "
        "AND min(coverage_a, coverage_b) >= ? "
        "AND mean_bit_error <= ? AND matched_span_seconds >= ? "
        "ORDER BY min(coverage_a, coverage_b) DESC, a_content_id, b_content_id",
        (run_id, min_coverage, max_bit_error, min_span_seconds),
    ).fetchall()
    out = []
    for p in rows:
        a_paths, a_quarantined = _content_paths(conn, p["a_content_id"])
        b_paths, b_quarantined = _content_paths(conn, p["b_content_id"])
        out.append({
            "a_content_id": p["a_content_id"],
            "b_content_id": p["b_content_id"],
            "coverage_a": p["coverage_a"],
            "coverage_b": p["coverage_b"],
            "mean_bit_error": p["mean_bit_error"],
            "matched_span_seconds": p["matched_span_seconds"],
            "a_paths": a_paths, "a_quarantined": a_quarantined,
            "b_paths": b_paths, "b_quarantined": b_quarantined,
        })
    return out


def render_probable(rows: list[dict]) -> str:
    if not rows:
        return "No probable duplicates."
    lines = ["Probable duplicates (review only: nothing here is approved "
             "or applied)", ""]
    for r in rows:
        lines.append(
            f"{r['a_content_id']}~{r['b_content_id']} "
            f"cov {r['coverage_a']:.2f}/{r['coverage_b']:.2f} "
            f"err {r['mean_bit_error']:.2f} "
            f"span {r['matched_span_seconds']:.1f}s")
        for label, paths, quarantined in (
                ("A", r["a_paths"], r["a_quarantined"]),
                ("B", r["b_paths"], r["b_quarantined"])):
            note = " [quarantined]" if quarantined else ""
            for path in paths:
                lines.append(f"  {label} {_escape(path)}{note}")
        lines.append("")
    return "\n".join(lines)
