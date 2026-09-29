"""Threshold tuning against known pairs, validated on a holdout set.

Upstream's constants are the starting point. Calibration confirms or adjusts
them for this library, and reports both sets, because a small holdout is not
strong evidence on its own.
"""
from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

from riffle import fingerprint, match

REQUIRED_EXPECTATIONS = {0, 1, 2}
REQUIRED_SETS = {"calibrate", "holdout"}


class CalibrationError(Exception):
    """The pair file cannot support a meaningful calibration."""


def load_pairs(path: Path) -> list[dict]:
    path = Path(path)
    try:
        text = path.read_text()
    except OSError:
        raise CalibrationError(f"pairs file not found: {path}") from None
    try:
        entries = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CalibrationError(f"{path} contains invalid JSON: {exc}") from None
    seen_sets = {e["set"] for e in entries}
    if not REQUIRED_SETS <= seen_sets:
        raise CalibrationError(
            f"need both {sorted(REQUIRED_SETS)} sets, found {sorted(seen_sets)}")
    for name in REQUIRED_SETS:
        expects = {e["expect"] for e in entries if e["set"] == name}
        if not REQUIRED_EXPECTATIONS <= expects:
            raise CalibrationError(
                f"the '{name}' set must contain all of "
                f"{sorted(REQUIRED_EXPECTATIONS)}, found {sorted(expects)}")
    return entries


def score(pairs: list[dict]) -> dict:
    confusion = Counter(
        (p["expect"], p["predicted"]) for p in pairs
    )
    correct = sum(n for (e, p), n in confusion.items() if e == p)
    total = len(pairs)
    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "confusion": dict(confusion),
    }


def _predict(entries, config, item_seconds, cache) -> list[dict]:
    out = []
    for e in entries:
        for side in ("a", "b"):
            if e[side] not in cache:
                cache[e[side]] = fingerprint.fingerprint_file(Path(e[side])).raw
        ev = match.compare(cache[e["a"]], cache[e["b"]], config, item_seconds)
        out.append({"expect": e["expect"], "predicted": ev.tier})
    return out


_GRID = {
    "tier1_min_coverage": [0.75, 0.85, 0.92],
    "tier1_max_bit_error": [4.0, 6.0, 8.0],
    "tier1_min_overlap_seconds": [10.0, 20.0, 30.0],
    "tier2_min_overlap_seconds": [8.0, 15.0, 25.0],
}


def search(entries: list[dict], base_config: dict,
           item_seconds: float | None = None) -> tuple[dict, dict]:
    item_seconds = item_seconds or fingerprint.item_duration_seconds()
    cal = [e for e in entries if e["set"] == "calibrate"]
    hold = [e for e in entries if e["set"] == "holdout"]
    cache: dict[str, object] = {}

    best_config = dict(base_config)
    best_accuracy = -1.0
    keys = sorted(_GRID)
    for combo in itertools.product(*(_GRID[k] for k in keys)):
        config = dict(base_config, **dict(zip(keys, combo)))
        result = score(_predict(cal, config, item_seconds, cache))
        # Ties resolve to the first combination in sorted grid order.
        if result["accuracy"] > best_accuracy:
            best_accuracy = result["accuracy"]
            best_config = config

    return best_config, {
        "calibrate": score(_predict(cal, best_config, item_seconds, cache)),
        "holdout": score(_predict(hold, best_config, item_seconds, cache)),
    }
