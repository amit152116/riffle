import json

import pytest

from audiolib import calibrate, match
from tests.fixtures import make_tone, transcode, trim


def test_load_pairs_requires_every_category(tmp_path):
    p = tmp_path / "pairs.json"
    p.write_text(json.dumps([{"a": "x", "b": "y", "expect": 1,
                              "set": "calibrate"}]))
    with pytest.raises(calibrate.CalibrationError):
        calibrate.load_pairs(p)


def test_load_pairs_requires_both_sets(tmp_path):
    entries = [{"a": f"a{i}", "b": f"b{i}", "expect": e, "set": "calibrate"}
               for i, e in enumerate([0, 1, 2, 1, 2, 0, 1])]
    p = tmp_path / "pairs.json"
    p.write_text(json.dumps(entries))
    with pytest.raises(calibrate.CalibrationError):
        calibrate.load_pairs(p)


def test_evaluate_scores_a_perfect_config():
    pairs = [{"fp_a": None, "fp_b": None, "expect": 1, "predicted": 1},
             {"fp_a": None, "fp_b": None, "expect": 0, "predicted": 0}]
    result = calibrate.score(pairs)
    assert result["accuracy"] == 1.0
    assert result["correct"] == 2


def test_evaluate_reports_the_confusion():
    pairs = [{"expect": 1, "predicted": 2}, {"expect": 1, "predicted": 1}]
    result = calibrate.score(pairs)
    assert result["accuracy"] == 0.5
    assert result["confusion"][(1, 2)] == 1


def test_search_reports_both_sets(tmp_path):
    src = make_tone(tmp_path / "src.flac", seconds=60.0)
    mp3 = transcode(src, tmp_path / "src.mp3", codec="libmp3lame",
                    bitrate="128k")
    clip = trim(src, tmp_path / "clip.flac", start=10.0, duration=25.0)
    other = make_tone(tmp_path / "other.flac", seconds=60.0, freq=1700)

    entries = [
        {"a": str(src), "b": str(mp3), "expect": 1, "set": "calibrate"},
        {"a": str(src), "b": str(clip), "expect": 2, "set": "calibrate"},
        {"a": str(src), "b": str(other), "expect": 0, "set": "calibrate"},
        {"a": str(src), "b": str(mp3), "expect": 1, "set": "holdout"},
        {"a": str(src), "b": str(clip), "expect": 2, "set": "holdout"},
        {"a": str(src), "b": str(other), "expect": 0, "set": "holdout"},
    ]
    p = tmp_path / "pairs.json"
    p.write_text(json.dumps(entries))

    best, scores = calibrate.search(
        calibrate.load_pairs(p), match.DEFAULT_MATCH_CONFIG)
    assert "calibrate" in scores and "holdout" in scores
    assert scores["calibrate"]["accuracy"] >= 0.6
    assert set(best) >= set(match.DEFAULT_MATCH_CONFIG)
