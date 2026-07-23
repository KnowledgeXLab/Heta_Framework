import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.kb.steps.community_report import parse_community_report_json  # noqa: E402


def test_parse_community_report_json_extracts_first_complete_json() -> None:
    parsed = parse_community_report_json(
        'Here is the report:\n{"title": "T", "summary": "S", "rating": 7}\nDone.'
    )

    assert parsed["title"] == "T"
    assert parsed["summary"] == "S"
    assert parsed["rating"] == 7


def test_parse_community_report_json_falls_back_to_relaxed_key_values() -> None:
    parsed = parse_community_report_json(
        'title: "T", summary: "S", rating: 3, rating_explanation: "ok"'
    )

    assert parsed["title"] == "T"
    assert parsed["summary"] == "S"
    assert parsed["rating"] == 3
    assert parsed["rating_explanation"] == "ok"
