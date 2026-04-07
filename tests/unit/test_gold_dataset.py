"""Validate gold dataset schema and distribution."""
import json
from pathlib import Path

import pytest

FIXTURE_PATH = Path("tests/fixtures/gold_qa_5.json")


def test_fixture_loads() -> None:
    data = json.loads(FIXTURE_PATH.read_text())
    assert len(data) == 5


def test_fixture_has_required_fields() -> None:
    data = json.loads(FIXTURE_PATH.read_text())
    required = {"id", "question", "answer", "supporting_chunks", "question_type", "difficulty", "tickers"}
    for item in data:
        missing = required - set(item.keys())
        assert not missing, f"Item {item['id']} missing fields: {missing}"


def test_fixture_question_types_valid() -> None:
    data = json.loads(FIXTURE_PATH.read_text())
    valid_types = {"simple", "multi_hop", "comparative"}
    for item in data:
        assert item["question_type"] in valid_types


def test_fixture_difficulty_in_range() -> None:
    data = json.loads(FIXTURE_PATH.read_text())
    for item in data:
        assert 1 <= item["difficulty"] <= 5


def test_full_dataset_distribution() -> None:
    full_path = Path("data/gold_dataset.json")
    if not full_path.exists():
        pytest.skip("Full gold dataset not generated yet (run: python scripts/generate_gold_dataset.py)")

    data = json.loads(full_path.read_text())
    assert len(data) == 100

    by_type: dict[str, int] = {}
    for item in data:
        qt = item["question_type"]
        by_type[qt] = by_type.get(qt, 0) + 1

    assert by_type.get("simple", 0) == 40
    assert by_type.get("multi_hop", 0) == 30
    assert by_type.get("comparative", 0) == 30
