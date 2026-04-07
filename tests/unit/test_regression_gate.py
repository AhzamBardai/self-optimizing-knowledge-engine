"""Unit tests for the CI regression gate."""
import json
import tempfile
from pathlib import Path

import pytest

from knowledge_engine.evaluation.regression_gate import (
    RegressionReport,
    check_regression,
    load_baseline,
    save_baseline,
)

BASELINE = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.75,
    "aggregate": 0.80,
}

WITHIN_THRESHOLD = {
    "faithfulness": 0.83,
    "answer_relevancy": 0.79,
    "context_precision": 0.74,
    "aggregate": 0.79,
}

EXCEEDS_THRESHOLD = {
    "faithfulness": 0.75,  # 11.8% drop
    "answer_relevancy": 0.80,
    "context_precision": 0.75,
    "aggregate": 0.77,
}


class TestCheckRegression:
    def test_passes_when_within_threshold(self) -> None:
        report = check_regression(WITHIN_THRESHOLD, BASELINE, threshold=0.05)
        assert report.passed is True
        assert report.failures == []

    def test_fails_when_metric_drops_more_than_threshold(self) -> None:
        report = check_regression(EXCEEDS_THRESHOLD, BASELINE, threshold=0.05)
        assert report.passed is False
        assert "faithfulness" in report.failures

    def test_passes_when_no_baseline(self) -> None:
        report = check_regression(WITHIN_THRESHOLD, {}, threshold=0.05)
        assert report.passed is True

    def test_passes_when_metric_improves(self) -> None:
        improved = {
            "faithfulness": 0.90,
            "answer_relevancy": 0.85,
            "context_precision": 0.80,
            "aggregate": 0.85,
        }
        report = check_regression(improved, BASELINE, threshold=0.05)
        assert report.passed is True

    def test_details_contain_all_tracked_metrics(self) -> None:
        report = check_regression(WITHIN_THRESHOLD, BASELINE, threshold=0.05)
        for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
            assert metric in report.details

    def test_detail_has_required_fields(self) -> None:
        report = check_regression(WITHIN_THRESHOLD, BASELINE, threshold=0.05)
        for detail in report.details.values():
            assert "baseline" in detail
            assert "current" in detail
            assert "delta" in detail
            assert "threshold" in detail

    def test_threshold_respected(self) -> None:
        at_threshold = {
            "faithfulness": BASELINE["faithfulness"] * (1 - 0.05),
            "answer_relevancy": BASELINE["answer_relevancy"],
            "context_precision": BASELINE["context_precision"],
            "aggregate": BASELINE["aggregate"],
        }
        report = check_regression(at_threshold, BASELINE, threshold=0.05)
        assert "faithfulness" not in report.failures

    def test_custom_threshold(self) -> None:
        report = check_regression(WITHIN_THRESHOLD, BASELINE, threshold=0.01)
        assert report.passed is False


class TestBaselineIO:
    def test_save_and_load_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/baseline.json"
            save_baseline(BASELINE, path)
            loaded = load_baseline(path)
            assert loaded == BASELINE

    def test_load_missing_baseline_returns_empty(self) -> None:
        result = load_baseline("/nonexistent/path/baseline.json")
        assert result == {}
