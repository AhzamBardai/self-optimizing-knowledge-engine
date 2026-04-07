"""
Unit tests for RAGAS pipeline.

Uses heuristic scorer (no API calls) on 5 fixture pairs.
"""
import json
from pathlib import Path

import pytest

from knowledge_engine.evaluation.metrics import bleu4, rouge_l, score_pair
from knowledge_engine.evaluation.ragas_pipeline import RAGASEvaluator, RAGASReport, RAGASResult


class TestRAGASEvaluatorHeuristic:
    @pytest.fixture
    def evaluator(self) -> RAGASEvaluator:
        return RAGASEvaluator(use_heuristics=True)

    def test_evaluate_single_returns_result(self, evaluator: RAGASEvaluator) -> None:
        result = evaluator.evaluate_single(
            question="What was Apple's revenue?",
            answer="Apple's revenue was $383.3 billion in fiscal 2023.",
            contexts=["Apple Inc. reported total net sales of $383.3 billion for fiscal year 2023."],
        )
        assert isinstance(result, RAGASResult)

    def test_evaluate_single_metrics_in_range(self, evaluator: RAGASEvaluator) -> None:
        result = evaluator.evaluate_single(
            question="What was Apple's revenue?",
            answer="Apple's revenue was $383.3 billion in fiscal 2023.",
            contexts=["Apple Inc. reported total net sales of $383.3 billion for fiscal year 2023."],
        )
        assert 0.0 <= result.faithfulness <= 1.0
        assert 0.0 <= result.answer_relevancy <= 1.0
        assert 0.0 <= result.context_precision <= 1.0

    def test_evaluate_dataset_on_fixture_pairs(self, evaluator: RAGASEvaluator) -> None:
        fixture = json.loads(Path("tests/fixtures/gold_qa_5.json").read_text())
        qa_pairs = [
            {
                "question": item["question"],
                "answer": item["answer"],
                "contexts": [item["answer"]],
                "reference": item["answer"],
            }
            for item in fixture
        ]
        report = evaluator.evaluate_dataset(qa_pairs)
        assert isinstance(report, RAGASReport)
        assert len(report.results) == 5

    def test_report_metrics_in_range(self, evaluator: RAGASEvaluator) -> None:
        fixture = json.loads(Path("tests/fixtures/gold_qa_5.json").read_text())
        qa_pairs = [
            {"question": i["question"], "answer": i["answer"], "contexts": [i["answer"]]}
            for i in fixture
        ]
        report = evaluator.evaluate_dataset(qa_pairs)
        assert 0.0 <= report.mean_faithfulness <= 1.0
        assert 0.0 <= report.mean_answer_relevancy <= 1.0
        assert 0.0 <= report.mean_context_precision <= 1.0

    def test_report_to_dict_has_all_keys(self, evaluator: RAGASEvaluator) -> None:
        fixture = json.loads(Path("tests/fixtures/gold_qa_5.json").read_text())
        qa_pairs = [
            {"question": i["question"], "answer": i["answer"], "contexts": [i["answer"]]}
            for i in fixture[:2]
        ]
        report = evaluator.evaluate_dataset(qa_pairs)
        d = report.to_dict()
        assert "faithfulness" in d
        assert "answer_relevancy" in d
        assert "context_precision" in d
        assert "aggregate" in d
        assert d["sample_count"] == 2.0


class TestMetrics:
    def test_bleu4_identical(self) -> None:
        score = bleu4("Apple revenue was 383 billion", ["Apple revenue was 383 billion"])
        assert score > 90

    def test_bleu4_empty_returns_zero(self) -> None:
        assert bleu4("", ["reference"]) == 0.0

    def test_rouge_l_identical(self) -> None:
        score = rouge_l("Apple revenue was 383 billion", "Apple revenue was 383 billion")
        assert abs(score - 1.0) < 0.01

    def test_rouge_l_empty_returns_zero(self) -> None:
        assert rouge_l("", "reference") == 0.0

    def test_score_pair_returns_both_metrics(self) -> None:
        result = score_pair("Apple revenue was 383 billion", "Apple revenue was 383 billion")
        assert "bleu4" in result
        assert "rouge_l" in result
