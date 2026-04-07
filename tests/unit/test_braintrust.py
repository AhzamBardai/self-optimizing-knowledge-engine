"""Unit tests for BraintrustJudge — all Anthropic calls are mocked."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from knowledge_engine.evaluation.braintrust_judge import BraintrustJudge, BraintrustResult, BraintrustScore


def _make_judge() -> BraintrustJudge:
    with patch("knowledge_engine.evaluation.braintrust_judge.anthropic.Anthropic"):
        judge = BraintrustJudge(anthropic_api_key="test-key")
    return judge


def _mock_haiku_response(text: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=text)]
    return mock_response


class TestBraintrustJudge:
    def test_judge_returns_result(self) -> None:
        judge = _make_judge()
        judge._client = MagicMock()
        judge._client.messages.create.return_value = _mock_haiku_response(
            "Score: 4\nRationale: The answer is well grounded."
        )

        result = judge.judge(
            question="What was Apple's revenue?",
            answer="Apple revenue was $383.3 billion.",
            contexts=["Apple total net sales of $383.3 billion for fiscal year 2023."],
        )
        assert isinstance(result, BraintrustResult)

    def test_judge_returns_three_dimension_scores(self) -> None:
        judge = _make_judge()
        judge._client = MagicMock()
        judge._client.messages.create.return_value = _mock_haiku_response(
            "Score: 4\nRationale: Good."
        )

        result = judge.judge(
            question="What was Apple's revenue?",
            answer="Apple revenue was $383.3 billion.",
            contexts=["Apple reported $383.3 billion."],
        )
        assert len(result.scores) == 3

    def test_scores_normalized_to_0_1(self) -> None:
        judge = _make_judge()
        judge._client = MagicMock()
        judge._client.messages.create.return_value = _mock_haiku_response(
            "Score: 5\nRationale: Perfect."
        )

        result = judge.judge(
            question="What was Apple's revenue?",
            answer="Apple revenue was $383.3 billion.",
            contexts=["Apple reported $383.3 billion."],
        )
        for score in result.scores:
            assert 0.0 <= score.score <= 1.0

    def test_aggregate_score_is_average(self) -> None:
        result = BraintrustResult(
            question="q",
            answer="a",
            scores=[
                BraintrustScore("correctness", 0.5, "ok"),
                BraintrustScore("groundedness", 1.0, "ok"),
                BraintrustScore("completeness", 0.75, "ok"),
            ],
        )
        assert abs(result.aggregate_score - 0.75) < 0.001

    def test_malformed_response_defaults_to_score_3(self) -> None:
        judge = _make_judge()
        judge._client = MagicMock()
        judge._client.messages.create.return_value = _mock_haiku_response(
            "I cannot determine a score."
        )

        result = judge.judge(
            question="q?",
            answer="answer",
            contexts=["context"],
        )
        # Score 3 normalizes to (3-1)/4 = 0.5
        for score in result.scores:
            assert abs(score.score - 0.5) < 0.001
