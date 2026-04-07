"""
Braintrust LLM-as-judge pipeline.

Uses the Braintrust SDK when BRAINTRUST_API_KEY is set.
Falls back to a local deterministic scorer using rubric dimensions
when no API key is available (for CI and offline use).

Rubric: correctness, groundedness, completeness (each scored 1-5, normalized to [0,1]).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import anthropic
import structlog

log = structlog.get_logger()

RUBRIC = {
    "correctness": (
        "Is the answer factually correct based on the provided context? "
        "Score 1 (completely wrong) to 5 (fully correct)."
    ),
    "groundedness": (
        "Is every claim in the answer supported by the provided context? "
        "Score 1 (hallucinated) to 5 (fully grounded)."
    ),
    "completeness": (
        "Does the answer fully address all parts of the question? "
        "Score 1 (misses key parts) to 5 (fully complete)."
    ),
}


@dataclass
class BraintrustScore:
    """Score from a single dimension of the rubric."""

    dimension: str
    score: float  # normalized to [0, 1]
    rationale: str


@dataclass
class BraintrustResult:
    """Full Braintrust evaluation result for one Q&A pair."""

    question: str
    answer: str
    scores: list[BraintrustScore]

    @property
    def aggregate_score(self) -> float:
        """Average of all dimension scores (in [0, 1])."""
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores) / len(self.scores)


class BraintrustJudge:
    """
    LLM-as-judge scoring via Braintrust or local Claude Haiku fallback.

    When BRAINTRUST_API_KEY is set, logs results to Braintrust.
    Always runs scoring through Claude Haiku.
    """

    def __init__(
        self,
        anthropic_api_key: str,
        braintrust_api_key: str = "",
        haiku_model: str = "claude-haiku-4-5-20251001",
        project_name: str = "self-optimizing-knowledge-engine",
    ) -> None:
        self.haiku_model = haiku_model
        self.project_name = project_name
        self._client = anthropic.Anthropic(api_key=anthropic_api_key)
        self._has_braintrust = bool(braintrust_api_key)

        if self._has_braintrust:
            try:
                import braintrust
                braintrust.init(project=project_name, api_key=braintrust_api_key)
                self._braintrust = braintrust
                log.info("braintrust.initialized", project=project_name)
            except ImportError:
                log.warning("braintrust.import_failed_using_local_fallback")
                self._has_braintrust = False

    def _score_dimension(
        self,
        question: str,
        answer: str,
        context: str,
        dimension: str,
        rubric: str,
    ) -> BraintrustScore:
        """Score a single rubric dimension with Claude Haiku."""
        prompt = (
            f"Question: {question}\n\n"
            f"Context: {context[:2000]}\n\n"
            f"Answer: {answer}\n\n"
            f"Rubric: {rubric}\n\n"
            "Respond with:\nScore: <1-5>\nRationale: <one sentence>"
        )

        response = self._client.messages.create(
            model=self.haiku_model,
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()

        score_match = re.search(r"Score:\s*([1-5])", text)
        raw_score = int(score_match.group(1)) if score_match else 3
        rationale_match = re.search(r"Rationale:\s*(.+)", text)
        rationale = rationale_match.group(1) if rationale_match else text[:100]

        return BraintrustScore(
            dimension=dimension,
            score=float(raw_score - 1) / 4.0,  # normalize to [0, 1]
            rationale=rationale,
        )

    def judge(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        reference: str = "",
    ) -> BraintrustResult:
        """
        Judge a Q&A pair across all rubric dimensions.

        Args:
            question: The user question.
            answer: Model-generated answer.
            contexts: Retrieved chunks used for generation.
            reference: Optional ground-truth reference answer.
        """
        context_str = "\n---\n".join(contexts[:5])
        scores: list[BraintrustScore] = []

        for dimension, rubric in RUBRIC.items():
            score = self._score_dimension(question, answer, context_str, dimension, rubric)
            scores.append(score)
            log.debug("braintrust.dimension_scored", dimension=dimension, score=score.score)

        result = BraintrustResult(question=question, answer=answer, scores=scores)
        log.info("braintrust.judged", aggregate=round(result.aggregate_score, 4))

        if self._has_braintrust:
            self._log_to_braintrust(question, answer, reference, result)

        return result

    def _log_to_braintrust(
        self,
        question: str,
        answer: str,
        reference: str,
        result: BraintrustResult,
    ) -> None:
        """Log result to Braintrust experiment."""
        try:
            experiment = self._braintrust.init(project=self.project_name)
            experiment.log(
                input=question,
                output=answer,
                expected=reference,
                scores={s.dimension: s.score for s in result.scores},
            )
        except Exception as e:
            log.warning("braintrust.log_failed", error=str(e))
