"""
RAGAS evaluation pipeline for RAG quality assessment.

Metrics:
  - faithfulness: answer grounded in retrieved context (no hallucinations)
  - answer_relevancy: answer addresses the question
  - context_precision: retrieved chunks are relevant to the question

Uses Claude Haiku as the judge LLM (configurable).
Falls back to a simple heuristic scorer if RAGAS is unavailable or API keys missing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

log = structlog.get_logger()


@dataclass
class RAGASResult:
    """RAGAS evaluation result for a single Q&A pair."""

    question: str
    answer: str
    contexts: list[str]
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def aggregate_score(self) -> float:
        """Average of all three RAGAS metrics."""
        return (self.faithfulness + self.answer_relevancy + self.context_precision) / 3


@dataclass
class RAGASReport:
    """Aggregate RAGAS report over multiple Q&A pairs."""

    results: list[RAGASResult]

    @property
    def mean_faithfulness(self) -> float:
        return sum(r.faithfulness for r in self.results) / len(self.results)

    @property
    def mean_answer_relevancy(self) -> float:
        return sum(r.answer_relevancy for r in self.results) / len(self.results)

    @property
    def mean_context_precision(self) -> float:
        return sum(r.context_precision for r in self.results) / len(self.results)

    @property
    def mean_aggregate(self) -> float:
        return sum(r.aggregate_score for r in self.results) / len(self.results)

    def to_dict(self) -> dict[str, float]:
        return {
            "faithfulness": round(self.mean_faithfulness, 4),
            "answer_relevancy": round(self.mean_answer_relevancy, 4),
            "context_precision": round(self.mean_context_precision, 4),
            "aggregate": round(self.mean_aggregate, 4),
            "sample_count": float(len(self.results)),
        }


def _heuristic_faithfulness(answer: str, contexts: list[str]) -> float:
    if not answer.strip() or not contexts:
        return 0.0
    context_text = " ".join(contexts).lower()
    answer_words = [w for w in answer.lower().split() if len(w) > 3]
    if not answer_words:
        return 0.5
    matches = sum(1 for w in answer_words if w in context_text)
    return min(matches / len(answer_words), 1.0)


def _heuristic_relevancy(question: str, answer: str) -> float:
    if not question.strip() or not answer.strip():
        return 0.0
    q_words = [w for w in question.lower().split() if len(w) > 3]
    if not q_words:
        return 0.5
    a_lower = answer.lower()
    matches = sum(1 for w in q_words if w in a_lower)
    return min(matches / len(q_words), 1.0)


def _heuristic_context_precision(question: str, contexts: list[str]) -> float:
    if not question.strip() or not contexts:
        return 0.0
    q_words = [w for w in question.lower().split() if len(w) > 3]
    if not q_words:
        return 0.5
    context_text = " ".join(contexts).lower()
    matches = sum(1 for w in q_words if w in context_text)
    return min(matches / len(q_words), 1.0)


class RAGASEvaluator:
    """
    Evaluates RAG quality using RAGAS metrics.

    Attempts to use the official RAGAS library with Claude Haiku as judge.
    Falls back to deterministic heuristics if RAGAS is unavailable or
    ANTHROPIC_API_KEY is not set.
    """

    def __init__(
        self,
        haiku_api_key: str = "",
        haiku_model: str = "claude-haiku-4-5-20251001",
        use_heuristics: bool = False,
    ) -> None:
        self.haiku_api_key = haiku_api_key
        self.haiku_model = haiku_model
        self._use_heuristics = use_heuristics or not haiku_api_key

        if not self._use_heuristics:
            try:
                self._init_ragas()
            except Exception as e:
                log.warning("ragas.init_failed_falling_back", error=str(e))
                self._use_heuristics = True

    def _init_ragas(self) -> None:
        """Initialize RAGAS with Claude Haiku judge."""
        from langchain_anthropic import ChatAnthropic
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness

        self._ragas_evaluate = evaluate
        self._ragas_metrics = [faithfulness, answer_relevancy, context_precision]
        self._judge_llm = ChatAnthropic(
            model=self.haiku_model,
            api_key=self.haiku_api_key,
        )
        self._embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        log.info("ragas.initialized", judge_model=self.haiku_model)

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        reference: str = "",
    ) -> RAGASResult:
        """Evaluate a single Q&A pair."""
        if self._use_heuristics:
            return RAGASResult(
                question=question,
                answer=answer,
                contexts=contexts,
                faithfulness=_heuristic_faithfulness(answer, contexts),
                answer_relevancy=_heuristic_relevancy(question, answer),
                context_precision=_heuristic_context_precision(question, contexts),
                metadata={"scorer": "heuristic"},
            )

        from datasets import Dataset

        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [reference or answer],
        }
        dataset = Dataset.from_dict(data)
        result = self._ragas_evaluate(
            dataset,
            metrics=self._ragas_metrics,
            llm=self._judge_llm,
            embeddings=self._embeddings,
        )
        scores = result.to_pandas().iloc[0]
        return RAGASResult(
            question=question,
            answer=answer,
            contexts=contexts,
            faithfulness=float(scores.get("faithfulness", 0.0)),
            answer_relevancy=float(scores.get("answer_relevancy", 0.0)),
            context_precision=float(scores.get("context_precision", 0.0)),
            metadata={"scorer": "ragas", "judge_model": self.haiku_model},
        )

    def evaluate_dataset(
        self,
        qa_pairs: list[dict[str, Any]],
    ) -> RAGASReport:
        """
        Evaluate a list of Q&A pairs.

        Each pair must have: question, answer, contexts (list of str).
        Optional: reference (ground truth answer).
        """
        results: list[RAGASResult] = []
        for i, pair in enumerate(qa_pairs):
            log.info("ragas.evaluating_pair", idx=i, total=len(qa_pairs))
            result = self.evaluate_single(
                question=pair["question"],
                answer=pair["answer"],
                contexts=pair.get("contexts", []),
                reference=pair.get("reference", ""),
            )
            results.append(result)

        report = RAGASReport(results=results)
        log.info("ragas.evaluation_complete", metrics=report.to_dict())
        return report
