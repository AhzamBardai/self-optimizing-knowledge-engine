"""BLEU-4 and ROUGE-L scoring utilities."""
from __future__ import annotations

import structlog
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU

log = structlog.get_logger()

_BLEU = BLEU(max_ngram_order=4)
_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def bleu4(hypothesis: str, references: list[str]) -> float:
    """
    Compute BLEU-4 score.

    Args:
        hypothesis: Model-generated answer.
        references: List of reference answers (at least one).

    Returns:
        BLEU-4 score in [0, 100].
    """
    if not hypothesis.strip() or not any(r.strip() for r in references):
        return 0.0
    result = _BLEU.sentence_score(hypothesis, references)
    return float(result.score)


def rouge_l(hypothesis: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 score.

    Args:
        hypothesis: Model-generated answer.
        reference: Reference answer.

    Returns:
        ROUGE-L F1 in [0, 1].
    """
    if not hypothesis.strip() or not reference.strip():
        return 0.0
    scores = _ROUGE.score(reference, hypothesis)
    return float(scores["rougeL"].fmeasure)


def score_pair(hypothesis: str, reference: str) -> dict[str, float]:
    """Compute all metrics for a hypothesis/reference pair."""
    return {
        "bleu4": bleu4(hypothesis, [reference]),
        "rouge_l": rouge_l(hypothesis, reference),
    }
