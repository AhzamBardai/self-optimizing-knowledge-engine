"""
Cross-encoder reranker using ms-marco-MiniLM-L-6-v2.

Scores each (query, chunk) pair with a cross-encoder and returns
top_n sorted by descending relevance score.
"""
from __future__ import annotations

import structlog
from sentence_transformers import CrossEncoder

log = structlog.get_logger()

_RERANKER_CACHE: dict[str, CrossEncoder] = {}


def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    """Return cached CrossEncoder model."""
    if model_name not in _RERANKER_CACHE:
        log.info("reranker.loading", model=model_name)
        _RERANKER_CACHE[model_name] = CrossEncoder(model_name)
    return _RERANKER_CACHE[model_name]


def rerank(
    query: str,
    candidates: list[dict[str, str | float]],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n: int = 5,
) -> list[dict[str, str | float]]:
    """
    Rerank candidate chunks with a cross-encoder.

    Args:
        query: The user's question.
        candidates: List of dicts with at least 'chunk_id' and 'text'.
        model_name: Cross-encoder model to use.
        top_n: Number of results to return.

    Returns:
        Top-n reranked candidates with added 'rerank_score' field.
    """
    if not candidates:
        return []

    model = get_reranker(model_name)
    pairs = [(query, str(c.get("text", ""))) for c in candidates]
    scores = model.predict(pairs)

    scored = sorted(
        zip(candidates, scores, strict=True),
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    result = [{**item, "rerank_score": float(score)} for item, score in scored]
    log.debug("reranker.done", input_count=len(candidates), output_count=len(result))
    return result
