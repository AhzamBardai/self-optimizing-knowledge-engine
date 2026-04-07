"""
Reciprocal Rank Fusion (RRF) to combine dense and sparse rankings.

Formula: RRF(d) = Σ 1 / (k + rank(d))
Default k=60 from the original Cormack et al. paper.
"""
from __future__ import annotations

import structlog

log = structlog.get_logger()


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, str | float]]],
    k: int = 60,
    top_n: int | None = None,
) -> list[dict[str, str | float]]:
    """
    Fuse multiple ranked result lists using RRF.

    Args:
        ranked_lists: Each list is ordered best-first, containing dicts
                      with at least 'chunk_id' and 'text' keys.
        k: RRF smoothing constant (default 60).
        top_n: Return only the top-n results (None = return all).

    Returns:
        Merged and re-ranked list of dicts with 'chunk_id', 'text', 'rrf_score'.
    """
    scores: dict[str, float] = {}
    texts: dict[str, str] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            chunk_id = str(item["chunk_id"])
            text = str(item.get("text", ""))
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            texts[chunk_id] = text

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    if top_n is not None:
        merged = merged[:top_n]

    result = [
        {"chunk_id": cid, "text": texts[cid], "rrf_score": score}
        for cid, score in merged
    ]

    log.debug(
        "rrf.fusion_done",
        input_lists=len(ranked_lists),
        unique_chunks=len(scores),
        output_count=len(result),
    )
    return result
