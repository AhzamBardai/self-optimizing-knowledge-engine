"""BM25 sparse retrieval over 10-K chunks."""
from __future__ import annotations

import re

import structlog
from rank_bm25 import BM25Okapi

log = structlog.get_logger()


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Index:
    """BM25 index over a corpus of text chunks."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._corpus: list[dict[str, str]] = []

    def build(self, chunks: list[dict[str, str]]) -> None:
        """
        Build the BM25 index from chunks.

        Args:
            chunks: List of dicts with 'chunk_id' and 'text' keys.
        """
        self._corpus = chunks
        tokenized = [_tokenize(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        log.info("bm25.index_built", corpus_size=len(chunks))

    def search(self, query: str, top_k: int = 10) -> list[dict[str, str | float]]:
        """
        BM25 retrieval.

        Returns list of dicts with 'chunk_id', 'text', 'score'.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25Index.build() must be called before search()")

        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        # Pair each score with its chunk and sort descending
        scored = sorted(
            zip(scores, self._corpus, strict=True),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]

        return [
            {"chunk_id": chunk["chunk_id"], "text": chunk["text"], "score": float(score)}
            for score, chunk in scored
        ]
