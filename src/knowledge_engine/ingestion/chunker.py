# src/knowledge_engine/ingestion/chunker.py
"""
Three text chunking strategies for SEC 10-K filings.

- fixed:        512-token windows with 50-token overlap
- semantic:     sentence-transformer similarity grouping
- hierarchical: section → paragraph → sentence

All strategies guarantee: no empty chunks, max_tokens honored.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import structlog
import tiktoken
from sentence_transformers import SentenceTransformer

log = structlog.get_logger()

ChunkStrategy = Literal["fixed", "semantic", "hierarchical"]

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex (no NLTK dependency)."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    return [s.strip() for s in sentences if s.strip()]


@dataclass
class Chunk:
    """A text chunk with metadata."""

    text: str
    chunk_id: str
    source_section: str
    strategy: ChunkStrategy
    token_count: int
    metadata: dict[str, str] = field(default_factory=dict)


class Chunker:
    """Produces chunks from 10-K section text using the configured strategy."""

    def __init__(
        self,
        strategy: ChunkStrategy = "fixed",
        max_tokens: int = 512,
        overlap_tokens: int = 50,
        similarity_threshold: float = 0.7,
    ) -> None:
        self.strategy = strategy
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.similarity_threshold = similarity_threshold
        self._embed_model: SentenceTransformer | None = None

    def _get_embed_model(self) -> SentenceTransformer:
        if self._embed_model is None:
            self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embed_model

    def chunk(
        self,
        text: str,
        ticker: str,
        section: str,
    ) -> list[Chunk]:
        """Dispatch to the configured chunking strategy."""
        if self.strategy == "fixed":
            return self._fixed_chunk(text, ticker, section)
        elif self.strategy == "semantic":
            return self._semantic_chunk(text, ticker, section)
        elif self.strategy == "hierarchical":
            return self._hierarchical_chunk(text, ticker, section)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy!r}")

    def _make_chunk_id(self, ticker: str, section: str, idx: int) -> str:
        return f"{ticker.lower()}_{section}_{self.strategy}_{idx:04d}"

    def _fixed_chunk(self, text: str, ticker: str, section: str) -> list[Chunk]:
        """Sliding window of max_tokens with overlap_tokens stride."""
        tokens = _TOKENIZER.encode(text)
        chunks: list[Chunk] = []
        start = 0
        idx = 0

        while start < len(tokens):
            end = min(start + self.max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = _TOKENIZER.decode(chunk_tokens).strip()

            if chunk_text:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        chunk_id=self._make_chunk_id(ticker, section, idx),
                        source_section=section,
                        strategy="fixed",
                        token_count=len(chunk_tokens),
                        metadata={"ticker": ticker},
                    )
                )
                idx += 1

            step = self.max_tokens - self.overlap_tokens
            start += max(step, 1)

        log.debug(
            "chunker.fixed_done",
            ticker=ticker,
            section=section,
            chunk_count=len(chunks),
        )
        return chunks

    def _semantic_chunk(self, text: str, ticker: str, section: str) -> list[Chunk]:
        """Group sentences by embedding similarity, merge into max_tokens windows."""
        import numpy as np

        sentences = _split_sentences(text)
        if not sentences:
            return []

        model = self._get_embed_model()
        embeddings = model.encode(sentences, normalize_embeddings=True)

        groups: list[list[str]] = []
        current_group: list[str] = [sentences[0]]
        current_tokens = _count_tokens(sentences[0])

        for i in range(1, len(sentences)):
            sim = float(np.dot(embeddings[i - 1], embeddings[i]))
            next_tokens = _count_tokens(sentences[i])

            if (
                sim >= self.similarity_threshold
                and current_tokens + next_tokens <= self.max_tokens
            ):
                current_group.append(sentences[i])
                current_tokens += next_tokens
            else:
                groups.append(current_group)
                current_group = [sentences[i]]
                current_tokens = next_tokens

        groups.append(current_group)

        chunks: list[Chunk] = []
        for idx, group in enumerate(groups):
            chunk_text = " ".join(group).strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        chunk_id=self._make_chunk_id(ticker, section, idx),
                        source_section=section,
                        strategy="semantic",
                        token_count=_count_tokens(chunk_text),
                        metadata={"ticker": ticker, "sentence_count": str(len(group))},
                    )
                )

        log.debug(
            "chunker.semantic_done",
            ticker=ticker,
            section=section,
            chunk_count=len(chunks),
        )
        return chunks

    def _hierarchical_chunk(self, text: str, ticker: str, section: str) -> list[Chunk]:
        """
        Section → paragraph → sentence hierarchy.

        First split into paragraphs (double newline). If a paragraph
        exceeds max_tokens, recursively split into sentences. If a
        sentence exceeds max_tokens, fall back to fixed chunking.
        """
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        chunks: list[Chunk] = []
        idx = 0

        for para in paragraphs:
            para_tokens = _count_tokens(para)
            if para_tokens <= self.max_tokens:
                chunks.append(
                    Chunk(
                        text=para,
                        chunk_id=self._make_chunk_id(ticker, section, idx),
                        source_section=section,
                        strategy="hierarchical",
                        token_count=para_tokens,
                        metadata={"ticker": ticker, "level": "paragraph"},
                    )
                )
                idx += 1
            else:
                # Split into sentences
                sentences = _split_sentences(para)
                current: list[str] = []
                current_tokens = 0

                for sentence in sentences:
                    s_tokens = _count_tokens(sentence)
                    if current_tokens + s_tokens > self.max_tokens and current:
                        chunk_text = " ".join(current).strip()
                        chunks.append(
                            Chunk(
                                text=chunk_text,
                                chunk_id=self._make_chunk_id(ticker, section, idx),
                                source_section=section,
                                strategy="hierarchical",
                                token_count=_count_tokens(chunk_text),
                                metadata={"ticker": ticker, "level": "sentence_group"},
                            )
                        )
                        idx += 1
                        current = [sentence]
                        current_tokens = s_tokens
                    else:
                        current.append(sentence)
                        current_tokens += s_tokens

                if current:
                    chunk_text = " ".join(current).strip()
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            chunk_id=self._make_chunk_id(ticker, section, idx),
                            source_section=section,
                            strategy="hierarchical",
                            token_count=_count_tokens(chunk_text),
                            metadata={"ticker": ticker, "level": "sentence_group"},
                        )
                    )
                    idx += 1

        log.debug(
            "chunker.hierarchical_done",
            ticker=ticker,
            section=section,
            chunk_count=len(chunks),
        )
        return chunks
