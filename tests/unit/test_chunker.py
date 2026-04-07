# tests/unit/test_chunker.py
"""
Unit tests for the three chunking strategies.

These tests run with zero external dependencies — no Qdrant, no network.
"""
from pathlib import Path

import pytest

from knowledge_engine.ingestion.chunker import Chunk, Chunker

SAMPLE_TEXT = (
    "Apple Inc. reported total net sales of $383.3 billion for fiscal year 2023. "
    "This represents a 3% decrease compared to the prior year. "
    "Products net sales were $298.1 billion. "
    "Services net sales were $85.2 billion, growing 5% year-over-year. "
    "The company maintained strong gross margins of 44.1% in fiscal 2023. "
    "iPhone revenue was $200.6 billion, representing 52% of total net sales. "
    "Mac revenue was $29.4 billion. iPad revenue was $28.3 billion. "
    "Wearables, Home and Accessories revenue was $39.8 billion. "
    "\n\n"
    "The company's operating income was $114.3 billion. "
    "Net income was $97.0 billion, or $6.13 per diluted share. "
    "Cash and cash equivalents totaled $29.9 billion at fiscal year end. "
    "\n\n"
    "Research and development expenses were $29.9 billion, up 14% year-over-year. "
    "Capital expenditures were $10.7 billion for property, plant and equipment. "
    "The company returned $89.3 billion to shareholders through dividends and buybacks. "
)

LONG_TEXT = SAMPLE_TEXT * 8  # ~2000 tokens to trigger multi-chunk behavior


@pytest.fixture
def fixed_chunker() -> Chunker:
    return Chunker(strategy="fixed", max_tokens=128, overlap_tokens=16)


@pytest.fixture
def semantic_chunker() -> Chunker:
    return Chunker(strategy="semantic", max_tokens=200, similarity_threshold=0.5)


@pytest.fixture
def hierarchical_chunker() -> Chunker:
    return Chunker(strategy="hierarchical", max_tokens=128)


class TestFixedChunker:
    def test_produces_chunks(self, fixed_chunker: Chunker) -> None:
        chunks = fixed_chunker.chunk(LONG_TEXT, "AAPL", "mda")
        assert len(chunks) > 1

    def test_no_empty_chunks(self, fixed_chunker: Chunker) -> None:
        chunks = fixed_chunker.chunk(LONG_TEXT, "AAPL", "mda")
        for chunk in chunks:
            assert chunk.text.strip(), f"Empty chunk: {chunk.chunk_id}"

    def test_no_chunk_exceeds_max_tokens(self, fixed_chunker: Chunker) -> None:
        chunks = fixed_chunker.chunk(LONG_TEXT, "AAPL", "mda")
        for chunk in chunks:
            assert chunk.token_count <= fixed_chunker.max_tokens, (
                f"Chunk {chunk.chunk_id} has {chunk.token_count} tokens "
                f"(max {fixed_chunker.max_tokens})"
            )

    def test_chunk_ids_are_unique(self, fixed_chunker: Chunker) -> None:
        chunks = fixed_chunker.chunk(LONG_TEXT, "AAPL", "mda")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_strategy_field(self, fixed_chunker: Chunker) -> None:
        chunks = fixed_chunker.chunk(SAMPLE_TEXT, "AAPL", "business")
        for chunk in chunks:
            assert chunk.strategy == "fixed"

    def test_short_text_produces_single_chunk(self, fixed_chunker: Chunker) -> None:
        short = "Apple revenue was $383 billion."
        chunks = fixed_chunker.chunk(short, "AAPL", "highlights")
        assert len(chunks) == 1


class TestSemanticChunker:
    def test_produces_chunks(self, semantic_chunker: Chunker) -> None:
        chunks = semantic_chunker.chunk(SAMPLE_TEXT, "MSFT", "mda")
        assert len(chunks) >= 1

    def test_no_empty_chunks(self, semantic_chunker: Chunker) -> None:
        chunks = semantic_chunker.chunk(SAMPLE_TEXT, "MSFT", "mda")
        for chunk in chunks:
            assert chunk.text.strip()

    def test_no_chunk_exceeds_max_tokens(self, semantic_chunker: Chunker) -> None:
        chunks = semantic_chunker.chunk(LONG_TEXT, "MSFT", "mda")
        for chunk in chunks:
            assert chunk.token_count <= semantic_chunker.max_tokens + 50, (
                f"Chunk {chunk.chunk_id} exceeds token limit by too much"
            )

    def test_strategy_field(self, semantic_chunker: Chunker) -> None:
        chunks = semantic_chunker.chunk(SAMPLE_TEXT, "MSFT", "business")
        for chunk in chunks:
            assert chunk.strategy == "semantic"


class TestHierarchicalChunker:
    def test_produces_chunks(self, hierarchical_chunker: Chunker) -> None:
        chunks = hierarchical_chunker.chunk(SAMPLE_TEXT, "AMZN", "mda")
        assert len(chunks) >= 1

    def test_no_empty_chunks(self, hierarchical_chunker: Chunker) -> None:
        chunks = hierarchical_chunker.chunk(SAMPLE_TEXT, "AMZN", "mda")
        for chunk in chunks:
            assert chunk.text.strip()

    def test_paragraphs_respected(self, hierarchical_chunker: Chunker) -> None:
        # Two distinct paragraphs should produce at least 2 chunks
        text = "First paragraph content here.\n\nSecond paragraph content here."
        chunks = hierarchical_chunker.chunk(text, "AMZN", "business")
        assert len(chunks) >= 2

    def test_strategy_field(self, hierarchical_chunker: Chunker) -> None:
        chunks = hierarchical_chunker.chunk(SAMPLE_TEXT, "AMZN", "business")
        for chunk in chunks:
            assert chunk.strategy == "hierarchical"


class TestChunkerConfig:
    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            chunker = Chunker(strategy="invalid")  # type: ignore[arg-type]
            chunker.chunk("text", "AAPL", "mda")

    def test_returns_chunk_dataclass(self) -> None:
        chunker = Chunker(strategy="fixed", max_tokens=256)
        chunks = chunker.chunk(SAMPLE_TEXT, "AAPL", "mda")
        assert all(isinstance(c, Chunk) for c in chunks)
