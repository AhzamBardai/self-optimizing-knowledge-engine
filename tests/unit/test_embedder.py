"""Unit tests for embedder — no network, no Qdrant."""
from knowledge_engine.retrieval.embedder import embed_texts, get_vector_dimension


def test_embed_texts_returns_correct_count() -> None:
    texts = ["Apple revenue 2023", "Microsoft Azure cloud services"]
    result = embed_texts(texts, model_name="all-MiniLM-L6-v2")
    assert len(result) == 2


def test_embed_texts_returns_correct_dimension() -> None:
    texts = ["Test query"]
    result = embed_texts(texts, model_name="all-MiniLM-L6-v2")
    assert len(result[0]) == 384


def test_get_vector_dimension_minilm() -> None:
    assert get_vector_dimension("all-MiniLM-L6-v2") == 384


def test_get_vector_dimension_bge() -> None:
    assert get_vector_dimension("BAAI/bge-large-en-v1.5") == 1024


def test_embeddings_are_normalized() -> None:
    import math
    texts = ["Some financial document text about revenue"]
    result = embed_texts(texts, model_name="all-MiniLM-L6-v2")
    norm = math.sqrt(sum(x**2 for x in result[0]))
    assert abs(norm - 1.0) < 0.01, f"Expected unit norm, got {norm:.4f}"
