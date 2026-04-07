"""Sentence-transformers embedding wrapper for MiniLM and BGE models."""
from __future__ import annotations

from typing import Literal

import structlog
from sentence_transformers import SentenceTransformer

log = structlog.get_logger()

EmbeddingModelName = Literal["all-MiniLM-L6-v2", "BAAI/bge-large-en-v1.5"]

_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def get_embedder(model_name: EmbeddingModelName) -> SentenceTransformer:
    """Return cached SentenceTransformer model."""
    if model_name not in _MODEL_CACHE:
        log.info("embedder.loading", model=model_name)
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def embed_texts(
    texts: list[str],
    model_name: EmbeddingModelName = "all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> list[list[float]]:
    """
    Embed a list of texts and return float vectors.

    Args:
        texts: Input strings to embed.
        model_name: Which sentence-transformer model to use.
        batch_size: Embedding batch size.

    Returns:
        List of embedding vectors (one per input text).
    """
    model = get_embedder(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 100,
    )
    return [emb.tolist() for emb in embeddings]


def get_vector_dimension(model_name: EmbeddingModelName) -> int:
    """Return the output dimension for a given model."""
    dims: dict[str, int] = {
        "all-MiniLM-L6-v2": 384,
        "BAAI/bge-large-en-v1.5": 1024,
    }
    return dims[model_name]
