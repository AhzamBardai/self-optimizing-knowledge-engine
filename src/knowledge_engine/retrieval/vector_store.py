"""
Qdrant vector store: create collections, upsert chunks, search.

Two collections are maintained:
  - edgar_minilm: all-MiniLM-L6-v2 (384-dim)
  - edgar_bge:    BAAI/bge-large-en-v1.5 (1024-dim)
"""
from __future__ import annotations

from typing import Any

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from knowledge_engine.ingestion.chunker import Chunk
from knowledge_engine.retrieval.embedder import EmbeddingModelName, embed_texts, get_vector_dimension

log = structlog.get_logger()


class VectorStore:
    """Manages Qdrant collections for two embedding models."""

    def __init__(self, host: str = "localhost", port: int = 6333) -> None:
        self.client = QdrantClient(host=host, port=port)

    def ensure_collection(
        self,
        collection_name: str,
        model_name: EmbeddingModelName,
    ) -> None:
        """Create collection if it does not exist."""
        existing = {c.name for c in self.client.get_collections().collections}
        if collection_name not in existing:
            dim = get_vector_dimension(model_name)
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            log.info(
                "vector_store.collection_created",
                collection=collection_name,
                dim=dim,
            )

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        collection_name: str,
        model_name: EmbeddingModelName,
        batch_size: int = 64,
    ) -> int:
        """
        Embed and upsert chunks into Qdrant.

        Returns number of points upserted.
        """
        self.ensure_collection(collection_name, model_name)
        texts = [c.text for c in chunks]
        vectors = embed_texts(texts, model_name=model_name, batch_size=batch_size)

        points = [
            PointStruct(
                id=abs(hash(chunk.chunk_id)) % (2**31),
                vector=vector,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "source_section": chunk.source_section,
                    "strategy": chunk.strategy,
                    "token_count": chunk.token_count,
                    **chunk.metadata,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=collection_name, points=batch)
            log.debug(
                "vector_store.batch_upserted",
                collection=collection_name,
                batch_start=i,
                count=len(batch),
            )

        log.info(
            "vector_store.upsert_complete",
            collection=collection_name,
            total=len(points),
        )
        return len(points)

    def search(
        self,
        query: str,
        collection_name: str,
        model_name: EmbeddingModelName,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Dense ANN search.

        Returns list of dicts with 'chunk_id', 'text', 'score', and payload fields.
        """
        query_vector = embed_texts([query], model_name=model_name)[0]
        results: list[ScoredPoint] = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "chunk_id": r.payload.get("chunk_id", str(r.id)) if r.payload else str(r.id),
                "text": r.payload.get("text", "") if r.payload else "",
                "score": r.score,
                **(r.payload or {}),
            }
            for r in results
        ]

    def get_collection_count(self, collection_name: str) -> int:
        """Return number of points in a collection."""
        info = self.client.get_collection(collection_name)
        return info.points_count or 0
