# 1. Vector Store: Qdrant over Pinecone/Weaviate

Date: 2026-04-06

## Status

Accepted

## Context

The project needs a vector store for embedding-based retrieval. Options: Qdrant (self-hosted), Pinecone (managed SaaS), Weaviate (self-hosted).

## Decision

Use Qdrant. It runs locally via Docker, supports named collections (two collections: `minilm` and `bge-large` for embedding ablations), and is free with no API key. This matches the docker-compose setup and keeps all data local.

## Consequences

No vendor lock-in. Integration tests use Testcontainers to spin up real Qdrant. No cost for testing or development.
