# src/knowledge_engine/config.py
"""Central configuration via pydantic-settings — all env vars read here."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Anthropic
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    sonnet_model: str = "claude-sonnet-4-6"
    haiku_model: str = "claude-haiku-4-5-20251001"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_minilm: str = "edgar_minilm"
    qdrant_collection_bge: str = "edgar_bge"

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "edgar"
    postgres_user: str = "edgar"
    postgres_password: str = Field(default="", description="PostgreSQL password")

    # vLLM
    vllm_base_url: str = "http://localhost:8000"
    vllm_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    enable_gpu_ops: bool = False

    # LangSmith
    langsmith_api_key: str = Field(default="", description="LangSmith API key")
    langchain_project: str = "self-optimizing-knowledge-engine"
    langchain_tracing_v2: bool = False

    # Braintrust
    braintrust_api_key: str = Field(default="", description="Braintrust API key")

    # Chunking
    chunking_strategy: Literal["semantic", "fixed", "hierarchical"] = "fixed"
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 50

    # Retrieval
    dense_top_k: int = 10
    bm25_top_k: int = 10
    rrf_k: int = 60
    rerank_top_n: int = 5
    embedding_model_minilm: str = "all-MiniLM-L6-v2"
    embedding_model_bge: str = "BAAI/bge-large-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # EDGAR
    edgar_companies: list[str] = Field(
        default=["AAPL", "MSFT", "AMZN", "GOOGL", "META"],
        description="Ticker symbols to fetch 10-K filings for",
    )
    edgar_user_agent: str = "KnowledgeEngine/0.1 research@example.com"

    # Evaluation
    regression_threshold: float = 0.05  # 5% max drop
    results_baseline_path: str = "results/baseline_metrics.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance (constructed once per process)."""
    return Settings()


def configure_logging() -> None:
    """Configure structlog for JSON output with service context."""
    import logging as stdlib_logging
    raw_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(stdlib_logging, raw_level, stdlib_logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )
