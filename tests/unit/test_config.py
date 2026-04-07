# tests/unit/test_config.py
"""Tests for configuration loading."""
import os
from unittest.mock import patch

from knowledge_engine.config import Settings, get_settings


def test_default_settings() -> None:
    s = Settings()
    assert s.chunking_strategy == "fixed"
    assert s.chunk_size_tokens == 512
    assert s.dense_top_k == 10
    assert s.rrf_k == 60
    assert s.rerank_top_n == 5
    assert s.enable_gpu_ops is False


def test_env_override() -> None:
    with patch.dict(os.environ, {"CHUNKING_STRATEGY": "semantic", "ENABLE_GPU_OPS": "1"}):
        s = Settings()
        assert s.chunking_strategy == "semantic"
        assert s.enable_gpu_ops is True


def test_get_settings_returns_settings() -> None:
    s = get_settings()
    assert isinstance(s, Settings)
