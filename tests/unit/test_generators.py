"""Unit tests for generators — all external calls are mocked."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from knowledge_engine.generation.base import GenerationResult
from knowledge_engine.generation.slm_generator import SLMGenerator
from knowledge_engine.generation.sonnet_generator import SonnetGenerator

SAMPLE_QUESTION = "What was Apple's revenue in fiscal year 2023?"
SAMPLE_CHUNKS = [
    "Apple Inc. reported total net sales of $383.3 billion for fiscal year 2023.",
    "This represents a decrease of 3% compared to fiscal year 2022.",
]
EXPECTED_ANSWER = "Apple's revenue was $383.3 billion in fiscal year 2023."


class TestSonnetGenerator:
    @patch("knowledge_engine.generation.sonnet_generator.anthropic.Anthropic")
    def test_generate_returns_result(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=EXPECTED_ANSWER)]
        mock_message.usage.input_tokens = 150
        mock_message.usage.output_tokens = 30
        mock_client.messages.create.return_value = mock_message

        gen = SonnetGenerator(api_key="test-key")
        result = gen.generate(SAMPLE_QUESTION, SAMPLE_CHUNKS)

        assert isinstance(result, GenerationResult)
        assert result.answer == EXPECTED_ANSWER
        assert result.model == "claude-sonnet-4-6"
        assert result.prompt_tokens == 150
        assert result.completion_tokens == 30
        assert result.cost_usd > 0
        assert result.latency_ms >= 0

    @patch("knowledge_engine.generation.sonnet_generator.anthropic.Anthropic")
    def test_generate_calls_api_with_system_prompt(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="answer")]
        mock_message.usage.input_tokens = 100
        mock_message.usage.output_tokens = 20
        mock_client.messages.create.return_value = mock_message

        gen = SonnetGenerator(api_key="test-key")
        gen.generate(SAMPLE_QUESTION, SAMPLE_CHUNKS)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "financial analyst" in call_kwargs["system"].lower()
        assert call_kwargs["model"] == "claude-sonnet-4-6"


class TestSLMGenerator:
    @patch("knowledge_engine.generation.slm_generator.httpx.Client")
    def test_generate_returns_result(self, mock_httpx_cls: MagicMock) -> None:
        mock_client = MagicMock().__enter__.return_value
        mock_httpx_cls.return_value.__enter__.return_value = mock_client

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": EXPECTED_ANSWER}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 40},
        }
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response

        gen = SLMGenerator(base_url="http://localhost:8000")
        result = gen.generate(SAMPLE_QUESTION, SAMPLE_CHUNKS)

        assert isinstance(result, GenerationResult)
        assert result.answer == EXPECTED_ANSWER
        assert result.cost_usd == 0.0001

    def test_build_user_prompt_includes_context(self) -> None:
        gen = SLMGenerator()
        prompt = gen._build_user_prompt(SAMPLE_QUESTION, SAMPLE_CHUNKS)
        assert SAMPLE_QUESTION in prompt
        assert SAMPLE_CHUNKS[0] in prompt
        assert SAMPLE_CHUNKS[1] in prompt
