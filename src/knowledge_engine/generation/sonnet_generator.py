"""Claude Sonnet 4.6 generator via Anthropic API."""
from __future__ import annotations

import time

import anthropic
import structlog

from knowledge_engine.generation.base import BaseGenerator, GenerationResult

log = structlog.get_logger()

# Cost per million tokens (Sonnet 4.6 pricing as of Q2 2026)
SONNET_INPUT_COST_PER_M = 3.0
SONNET_OUTPUT_COST_PER_M = 15.0


class SonnetGenerator(BaseGenerator):
    """Generates answers using Claude Sonnet 4.6."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
    ) -> None:
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, question: str, context_chunks: list[str]) -> GenerationResult:
        """Call Claude Sonnet and return structured result."""
        user_prompt = self._build_user_prompt(question, context_chunks)
        start = time.monotonic()

        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        latency_ms = (time.monotonic() - start) * 1000
        answer = message.content[0].text if message.content else ""

        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens
        cost = (
            input_tokens / 1_000_000 * SONNET_INPUT_COST_PER_M
            + output_tokens / 1_000_000 * SONNET_OUTPUT_COST_PER_M
        )

        log.info(
            "sonnet.generated",
            question_preview=question[:60],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(cost, 6),
            latency_ms=round(latency_ms, 1),
        )

        return GenerationResult(
            answer=answer,
            model=self.model,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
        )
