"""
vLLM-hosted fine-tuned Llama 3.1 8B generator.

In test/CI environments, connects to the stub FastAPI server.
In production (ENABLE_GPU_OPS=1), connects to the real vLLM server
with the LoRA adapter loaded.
"""
from __future__ import annotations

import time

import httpx
import structlog

from knowledge_engine.generation.base import BaseGenerator, GenerationResult

log = structlog.get_logger()

# vLLM self-hosted cost model: ~$0.0001/query at typical GPU pricing
VLLM_COST_PER_QUERY = 0.0001


class SLMGenerator(BaseGenerator):
    """Generates answers using vLLM-served Llama 3.1 8B + LoRA."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        max_tokens: int = 1024,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout_seconds

    def generate(self, question: str, context_chunks: list[str]) -> GenerationResult:
        """Call vLLM OpenAI-compatible API and return structured result."""
        user_prompt = self._build_user_prompt(question, context_chunks)
        start = time.monotonic()

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.1,
        }

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()

        latency_ms = (time.monotonic() - start) * 1000
        answer = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        log.info(
            "slm.generated",
            question_preview=question[:60],
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=round(latency_ms, 1),
        )

        return GenerationResult(
            answer=answer,
            model=self.model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            cost_usd=VLLM_COST_PER_QUERY,
            latency_ms=latency_ms,
        )
