"""Abstract base class for LLM generators."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    """Result from an LLM generation call."""

    answer: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float


class BaseGenerator(ABC):
    """Interface all generators must implement."""

    SYSTEM_PROMPT = (
        "You are a financial analyst specializing in SEC 10-K filings. "
        "Answer questions accurately based solely on the provided context. "
        "If the context does not contain sufficient information, state that clearly. "
        "Show calculations when comparing financial metrics across companies."
    )

    @abstractmethod
    def generate(self, question: str, context_chunks: list[str]) -> GenerationResult:
        """
        Generate an answer given a question and retrieved context.

        Args:
            question: The user's question.
            context_chunks: Retrieved text chunks to ground the answer.

        Returns:
            GenerationResult with answer and metadata.
        """
        ...

    def _build_user_prompt(self, question: str, context_chunks: list[str]) -> str:
        context = "\n\n---\n\n".join(context_chunks)
        return f"Context:\n{context}\n\nQuestion: {question}"
