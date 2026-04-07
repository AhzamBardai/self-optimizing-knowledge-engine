"""
Stub vLLM-compatible OpenAI API server for testing without GPU.

Returns deterministic fixture responses so tests don't need a real model.
Activated when STUB_MODE=true or ENABLE_GPU_OPS is unset.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="vLLM Stub", version="0.1.0")

STUB_RESPONSE = (
    "Based on the provided context, the answer is: "
    "Apple Inc. reported revenue of $383.3 billion for fiscal year 2023, "
    "representing a decrease of approximately 3% year-over-year. "
    "This is a stub response for testing purposes."
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int = 512
    temperature: float = 0.1


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "mode": "stub"}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest) -> ChatResponse:
    response_text = STUB_RESPONSE
    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=response_text),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=len(" ".join(m.content for m in request.messages).split()),
            completion_tokens=len(response_text.split()),
            total_tokens=len(" ".join(m.content for m in request.messages).split())
            + len(response_text.split()),
        ),
    )


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": "meta-llama/Llama-3.1-8B-Instruct",
                "object": "model",
                "created": int(time.time()),
            }
        ],
    }
