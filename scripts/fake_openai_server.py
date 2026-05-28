"""Local fake OpenAI-compatible server for CPU-safe smoke tests."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="fake-coder")
    messages: list[ChatMessage] = Field(default_factory=list)
    max_tokens: int = Field(default=32, ge=1)
    temperature: float = Field(default=0.0, ge=0)
    stream: bool = Field(default=False)
    stream_delay_seconds: float | None = Field(default=None, ge=0)

    model_config = ConfigDict(extra="ignore")


app = FastAPI(title="FastCoder Fake OpenAI Server", version="0.1.0")


def default_stream_delay_seconds() -> float:
    raw_value = os.getenv("FASTCODER_FAKE_STREAM_DELAY_SECONDS", "0.005")
    try:
        return max(0.0, float(raw_value))
    except ValueError:
        return 0.005


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": "fake-coder",
                "object": "model",
                "created": 0,
                "owned_by": "fastcoder-local",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> Response:
    content = _fake_completion(request)
    if request.stream:
        return StreamingResponse(
            _stream_response(request, content),
            media_type="text/event-stream",
        )
    return JSONResponse(_completion_response(request, content))


def _fake_completion(request: ChatCompletionRequest) -> str:
    last_user = next(
        (message.content for message in reversed(request.messages) if message.role == "user"),
        "",
    )
    if "def add" in last_user:
        content = "return a + b"
    elif "between alpha and gamma" in last_user.lower():
        content = "beta"
    elif "dict" in last_user.lower():
        content = "A Python dict maps keys to values for fast lookup."
    else:
        content = "FastCoder local smoke response."

    words = content.split()
    if len(words) > request.max_tokens:
        content = " ".join(words[: request.max_tokens])
    return content


def _completion_response(request: ChatCompletionRequest, content: str) -> dict[str, Any]:
    created = int(time.time())
    return {
        "id": "chatcmpl-fastcoder-fake",
        "object": "chat.completion",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": _prompt_tokens(request.messages),
            "completion_tokens": _estimate_tokens(content),
            "total_tokens": _prompt_tokens(request.messages) + _estimate_tokens(content),
        },
    }


async def _stream_response(
    request: ChatCompletionRequest,
    content: str,
) -> AsyncIterator[str]:
    created = int(time.time())
    delay_seconds = (
        request.stream_delay_seconds
        if request.stream_delay_seconds is not None
        else default_stream_delay_seconds()
    )
    for chunk in _stream_chunks(content):
        payload = {
            "id": "chatcmpl-fastcoder-fake",
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(payload)}\n\n"
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)

    final_payload = {
        "id": "chatcmpl-fastcoder-fake",
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_payload)}\n\n"
    yield "data: [DONE]\n\n"


def _stream_chunks(content: str) -> list[str]:
    words = content.split()
    if not words:
        return [""]
    return [word if index == 0 else f" {word}" for index, word in enumerate(words)]


def _prompt_tokens(messages: list[ChatMessage]) -> int:
    return sum(_estimate_tokens(message.content) for message in messages)


def _estimate_tokens(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return len(stripped.split())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FastCoder fake OpenAI server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--stream-delay-seconds", type=float, default=None)
    args = parser.parse_args()

    if args.stream_delay_seconds is not None:
        os.environ["FASTCODER_FAKE_STREAM_DELAY_SECONDS"] = str(args.stream_delay_seconds)

    import uvicorn

    uvicorn.run("scripts.fake_openai_server:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
