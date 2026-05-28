from __future__ import annotations

import httpx
import pytest

from scripts.fake_openai_server import app


@pytest.mark.asyncio
async def test_fake_server_health_and_completion() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://fake") as client:
        health = await client.get("/health")
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "fake-coder",
                "messages": [{"role": "user", "content": "What does a dict do?"}],
                "max_tokens": 16,
            },
        )

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    body = response.json()
    assert response.status_code == 200
    assert body["choices"][0]["message"]["content"]
    assert body["usage"]["completion_tokens"] > 0


@pytest.mark.asyncio
async def test_fake_server_streaming_completion() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://fake") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "fake-coder",
                "messages": [{"role": "user", "content": "def add(a: int, b: int) -> int:"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    data_lines = [line for line in response.text.splitlines() if line.startswith("data:")]
    assert len(data_lines) >= 3
    assert data_lines[-1] == "data: [DONE]"
    assert "[DONE]" in response.text
