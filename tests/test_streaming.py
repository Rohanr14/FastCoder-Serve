from __future__ import annotations

import asyncio
import socket
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest
import uvicorn

from fastcoder.benchmark.config import BenchmarkConfig, ModelServerConfig
from fastcoder.benchmark.runner import run_benchmark_config
from fastcoder.gateway.app import create_app
from fastcoder.gateway.config import GatewaySettings
from scripts.fake_openai_server import app as fake_app


@pytest.mark.asyncio
async def test_gateway_streams_sse_without_buffering() -> None:
    fake_port = _free_port()
    gateway_port = _free_port()
    gateway_app = create_app(
        GatewaySettings(
            api_key="test-token",
            upstream_base_url=f"http://127.0.0.1:{fake_port}/v1",
            auth_bypass=False,
            rate_limit_per_minute=100,
        )
    )

    async with _serve(fake_app, fake_port), _serve(gateway_app, gateway_port):
        line_times: list[float] = []
        data_lines: list[str] = []
        async with httpx.AsyncClient(timeout=5.0) as client:
            async with client.stream(
                "POST",
                f"http://127.0.0.1:{gateway_port}/v1/chat/completions",
                headers={"authorization": "Bearer test-token"},
                json={
                    "model": "fake-coder",
                    "messages": [{"role": "user", "content": "What does a Python dict do?"}],
                    "stream": True,
                    "stream_delay_seconds": 0.02,
                },
            ) as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data_lines.append(line)
                        line_times.append(time.perf_counter())

    assert len(data_lines) >= 4
    assert data_lines[-1] == "data: [DONE]"
    assert line_times[-1] - line_times[0] >= 0.05


@pytest.mark.asyncio
async def test_streaming_benchmark_result_has_ttft_and_itl() -> None:
    fake_port = _free_port()
    config = BenchmarkConfig(
        experiment_name="test_streaming",
        model_server=ModelServerConfig(base_url=f"http://127.0.0.1:{fake_port}/v1"),
        workloads=["short_chat"],
        concurrency=[1],
        requests_per_workload_per_concurrency=1,
        stream=True,
    )

    async with _serve(fake_app, fake_port):
        result = await run_benchmark_config(config)

    aggregate = result["aggregate_metrics"]
    request = result["per_request_metrics"][0]
    assert result["schema_version"] == "0.1"
    assert aggregate["p50_ttft_seconds"] is not None
    assert aggregate["p50_itl_seconds"] is not None
    assert request["ttft_seconds"] is not None
    assert request["mean_itl_seconds"] is not None
    assert len(request["chunk_arrival_offsets_seconds"]) >= 2


@asynccontextmanager
async def _serve(app: object, port: int) -> AsyncIterator[None]:
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    try:
        await _wait_for_health(port)
        yield
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=5)


async def _wait_for_health(port: int) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + 5.0
    async with httpx.AsyncClient(timeout=1.0) as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.05)
    raise TimeoutError(f"server did not become healthy at {url}")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
