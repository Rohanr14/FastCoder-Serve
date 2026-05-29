from __future__ import annotations

import asyncio
import socket
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn

from fastcoder.benchmark.config import BenchmarkConfig, ModelServerConfig
from fastcoder.benchmark.humaneval_runner import run_humaneval_eval
from fastcoder.benchmark.result_validation import validate_result_sanity
from fastcoder.benchmark.results import BenchmarkResult
from scripts.fake_openai_server import app as fake_app

_FIXTURE = Path(__file__).parent / "data" / "humaneval_sample.jsonl"


def _checker_passes_when_return_present(
    problem: dict[str, Any],
    completion: str,
    timeout: float,
    completion_id: int | None = None,
) -> dict[str, Any]:
    passed = "return" in completion
    return {
        "task_id": problem["task_id"],
        "passed": passed,
        "result": "passed" if passed else "failed: no return",
        "completion_id": completion_id,
    }


@pytest.mark.parametrize("stream", [False, True])
async def test_run_humaneval_eval_captures_text_and_scores_pass_at_1(stream: bool) -> None:
    port = _free_port()
    config = BenchmarkConfig(
        experiment_name="humaneval_eval_test",
        model_server=ModelServerConfig(base_url=f"http://127.0.0.1:{port}/v1", model="fake-coder"),
        workloads=["humaneval"],
        concurrency=[2],
        requests_per_workload_per_concurrency=1,
        max_tokens=512,
        temperature=0.0,
        stream=stream,
    )

    async with _serve(fake_app, port):
        result = await run_humaneval_eval(
            config,
            humaneval_path=str(_FIXTURE),
            check_correctness=_checker_passes_when_return_present,
        )

    aggregate = result["aggregate_metrics"]
    # Fixture has 2 problems; the fake server answers HumanEval/0 with "return a + b" (passes the
    # injected checker) and HumanEval/1 with non-code prose (fails) -> pass@1 == 0.5.
    assert aggregate["humaneval_pass_at_1"] == pytest.approx(0.5)
    assert aggregate["request_count"] == 2
    assert aggregate["success_count"] == 2

    parsed = BenchmarkResult.model_validate(result)
    assert validate_result_sanity(parsed) == []
    # Raw model output must never leak into the persisted per-request schema.
    assert all(not hasattr(record, "response_text") for record in parsed.per_request_metrics)


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
