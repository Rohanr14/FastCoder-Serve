from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from fastcoder.benchmark.config import BenchmarkConfig
from fastcoder.benchmark.metrics import TOKEN_SOURCE_USAGE, RequestMetric
from fastcoder.benchmark.runner import _run_workload_at_concurrency
from fastcoder.benchmark.workloads import Workload, WorkloadSample


class _RecordingClient:
    """Stub client that records call kwargs instead of issuing HTTP requests."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat_completion(
        self,
        *,
        experiment: str,
        workload: str,
        concurrency: int,
        request_index: int,
        messages: Sequence[Mapping[str, str]],
        max_tokens: int,
        temperature: float,
        stream: bool,
        sample_id: str | None = None,
    ) -> RequestMetric:
        self.calls.append(
            {
                "request_index": request_index,
                "sample_id": sample_id,
                "content": messages[-1]["content"],
                "max_tokens": max_tokens,
            }
        )
        return RequestMetric(
            experiment=experiment,
            workload=workload,
            concurrency=concurrency,
            request_id=f"{workload}-{request_index}",
            ok=True,
            status_code=200,
            start_time=0.0,
            end_time=0.01,
            first_token_time=None,
            output_tokens=1,
            output_token_source=TOKEN_SOURCE_USAGE,
            sample_id=sample_id,
        )


def _three_sample_workload(max_output_tokens: int | None) -> Workload:
    samples = tuple(
        WorkloadSample(
            messages=({"role": "user", "content": f"prompt-{i}"},),
            sample_id=f"w/{i}",
        )
        for i in range(3)
    )
    return Workload(
        name="w",
        description="d",
        samples=samples,
        max_output_tokens=max_output_tokens,
    )


def test_runner_cycles_samples_and_prefers_workload_max_tokens() -> None:
    client = _RecordingClient()
    config = BenchmarkConfig(
        experiment_name="t",
        workloads=["short_chat"],
        max_tokens=999,
        requests_per_workload_per_concurrency=5,
        concurrency=[1],
    )

    records = asyncio.run(
        _run_workload_at_concurrency(
            client=client,  # type: ignore[arg-type]
            config=config,
            workload=_three_sample_workload(max_output_tokens=128),
            concurrency=1,
        )
    )

    assert len(records) == 5
    by_index = sorted(client.calls, key=lambda call: call["request_index"])
    assert [call["sample_id"] for call in by_index] == ["w/0", "w/1", "w/2", "w/0", "w/1"]
    assert [call["content"] for call in by_index] == [
        "prompt-0",
        "prompt-1",
        "prompt-2",
        "prompt-0",
        "prompt-1",
    ]
    # Workload override (128) wins over config.max_tokens (999).
    assert all(call["max_tokens"] == 128 for call in by_index)


def test_runner_falls_back_to_config_max_tokens_without_override() -> None:
    client = _RecordingClient()
    config = BenchmarkConfig(
        experiment_name="t",
        workloads=["short_chat"],
        max_tokens=42,
        requests_per_workload_per_concurrency=2,
        concurrency=[1],
    )

    asyncio.run(
        _run_workload_at_concurrency(
            client=client,  # type: ignore[arg-type]
            config=config,
            workload=_three_sample_workload(max_output_tokens=None),
            concurrency=1,
        )
    )

    assert all(call["max_tokens"] == 42 for call in client.calls)
