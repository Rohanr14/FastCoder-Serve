"""Async benchmark runner."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastcoder.benchmark.client import OpenAICompatibleClient
from fastcoder.benchmark.config import BenchmarkConfig, load_benchmark_config
from fastcoder.benchmark.endpoint import fetch_vllm_version
from fastcoder.benchmark.gpu import collect_gpu_snapshot
from fastcoder.benchmark.metrics import RequestMetric
from fastcoder.benchmark.results import build_benchmark_result
from fastcoder.benchmark.workloads import Workload, get_workloads


async def run_benchmark_config(config: BenchmarkConfig) -> dict[str, Any]:
    """Run all configured workloads and return a JSON-serializable result object."""

    workloads = get_workloads(config.workloads)
    config = await capture_and_apply_vllm_version(config)
    all_records: list[RequestMetric] = []
    async with OpenAICompatibleClient(config.model_server) as client:
        for workload in workloads:
            for concurrency in config.concurrency:
                records = await _run_workload_at_concurrency(
                    client=client,
                    config=config,
                    workload=workload,
                    concurrency=concurrency,
                )
                all_records.extend(records)

    result = build_benchmark_result(
        config=config,
        workloads=workloads,
        records=all_records,
        gpu_snapshot=collect_gpu_snapshot(),
    )
    return result.model_dump(mode="json")


async def run_benchmark_file(path: str | Path) -> dict[str, Any]:
    """Load, run, and persist a benchmark config file."""

    config = load_benchmark_config(path)
    result = await run_benchmark_config(config)
    output_path = config.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


async def capture_and_apply_vllm_version(config: BenchmarkConfig) -> BenchmarkConfig:
    """Read the live server version and fold it into the config.

    The live server is the ground truth for what actually served the requests, so a captured
    version overrides any config value; a missing ``/version`` route leaves the config as-is.
    """

    captured_version = await fetch_vllm_version(
        base_url=config.model_server.base_url,
        api_key=config.model_server.api_key,
        timeout_seconds=min(config.model_server.timeout_seconds, 10.0),
    )
    return _with_captured_vllm_version(config, captured_version)


def _with_captured_vllm_version(config: BenchmarkConfig, version: str | None) -> BenchmarkConfig:
    if version is None:
        return config
    software = config.software.model_copy(update={"vllm_version": version})
    return config.model_copy(update={"software": software, "vllm_version": version})


async def _run_workload_at_concurrency(
    *,
    client: OpenAICompatibleClient,
    config: BenchmarkConfig,
    workload: Workload,
    concurrency: int,
) -> list[RequestMetric]:
    semaphore = asyncio.Semaphore(concurrency)
    request_count = max(config.requests_per_workload_per_concurrency, concurrency)
    samples = workload.samples
    max_tokens = (
        config.max_tokens if workload.max_output_tokens is None else workload.max_output_tokens
    )

    async def run_one(request_index: int) -> RequestMetric:
        sample = samples[request_index % len(samples)]
        async with semaphore:
            return await client.chat_completion(
                experiment=config.experiment_name,
                workload=workload.name,
                concurrency=concurrency,
                request_index=request_index,
                messages=sample.messages,
                sample_id=sample.sample_id,
                max_tokens=max_tokens,
                temperature=config.temperature,
                stream=config.stream,
            )

    return await asyncio.gather(*(run_one(index) for index in range(request_count)))
