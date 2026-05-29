"""Dedicated HumanEval eval runner.

Unlike the latency sweep (``run_benchmark_config``), this runs each of the 164 problems exactly
once at a single concurrency, captures the completion text, and scores pass@1. Quality needs one
greedy sample per problem, not a concurrency sweep of repeated prompts.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastcoder.benchmark.client import OpenAICompatibleClient
from fastcoder.benchmark.config import BenchmarkConfig
from fastcoder.benchmark.gpu import collect_gpu_snapshot
from fastcoder.benchmark.humaneval_scoring import score_completions
from fastcoder.benchmark.metrics import RequestMetric
from fastcoder.benchmark.results import build_benchmark_result
from fastcoder.benchmark.runner import capture_and_apply_vllm_version
from fastcoder.benchmark.workloads import (
    Workload,
    load_humaneval_problems_by_task_id,
    load_humaneval_workloads,
)


async def run_humaneval_eval(
    config: BenchmarkConfig,
    *,
    humaneval_path: str | None = None,
    timeout_seconds: float = 10.0,
    check_correctness: Any | None = None,
) -> dict[str, Any]:
    """Run HumanEval once, score pass@1, and return a result dict with pass@1 populated.

    ``check_correctness`` is an injection seam for tests; production passes ``None`` so the real,
    sandboxed scorer is imported lazily (pod-only).
    """

    problems_by_task_id = load_humaneval_problems_by_task_id(humaneval_path)
    workload = load_humaneval_workloads(humaneval_path)[0]

    config = await capture_and_apply_vllm_version(config)
    records, completions_by_task_id = await _run_all_samples_once(config, workload)

    score = score_completions(
        problems_by_task_id=problems_by_task_id,
        completions_by_task_id=completions_by_task_id,
        timeout_seconds=timeout_seconds,
        check_correctness=check_correctness,
    )

    result = build_benchmark_result(
        config=config,
        workloads=[workload],
        records=records,
        gpu_snapshot=collect_gpu_snapshot(),
        humaneval_pass_at_1=score.pass_at_1,
    )
    return result.model_dump(mode="json")


async def _run_all_samples_once(
    config: BenchmarkConfig,
    workload: Workload,
) -> tuple[list[RequestMetric], dict[str, str]]:
    concurrency = max(1, min(config.concurrency))
    semaphore = asyncio.Semaphore(concurrency)
    max_tokens = (
        config.max_tokens if workload.max_output_tokens is None else workload.max_output_tokens
    )
    samples = workload.samples

    async with OpenAICompatibleClient(config.model_server) as client:

        async def run_one(index: int) -> RequestMetric:
            sample = samples[index]
            async with semaphore:
                return await client.chat_completion(
                    experiment=config.experiment_name,
                    workload=workload.name,
                    concurrency=concurrency,
                    request_index=index,
                    messages=sample.messages,
                    sample_id=sample.sample_id,
                    max_tokens=max_tokens,
                    temperature=config.temperature,
                    stream=config.stream,
                    capture_text=True,
                )

        records = await asyncio.gather(*(run_one(index) for index in range(len(samples))))

    completions_by_task_id: dict[str, str] = {}
    for record in records:
        if record.ok and record.sample_id is not None and record.response_text is not None:
            completions_by_task_id[record.sample_id] = record.response_text
    return list(records), completions_by_task_id
