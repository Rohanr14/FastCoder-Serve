"""Open-loop (arrival-rate) load testing and analysis.

The concurrency sweep is *closed-loop*: offered load adapts to server speed. Open-loop instead
fires requests at a fixed **arrival rate** (Poisson inter-arrivals) regardless of response time, so
the server's queue absorbs overload. Sweeping the rate exposes queueing latency, the
latency-vs-throughput knee, overload (latency divergence past capacity), and SLO-violation curves.

Each request is fire-and-forget at its scheduled arrival time, so a slow request never delays the
next arrival — this avoids coordinated omission. Latency is measured from the *scheduled* arrival,
so it includes any queueing the server (or an overwhelmed client) introduces.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from fastcoder.benchmark.client import OpenAICompatibleClient
from fastcoder.benchmark.config import BenchmarkConfig
from fastcoder.benchmark.metrics import mean, percentile
from fastcoder.benchmark.runner import capture_and_apply_vllm_version
from fastcoder.benchmark.workloads import Workload, get_workloads

OPEN_LOOP_SCHEMA_VERSION = "openloop-0.1"


def poisson_arrival_offsets(rate_rps: float, count: int, seed: int) -> list[float]:
    """Cumulative arrival times (s from start) for ``count`` Poisson arrivals at ``rate_rps``."""

    if rate_rps <= 0:
        msg = "rate_rps must be > 0"
        raise ValueError(msg)
    rng = random.Random(seed)
    offsets: list[float] = []
    clock = 0.0
    for _ in range(count):
        clock += rng.expovariate(rate_rps)
        offsets.append(clock)
    return offsets


@dataclass(frozen=True, slots=True)
class OpenLoopRequest:
    """One open-loop request: scheduled arrival (s from start) and how it resolved."""

    arrival_offset: float
    latency_seconds: float | None
    ttft_seconds: float | None
    ok: bool


def _violates_slo(
    request: OpenLoopRequest, slo_ttft_seconds: float, slo_latency_seconds: float | None
) -> bool:
    if not request.ok or request.latency_seconds is None or request.ttft_seconds is None:
        return True
    if request.ttft_seconds > slo_ttft_seconds:
        return True
    return slo_latency_seconds is not None and request.latency_seconds > slo_latency_seconds


def summarize_open_loop(
    *,
    target_rate_rps: float,
    requests: Sequence[OpenLoopRequest],
    slo_ttft_seconds: float,
    slo_latency_seconds: float | None = None,
) -> dict[str, Any]:
    """Aggregate one arrival-rate point into latency/TTFT percentiles + SLO-violation rate."""

    ok = [r for r in requests if r.ok and r.latency_seconds is not None]
    latencies = [r.latency_seconds for r in ok if r.latency_seconds is not None]
    ttfts = [r.ttft_seconds for r in ok if r.ttft_seconds is not None]
    total = len(requests)

    completions = [
        r.arrival_offset + r.latency_seconds for r in ok if r.latency_seconds is not None
    ]
    arrivals = [r.arrival_offset for r in requests]
    wall = (max(completions) - min(arrivals)) if completions and arrivals else 0.0
    violations = sum(1 for r in requests if _violates_slo(r, slo_ttft_seconds, slo_latency_seconds))

    return {
        "target_rate_rps": target_rate_rps,
        "request_count": total,
        "success_count": len(ok),
        "error_count": total - len(ok),
        "achieved_throughput_rps": (len(ok) / wall) if wall > 0 else None,
        "wall_time_seconds": wall,
        "slo_ttft_seconds": slo_ttft_seconds,
        "slo_latency_seconds": slo_latency_seconds,
        "slo_violation_rate": (violations / total) if total else None,
        "p50_latency_seconds": percentile(latencies, 50),
        "p95_latency_seconds": percentile(latencies, 95),
        "p99_latency_seconds": percentile(latencies, 99),
        "mean_latency_seconds": mean(latencies),
        "p50_ttft_seconds": percentile(ttfts, 50),
        "p95_ttft_seconds": percentile(ttfts, 95),
        "p99_ttft_seconds": percentile(ttfts, 99),
    }


def validate_open_loop_result(result: dict[str, Any]) -> list[str]:
    """Light sanity checks on an open-loop result (returns a list of problems; empty == valid)."""

    issues: list[str] = []
    if result.get("schema_version") != OPEN_LOOP_SCHEMA_VERSION:
        issues.append("schema_version is not " + OPEN_LOOP_SCHEMA_VERSION)
    points = result.get("arrival_rate_points")
    if not isinstance(points, list) or not points:
        issues.append("no arrival_rate_points")
        return issues
    for point in points:
        rate = point.get("target_rate_rps")
        violation = point.get("slo_violation_rate")
        if violation is not None and not 0.0 <= violation <= 1.0:
            issues.append(f"rate {rate}: slo_violation_rate outside [0, 1]")
        counts = point.get("success_count", 0) + point.get("error_count", 0)
        if counts != point.get("request_count"):
            issues.append(f"rate {rate}: success+error != request_count")
        lat = [point.get(k) for k in ("p50_latency_seconds", "p95_latency_seconds",
                                      "p99_latency_seconds")]
        present = [x for x in lat if isinstance(x, int | float)]
        if present != sorted(present):
            issues.append(f"rate {rate}: latency percentiles not monotonic")
    return issues


async def _fire_request(
    client: OpenAICompatibleClient,
    config: BenchmarkConfig,
    workload: Workload,
    index: int,
    base_perf: float,
    arrival_offset: float,
    max_tokens: int,
) -> OpenLoopRequest:
    sample = workload.samples[index % len(workload.samples)]
    delay = (base_perf + arrival_offset) - time.perf_counter()
    if delay > 0:
        await asyncio.sleep(delay)
    arrival_perf = time.perf_counter()
    metric = await client.chat_completion(
        experiment=config.experiment_name,
        workload=workload.name,
        concurrency=0,
        request_index=index,
        messages=sample.messages,
        max_tokens=max_tokens,
        temperature=config.temperature,
        stream=config.stream,
    )
    latency = (metric.end_time - arrival_perf) if metric.ok else None
    ttft = (
        (metric.first_token_time - arrival_perf)
        if metric.ok and metric.first_token_time is not None
        else None
    )
    return OpenLoopRequest(
        arrival_offset=arrival_offset, latency_seconds=latency, ttft_seconds=ttft, ok=metric.ok
    )


async def run_open_loop_point(
    config: BenchmarkConfig, workload: Workload, *, rate_rps: float, requests: int, seed: int
) -> list[OpenLoopRequest]:
    """Fire ``requests`` Poisson-timed requests at ``rate_rps`` — fire-and-forget, no cap."""

    offsets = poisson_arrival_offsets(rate_rps, requests, seed)
    max_tokens = (
        config.max_tokens if workload.max_output_tokens is None else workload.max_output_tokens
    )
    async with OpenAICompatibleClient(config.model_server) as client:
        base = time.perf_counter()
        tasks = [
            _fire_request(client, config, workload, index, base, offsets[index], max_tokens)
            for index in range(requests)
        ]
        return list(await asyncio.gather(*tasks))


async def run_open_loop_sweep(config: BenchmarkConfig) -> dict[str, Any]:
    """Run the arrival-rate sweep and return a result dict (schema ``openloop-0.1``)."""

    settings = config.open_loop
    if not settings.arrival_rates_rps:
        msg = "open_loop.arrival_rates_rps is empty; nothing to sweep"
        raise ValueError(msg)

    workload = get_workloads([config.workloads[0]])[0]
    config = await capture_and_apply_vllm_version(config)
    slo_ttft = settings.slo_ttft_ms / 1000.0
    slo_latency = settings.slo_latency_ms / 1000.0 if settings.slo_latency_ms is not None else None

    points: list[dict[str, Any]] = []
    for rate in settings.arrival_rates_rps:
        records = await run_open_loop_point(
            config, workload, rate_rps=rate, requests=settings.requests_per_rate, seed=settings.seed
        )
        points.append(
            summarize_open_loop(
                target_rate_rps=rate,
                requests=records,
                slo_ttft_seconds=slo_ttft,
                slo_latency_seconds=slo_latency,
            )
        )

    return {
        "schema_version": OPEN_LOOP_SCHEMA_VERSION,
        "experiment_name": config.experiment_name,
        "created_at": datetime.now(UTC).isoformat(),
        "model": config.model_server.model,
        "serving_backend": config.serving_backend,
        "vllm_version": config.vllm_version,
        "workload": workload.name,
        "requests_per_rate": settings.requests_per_rate,
        "slo_ttft_seconds": slo_ttft,
        "slo_latency_seconds": slo_latency,
        "hardware": config.hardware.model_dump(mode="json"),
        "arrival_rate_points": points,
    }
