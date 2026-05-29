"""Metrics helpers for benchmark runs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import ceil, floor
from statistics import fmean
from typing import Any

TOKEN_SOURCE_USAGE = "usage"
TOKEN_SOURCE_APPROXIMATE = "approximate_whitespace"


@dataclass(frozen=True, slots=True)
class RequestMetric:
    """Per-request benchmark measurement."""

    experiment: str
    workload: str
    concurrency: int
    request_id: str
    ok: bool
    status_code: int | None
    start_time: float
    end_time: float
    first_token_time: float | None
    output_tokens: int
    output_token_source: str
    sample_id: str | None = None
    input_tokens: int | None = None
    model: str = ""
    error: str | None = None
    response_chars: int = 0
    stream: bool = False
    chunk_arrival_times: tuple[float, ...] = ()
    # Captured only when a caller opts in (e.g. HumanEval scoring). Kept out of to_dict and the
    # persisted result schema so committed result JSONs never carry raw model output.
    response_text: str | None = None

    @property
    def latency_seconds(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def ttft_seconds(self) -> float | None:
        if self.first_token_time is None:
            return None
        return max(0.0, self.first_token_time - self.start_time)

    @property
    def inter_token_latencies_seconds(self) -> list[float]:
        return inter_token_latencies(self.chunk_arrival_times)

    @property
    def mean_itl_seconds(self) -> float | None:
        return mean(self.inter_token_latencies_seconds)

    @property
    def chunk_arrival_offsets_seconds(self) -> list[float]:
        return [max(0.0, arrival - self.start_time) for arrival in self.chunk_arrival_times]

    @property
    def output_token_count_estimated(self) -> bool:
        return self.output_token_source != TOKEN_SOURCE_USAGE

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "workload": self.workload,
            "concurrency": self.concurrency,
            "request_id": self.request_id,
            "sample_id": self.sample_id,
            "ok": self.ok,
            "status_code": self.status_code,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "first_token_time": self.first_token_time,
            "latency_seconds": self.latency_seconds,
            "ttft_seconds": self.ttft_seconds,
            "mean_itl_seconds": self.mean_itl_seconds,
            "inter_token_latencies_seconds": self.inter_token_latencies_seconds,
            "chunk_arrival_offsets_seconds": self.chunk_arrival_offsets_seconds,
            "stream": self.stream,
            "output_tokens": self.output_tokens,
            "output_token_source": self.output_token_source,
            "output_token_count_estimated": self.output_token_count_estimated,
            "input_tokens": self.input_tokens,
            "model": self.model,
            "error": self.error,
            "response_chars": self.response_chars,
        }


def estimate_token_count(text: str) -> int:
    """Approximate token count with a transparent whitespace-based estimate."""

    stripped = text.strip()
    if not stripped:
        return 0
    return len(stripped.split())


def percentile(values: Sequence[float], percentile_value: float) -> float | None:
    """Compute a linear-interpolated percentile without requiring NumPy at runtime."""

    if not values:
        return None
    if percentile_value < 0 or percentile_value > 100:
        msg = "percentile_value must be between 0 and 100"
        raise ValueError(msg)

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (len(sorted_values) - 1) * (percentile_value / 100.0)
    lower_index = floor(rank)
    upper_index = ceil(rank)
    if lower_index == upper_index:
        return sorted_values[int(rank)]
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    weight = rank - lower_index
    return lower_value + ((upper_value - lower_value) * weight)


def mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return fmean(values)


def inter_token_latencies(chunk_arrival_times: Sequence[float]) -> list[float]:
    """Return deltas between streamed token/chunk arrival times."""

    if len(chunk_arrival_times) < 2:
        return []
    return [
        max(0.0, current_time - previous_time)
        for previous_time, current_time in pairwise(chunk_arrival_times)
    ]


def cost_per_1m_output_tokens(
    throughput_output_tokens_per_second: float | None,
    accelerator_hourly_cost_usd: float | None,
) -> float | None:
    """Estimate serving cost from measured output throughput and hourly accelerator cost."""

    if throughput_output_tokens_per_second is None or throughput_output_tokens_per_second <= 0:
        return None
    if accelerator_hourly_cost_usd is None or accelerator_hourly_cost_usd <= 0:
        return None
    output_tokens_per_hour = throughput_output_tokens_per_second * 3600.0
    return accelerator_hourly_cost_usd / (output_tokens_per_hour / 1_000_000.0)


def compute_summary(
    records: Iterable[RequestMetric],
    *,
    accelerator_hourly_cost_usd: float | None = None,
    peak_gpu_memory_gb: float | None = None,
    humaneval_pass_at_1: float | None = None,
) -> dict[str, Any]:
    """Aggregate per-request records into benchmark summary metrics."""

    record_list = list(records)
    ok_records = [record for record in record_list if record.ok]
    error_count = len(record_list) - len(ok_records)
    latencies = [record.latency_seconds for record in ok_records]
    ttfts = [
        ttft
        for record in ok_records
        if (ttft := record.ttft_seconds) is not None
    ]
    itls = [
        itl
        for record in ok_records
        for itl in record.inter_token_latencies_seconds
    ]
    output_tokens = sum(record.output_tokens for record in ok_records)
    estimated_output_tokens = sum(
        record.output_tokens for record in ok_records if record.output_token_count_estimated
    )
    prompt_token_values = [
        record.input_tokens for record in ok_records if record.input_tokens is not None
    ]
    prompt_tokens = sum(prompt_token_values) if prompt_token_values else None

    wall_seconds: float | None = None
    request_throughput: float | None = None
    output_token_throughput: float | None = None
    if ok_records:
        start = min(record.start_time for record in ok_records)
        end = max(record.end_time for record in ok_records)
        wall_seconds = max(0.0, end - start)
        if wall_seconds > 0:
            request_throughput = len(ok_records) / wall_seconds
            output_token_throughput = output_tokens / wall_seconds

    return {
        "request_count": len(record_list),
        "success_count": len(ok_records),
        "error_count": error_count,
        "streaming_request_count": sum(1 for record in ok_records if record.stream),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "output_token_count_estimated": estimated_output_tokens > 0,
        "wall_time_seconds": wall_seconds,
        "mean_latency_seconds": mean(latencies),
        "p50_latency_seconds": percentile(latencies, 50),
        "p95_latency_seconds": percentile(latencies, 95),
        "p99_latency_seconds": percentile(latencies, 99),
        "mean_ttft_seconds": mean(ttfts),
        "p50_ttft_seconds": percentile(ttfts, 50),
        "p95_ttft_seconds": percentile(ttfts, 95),
        "p99_ttft_seconds": percentile(ttfts, 99),
        "mean_itl_seconds": mean(itls),
        "p50_itl_seconds": percentile(itls, 50),
        "p95_itl_seconds": percentile(itls, 95),
        "p99_itl_seconds": percentile(itls, 99),
        "request_throughput_rps": request_throughput,
        "output_token_throughput_tps": output_token_throughput,
        "throughput_output_tokens_per_second": output_token_throughput,
        "peak_gpu_memory_gb": peak_gpu_memory_gb,
        "humaneval_pass_at_1": humaneval_pass_at_1,
        "cost_per_1m_output_tokens_usd": cost_per_1m_output_tokens(
            output_token_throughput,
            accelerator_hourly_cost_usd,
        ),
        "token_counting_note": (
            "Uses API usage.completion_tokens when provided; otherwise uses an "
            "approximate whitespace-based output-token estimate."
        ),
    }
