"""Typed benchmark result schema."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from fastcoder.benchmark.config import BenchmarkConfig, dump_config_for_json
from fastcoder.benchmark.gpu import GPUSnapshot
from fastcoder.benchmark.metrics import RequestMetric, compute_summary
from fastcoder.benchmark.workloads import Workload

SCHEMA_VERSION: Literal["0.1"] = "0.1"


class ResultModelInfo(BaseModel):
    name: str
    server: str
    quantization: str | None = None
    dtype: str | None = None
    notes: str | None = None


class ResultHardwareInfo(BaseModel):
    gpu_name: str | None = None
    gpu_memory_gb: float | None = None
    provider: str | None = None
    hourly_cost_usd: float | None = None
    peak_gpu_memory_gb: float | None = None


class ResultSoftwareInfo(BaseModel):
    vllm_version: str | None = None
    serving_backend: str | None = None
    backend_commit: str | None = None


class ResultSpeculationInfo(BaseModel):
    method: str | None = None
    draft_model: str | None = None
    num_speculative_tokens: int | None = None
    acceptance_rate: float | None = None
    notes: str | None = None


class ResultWorkloadInfo(BaseModel):
    name: str
    description: str


class MetricSummary(BaseModel):
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    streaming_request_count: int = 0
    prompt_tokens: int | None = None
    output_tokens: int = 0
    estimated_output_tokens: int = 0
    output_token_count_estimated: bool = False
    wall_time_seconds: float | None = None
    mean_latency_seconds: float | None = None
    p50_latency_seconds: float | None = None
    p95_latency_seconds: float | None = None
    p99_latency_seconds: float | None = None
    mean_ttft_seconds: float | None = None
    p50_ttft_seconds: float | None = None
    p95_ttft_seconds: float | None = None
    p99_ttft_seconds: float | None = None
    mean_itl_seconds: float | None = None
    p50_itl_seconds: float | None = None
    p95_itl_seconds: float | None = None
    p99_itl_seconds: float | None = None
    request_throughput_rps: float | None = None
    output_token_throughput_tps: float | None = None
    peak_gpu_memory_gb: float | None = None
    humaneval_pass_at_1: float | None = None
    cost_per_1m_output_tokens_usd: float | None = None
    token_counting_note: str


class PerWorkloadMetrics(BaseModel):
    workload: str
    concurrency: int
    metrics: MetricSummary


class PerRequestMetrics(BaseModel):
    request_id: str
    workload: str
    concurrency: int
    ok: bool
    status_code: int | None
    stream: bool
    start_time: float
    end_time: float
    first_token_time: float | None
    latency_seconds: float
    ttft_seconds: float | None
    mean_itl_seconds: float | None
    inter_token_latencies_seconds: list[float]
    chunk_arrival_offsets_seconds: list[float]
    prompt_tokens: int | None
    output_tokens: int
    output_token_source: str
    output_token_count_estimated: bool
    model: str
    error: str | None
    response_chars: int


class BenchmarkResult(BaseModel):
    schema_version: Literal["0.1"] = SCHEMA_VERSION
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    config_name: str
    model: ResultModelInfo
    hardware: ResultHardwareInfo
    software: ResultSoftwareInfo
    speculation: ResultSpeculationInfo
    workloads: list[ResultWorkloadInfo]
    aggregate_metrics: MetricSummary
    per_workload_metrics: list[PerWorkloadMetrics]
    per_request_metrics: list[PerRequestMetrics]
    config: dict[str, Any]


def build_benchmark_result(
    *,
    config: BenchmarkConfig,
    workloads: list[Workload],
    records: list[RequestMetric],
    gpu_snapshot: GPUSnapshot,
) -> BenchmarkResult:
    """Build a schema-stable benchmark result object."""

    hourly_cost = _hourly_cost(config)
    peak_gpu_memory_gb = gpu_snapshot.peak_memory_gb
    aggregate_summary = compute_summary(
        records,
        accelerator_hourly_cost_usd=hourly_cost,
        peak_gpu_memory_gb=peak_gpu_memory_gb,
    )

    return BenchmarkResult(
        config_name=config.experiment_name,
        model=ResultModelInfo(
            name=config.model_server.model,
            server=config.model_server.base_url,
            quantization=config.model_server.quantization,
            dtype=config.model_server.dtype,
            notes=config.model_server.notes,
        ),
        hardware=ResultHardwareInfo(
            gpu_name=config.hardware.gpu_name or gpu_snapshot.gpu_name,
            gpu_memory_gb=config.hardware.gpu_memory_gb or gpu_snapshot.gpu_memory_gb,
            provider=config.hardware.provider,
            hourly_cost_usd=hourly_cost,
            peak_gpu_memory_gb=peak_gpu_memory_gb,
        ),
        software=ResultSoftwareInfo(
            vllm_version=config.software.vllm_version,
            serving_backend=config.software.serving_backend,
            backend_commit=config.software.backend_commit,
        ),
        speculation=ResultSpeculationInfo(
            method=config.speculation.method,
            draft_model=config.speculation.draft_model,
            num_speculative_tokens=config.speculation.num_speculative_tokens,
            acceptance_rate=config.speculation.acceptance_rate,
            notes=config.speculation.notes,
        ),
        workloads=[
            ResultWorkloadInfo(name=workload.name, description=workload.description)
            for workload in workloads
        ],
        aggregate_metrics=MetricSummary.model_validate(aggregate_summary),
        per_workload_metrics=_per_workload_metrics(
            records,
            hourly_cost=hourly_cost,
        ),
        per_request_metrics=[_request_metric(record) for record in records],
        config=dump_config_for_json(config),
    )


def _per_workload_metrics(
    records: list[RequestMetric],
    *,
    hourly_cost: float | None,
) -> list[PerWorkloadMetrics]:
    grouped: dict[tuple[str, int], list[RequestMetric]] = defaultdict(list)
    for record in records:
        grouped[(record.workload, record.concurrency)].append(record)

    summaries: list[PerWorkloadMetrics] = []
    for (workload, concurrency), group_records in sorted(grouped.items()):
        summaries.append(
            PerWorkloadMetrics(
                workload=workload,
                concurrency=concurrency,
                metrics=MetricSummary.model_validate(
                    compute_summary(
                        group_records,
                        accelerator_hourly_cost_usd=hourly_cost,
                    )
                ),
            )
        )
    return summaries


def _request_metric(record: RequestMetric) -> PerRequestMetrics:
    return PerRequestMetrics(
        request_id=record.request_id,
        workload=record.workload,
        concurrency=record.concurrency,
        ok=record.ok,
        status_code=record.status_code,
        stream=record.stream,
        start_time=record.start_time,
        end_time=record.end_time,
        first_token_time=record.first_token_time,
        latency_seconds=record.latency_seconds,
        ttft_seconds=record.ttft_seconds,
        mean_itl_seconds=record.mean_itl_seconds,
        inter_token_latencies_seconds=record.inter_token_latencies_seconds,
        chunk_arrival_offsets_seconds=record.chunk_arrival_offsets_seconds,
        prompt_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        output_token_source=record.output_token_source,
        output_token_count_estimated=record.output_token_count_estimated,
        model=record.model,
        error=record.error,
        response_chars=record.response_chars,
    )


def _hourly_cost(config: BenchmarkConfig) -> float | None:
    if config.hardware.hourly_cost_usd is not None:
        return config.hardware.hourly_cost_usd
    return config.cost.accelerator_hourly_cost_usd
