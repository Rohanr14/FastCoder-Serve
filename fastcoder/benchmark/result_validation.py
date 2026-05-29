"""Benchmark result schema and sanity validation."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from fastcoder.benchmark.results import BenchmarkResult, MetricSummary

REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "run_id",
    "created_at",
    "config_name",
    "model",
    "hardware",
    "software",
    "speculation",
    "aggregate_metrics",
    "per_workload_metrics",
    "per_request_metrics",
}


class ResultValidationError(ValueError):
    """Raised when a result JSON file fails schema or sanity validation."""


def validate_result_file(path: str | Path) -> BenchmarkResult:
    """Load and validate a benchmark result JSON file."""

    result_path = Path(path)
    try:
        raw = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"{result_path} is not valid JSON: {exc}"
        raise ResultValidationError(msg) from exc

    if not isinstance(raw, dict):
        msg = f"{result_path} must contain a JSON object"
        raise ResultValidationError(msg)

    missing = sorted(REQUIRED_TOP_LEVEL_FIELDS - set(raw))
    if missing:
        msg = f"{result_path} is missing required top-level fields: {', '.join(missing)}"
        raise ResultValidationError(msg)

    try:
        result = BenchmarkResult.model_validate(raw)
    except ValidationError as exc:
        msg = f"{result_path} does not match benchmark result schema: {exc}"
        raise ResultValidationError(msg) from exc

    issues = validate_result_sanity(result)
    if issues:
        msg = f"{result_path} failed result sanity checks:\n- " + "\n- ".join(issues)
        raise ResultValidationError(msg)
    return result


def validate_result_sanity(result: BenchmarkResult) -> list[str]:
    """Return human-readable sanity issues for a parsed result."""

    issues: list[str] = []
    _validate_summary(
        "aggregate_metrics",
        result.aggregate_metrics,
        result.hardware.hourly_cost_usd,
        issues,
    )
    for index, group in enumerate(result.per_workload_metrics):
        label = f"per_workload_metrics[{index}] {group.workload}/c{group.concurrency}"
        _validate_summary(label, group.metrics, result.hardware.hourly_cost_usd, issues)

    if result.aggregate_metrics.request_count != len(result.per_request_metrics):
        issues.append(
            "aggregate request_count does not match per_request_metrics length "
            f"({result.aggregate_metrics.request_count} != {len(result.per_request_metrics)})"
        )
    return issues


def _validate_summary(
    label: str,
    summary: MetricSummary,
    hourly_cost_usd: float | None,
    issues: list[str],
) -> None:
    for field_name in ("request_count", "success_count", "error_count", "output_tokens"):
        value = getattr(summary, field_name)
        if value < 0:
            issues.append(f"{label}.{field_name} must be nonnegative")

    for prefix in ("latency", "ttft", "itl"):
        p50 = getattr(summary, f"p50_{prefix}_seconds")
        p95 = getattr(summary, f"p95_{prefix}_seconds")
        p99 = getattr(summary, f"p99_{prefix}_seconds")
        if p50 is not None and p95 is not None and p99 is not None and not p50 <= p95 <= p99:
            issues.append(
                f"{label} has invalid percentile order for {prefix}: p50 > p95 or p95 > p99"
            )

    for field_name in ("request_throughput_rps", "output_token_throughput_tps"):
        value = getattr(summary, field_name)
        if value is not None and value < 0:
            issues.append(f"{label}.{field_name} must be nonnegative when present")

    pass_at_1 = summary.humaneval_pass_at_1
    if pass_at_1 is not None and not 0.0 <= pass_at_1 <= 1.0:
        issues.append(f"{label}.humaneval_pass_at_1 must be within [0, 1] when present")

    cost = summary.cost_per_1m_output_tokens_usd
    tps = summary.output_token_throughput_tps
    if cost is not None:
        if hourly_cost_usd is None or hourly_cost_usd <= 0:
            issues.append(
                f"{label}.cost_per_1m_output_tokens_usd is set without hourly hardware cost"
            )
        if tps is None or tps <= 0:
            issues.append(
                f"{label}.cost_per_1m_output_tokens_usd is set without output token throughput"
            )
        if cost < 0:
            issues.append(f"{label}.cost_per_1m_output_tokens_usd must be nonnegative")
