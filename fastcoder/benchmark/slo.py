"""SLO / capacity analysis over committed benchmark results.

Pure post-processing of per-operating-point metrics (``per_workload_metrics``) — no GPU, no network,
no model. It answers: under a latency SLO, what is the maximum sustained output throughput (and its
cost), and at which operating point?

The benchmark is **closed-loop**: ``concurrency`` is the number of concurrent in-flight requests the
driver maintained, not an open-loop arrival rate. "Max sustained throughput under an SLO" therefore
means the highest-throughput measured operating point whose tail latency still satisfies the SLO.
This is robust to non-monotonic tails (a higher concurrency can occasionally show a *lower* p99 than
a smaller one) because it maximises over all qualifying points rather than stopping at the first
violation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """One (precision, workload, concurrency) measurement from a result file."""

    label: str
    workload: str
    concurrency: int
    p99_ttft_seconds: float | None
    p99_latency_seconds: float | None
    output_token_throughput_tps: float | None
    cost_per_1m_output_tokens_usd: float | None


def operating_points_from_result(result: Mapping[str, Any], label: str) -> list[OperatingPoint]:
    """Extract operating points from a benchmark result dict's ``per_workload_metrics``."""

    points: list[OperatingPoint] = []
    groups = result.get("per_workload_metrics")
    if not isinstance(groups, list):
        return points
    for group in groups:
        if not isinstance(group, Mapping):
            continue
        metrics = group.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        points.append(
            OperatingPoint(
                label=label,
                workload=str(group.get("workload", "")),
                concurrency=int(group.get("concurrency", 0)),
                p99_ttft_seconds=_opt_float(metrics.get("p99_ttft_seconds")),
                p99_latency_seconds=_opt_float(metrics.get("p99_latency_seconds")),
                output_token_throughput_tps=_opt_float(metrics.get("output_token_throughput_tps")),
                cost_per_1m_output_tokens_usd=_opt_float(
                    metrics.get("cost_per_1m_output_tokens_usd")
                ),
            )
        )
    return points


def max_throughput_under_slo(
    points: Iterable[OperatingPoint],
    *,
    slo_seconds: float,
    slo_attr: str = "p99_ttft_seconds",
    workload: str | None = None,
) -> OperatingPoint | None:
    """Return the highest-throughput operating point that still meets the latency SLO.

    A point qualifies when ``slo_attr`` is present and ``<= slo_seconds`` and it has a throughput.
    Pass ``slo_seconds=float("inf")`` to get the unconstrained throughput peak (useful for exposing
    saturation). Optionally restrict to a single ``workload``. Returns ``None`` if nothing
    qualifies.
    """

    best: OperatingPoint | None = None
    for point in points:
        if workload is not None and point.workload != workload:
            continue
        slo_value = getattr(point, slo_attr)
        throughput = point.output_token_throughput_tps
        if slo_value is None or throughput is None:
            continue
        if slo_value > slo_seconds:
            continue
        if best is None or throughput > (best.output_token_throughput_tps or float("-inf")):
            best = point
    return best


def _opt_float(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None
