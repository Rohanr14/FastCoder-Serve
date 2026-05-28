"""Placeholder Pareto plot generation."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True, slots=True)
class ParetoPoint:
    label: str
    x_value: float
    y_value: float


def generate_pareto_plot(result_paths: Iterable[str | Path], output_path: str | Path) -> Path:
    """Generate a simple placeholder Pareto-style scatter plot."""

    points = [point for path in result_paths for point in _load_points(path)]
    if not points:
        msg = "no plottable points found in result files"
        raise ValueError(msg)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter([point.x_value for point in points], [point.y_value for point in points])
    for point in points:
        ax.annotate(
            point.label,
            (point.x_value, point.y_value),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("p50 latency or TTFT (seconds)")
    ax.set_ylabel("Throughput (output tokens / second)")
    ax.set_title("FastCoder-Serve Pareto Placeholder")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)
    return destination


def _load_points(path: str | Path) -> list[ParetoPoint]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return []

    points: list[ParetoPoint] = []
    experiment_name = str(data.get("config_name", data.get("experiment_name", Path(path).stem)))
    for group in _groups_or_summary(data):
        label = f"{experiment_name}:{group['label']}"
        summary = group["summary"]
        x_value = _first_float(
            summary.get("p50_ttft_seconds"),
            summary.get("p50_latency_seconds"),
            summary.get("cost_per_1m_output_tokens_usd"),
        )
        y_value = _first_float(
            summary.get("output_token_throughput_tps"),
            summary.get("throughput_output_tokens_per_second"),
        )
        if x_value is not None and y_value is not None:
            points.append(ParetoPoint(label=label, x_value=x_value, y_value=y_value))
    return points


def _groups_or_summary(data: dict[str, Any]) -> list[dict[str, Any]]:
    groups = data.get("groups")
    if isinstance(groups, list) and groups:
        result: list[dict[str, Any]] = []
        for group in groups:
            if not isinstance(group, dict):
                continue
            summary = group.get("summary")
            if not isinstance(summary, dict):
                continue
            workload = str(group.get("workload", "workload"))
            concurrency = str(group.get("concurrency", "c?"))
            result.append({"label": f"{workload}/c{concurrency}", "summary": summary})
        return result
    summary = data.get("summary")
    if isinstance(summary, dict):
        return [{"label": "summary", "summary": summary}]
    per_workload_metrics = data.get("per_workload_metrics")
    if isinstance(per_workload_metrics, list) and per_workload_metrics:
        result = []
        for group in per_workload_metrics:
            if not isinstance(group, dict):
                continue
            metrics = group.get("metrics")
            if not isinstance(metrics, dict):
                continue
            workload = str(group.get("workload", "workload"))
            concurrency = str(group.get("concurrency", "c?"))
            result.append({"label": f"{workload}/c{concurrency}", "summary": metrics})
        return result
    aggregate_metrics = data.get("aggregate_metrics")
    if isinstance(aggregate_metrics, dict):
        return [{"label": "summary", "summary": aggregate_metrics}]
    return []


def _first_float(*values: object) -> float | None:
    for value in values:
        if isinstance(value, int | float):
            return float(value)
    return None
