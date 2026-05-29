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
from matplotlib.lines import Line2D


@dataclass(frozen=True, slots=True)
class ParetoPoint:
    label: str
    x_value: float
    y_value: float


def generate_pareto_plot(result_paths: Iterable[str | Path], output_path: str | Path) -> Path:
    """Plot output throughput vs latency, colored by precision/experiment.

    Each result file is one color series and each workload a marker shape, with the encoding carried
    by two legends rather than per-point text labels (which overlap badly once several precisions x
    workloads x concurrency levels are plotted together).
    """

    points = [point for path in result_paths for point in _load_points(path)]
    if not points:
        msg = "no plottable points found in result files"
        raise ValueError(msg)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    series_names = sorted({_series_of(point) for point in points})
    workloads = sorted({_workload_of(point) for point in points})
    palette = plt.get_cmap("tab10")
    color_by_series = {name: palette(index % 10) for index, name in enumerate(series_names)}
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*"]
    marker_by_workload = {
        name: marker_cycle[index % len(marker_cycle)] for index, name in enumerate(workloads)
    }

    fig, ax = plt.subplots(figsize=(9, 6))
    for point in points:
        ax.scatter(
            point.x_value,
            point.y_value,
            color=color_by_series[_series_of(point)],
            marker=marker_by_workload[_workload_of(point)],
            s=70,
            edgecolors="black",
            linewidths=0.4,
            alpha=0.9,
        )

    precision_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=8,
               color=color_by_series[name], label=name)
        for name in series_names
    ]
    workload_handles = [
        Line2D([0], [0], marker=marker_by_workload[name], linestyle="", markersize=8,
               color="0.4", label=name)
        for name in workloads
    ]
    precision_legend = ax.legend(
        handles=precision_handles, title="precision", loc="upper left", fontsize=8
    )
    ax.add_artist(precision_legend)
    ax.legend(handles=workload_handles, title="workload", loc="lower right", fontsize=8)

    ax.set_xlabel("p50 TTFT or latency (seconds)")
    ax.set_ylabel("output throughput (tokens / second)")
    ax.set_title(
        "FastCoder-Serve: throughput vs latency\n"
        "Qwen2.5-Coder-7B-Instruct, single H100, vLLM"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(destination, dpi=120)
    plt.close(fig)
    return destination


def _series_of(point: ParetoPoint) -> str:
    """Precision/experiment portion of a ``experiment:workload/cN`` label."""

    return point.label.split(":", 1)[0]


def _workload_of(point: ParetoPoint) -> str:
    """Workload portion of a ``experiment:workload/cN`` label (drops the ``/cN`` suffix)."""

    rest = point.label.split(":", 1)[1] if ":" in point.label else point.label
    return rest.rsplit("/c", 1)[0]


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
