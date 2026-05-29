"""SLO / capacity analysis over benchmark result JSONs (pure post-processing, no GPU/network).

For each p99 TTFT SLO, prints the maximum sustained output throughput (with cost and operating
point) that still meets it, per precision. Also prints the throughput-peak concurrency per workload,
which exposes saturation (e.g. long-context collapsing at high concurrency).

Defaults to the committed FP16/FP8/AWQ baselines; override with ``label=path`` arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastcoder.benchmark.slo import (  # noqa: E402
    OperatingPoint,
    max_throughput_under_slo,
    operating_points_from_result,
)

DEFAULT_RESULTS = (
    ("FP16", "results/baseline_fp16.json"),
    ("FP8", "results/baseline_fp8.json"),
    ("AWQ-INT4", "results/baseline_awq.json"),
)
DEFAULT_TTFT_SLOS_MS = (250.0, 500.0, 1000.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="SLO / capacity analysis over result JSONs.")
    parser.add_argument(
        "results",
        nargs="*",
        help="Optional label=path pairs; defaults to the committed FP16/FP8/AWQ baselines.",
    )
    parser.add_argument(
        "--ttft-slo-ms",
        type=float,
        nargs="*",
        default=list(DEFAULT_TTFT_SLOS_MS),
        help="p99 TTFT SLO thresholds in milliseconds.",
    )
    args = parser.parse_args()

    labelled = _parse_pairs(args.results) or list(DEFAULT_RESULTS)
    points_by_label: list[tuple[str, list[OperatingPoint]]] = []
    for label, path in labelled:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        points_by_label.append((label, operating_points_from_result(result, label)))

    _print_slo_tables(points_by_label, args.ttft_slo_ms)
    _print_peak_per_workload(points_by_label)
    return 0


def _print_slo_tables(
    points_by_label: list[tuple[str, list[OperatingPoint]]],
    slos_ms: list[float],
) -> None:
    for slo_ms in slos_ms:
        slo_s = slo_ms / 1000.0
        print(f"\n### Max sustained throughput under p99 TTFT <= {slo_ms:.0f} ms")
        print("| precision | throughput (tok/s) | $/1M out | operating point |")
        print("| --- | --- | --- | --- |")
        for label, points in points_by_label:
            best = max_throughput_under_slo(points, slo_seconds=slo_s)
            if best is None:
                print(f"| {label} | — | — | no point meets SLO |")
                continue
            op = f"{best.workload}/c{best.concurrency} (p99 TTFT {best.p99_ttft_seconds:.3f}s)"
            print(
                f"| {label} | {best.output_token_throughput_tps:.0f} | "
                f"{best.cost_per_1m_output_tokens_usd:.3f} | {op} |"
            )


def _print_peak_per_workload(
    points_by_label: list[tuple[str, list[OperatingPoint]]],
) -> None:
    workloads = sorted({point.workload for _, points in points_by_label for point in points})
    print("\n### Peak throughput operating point per workload (exposes saturation)")
    print("| precision | workload | peak tok/s | @ concurrency |")
    print("| --- | --- | --- | --- |")
    for label, points in points_by_label:
        for workload in workloads:
            peak = max_throughput_under_slo(points, slo_seconds=float("inf"), workload=workload)
            if peak is None:
                continue
            print(
                f"| {label} | {workload} | {peak.output_token_throughput_tps:.0f} | "
                f"c{peak.concurrency} |"
            )


def _parse_pairs(raw: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in raw:
        if "=" not in item:
            msg = f"expected label=path, got {item!r}"
            raise SystemExit(msg)
        label, path = item.split("=", 1)
        pairs.append((label, path))
    return pairs


if __name__ == "__main__":
    raise SystemExit(main())
