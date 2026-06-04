"""Dry-run-gated open-loop (arrival-rate) load test (pod-only execution).

Mirrors scripts/run_h100_baseline.py: dry-run by default (validates config, prints the plan, runs
nothing), and only with --confirm-paid-run does it fire real Poisson-timed load at the configured
arrival rates. Open-loop intentionally does NOT cap concurrency, so it can drive the server into
overload — that is the point.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastcoder.benchmark.config import BenchmarkConfig, load_benchmark_config  # noqa: E402
from fastcoder.benchmark.endpoint import check_openai_compatible_endpoint  # noqa: E402
from fastcoder.benchmark.manifest import write_run_manifest  # noqa: E402
from fastcoder.benchmark.openloop import (  # noqa: E402
    run_open_loop_sweep,
    validate_open_loop_result,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or dry-run the open-loop arrival-rate sweep.")
    parser.add_argument("--config", default="configs/openloop_fp8.yaml")
    parser.add_argument("--base-url", default=None, help="Override config model_server.base_url.")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--confirm-paid-run", action="store_true")
    args = parser.parse_args()

    config = _load_effective_config(args.config, args.base_url, args.api_key)
    if not args.confirm_paid_run:
        _print_dry_run(config, args.config)
        return 0
    return asyncio.run(_run_confirmed(config, args.config))


def _load_effective_config(
    config_path: str, base_url: str | None, api_key: str | None
) -> BenchmarkConfig:
    config = load_benchmark_config(config_path)
    model_server = config.model_server
    if base_url is not None:
        model_server = model_server.model_copy(update={"base_url": base_url})
    if api_key is not None:
        model_server = model_server.model_copy(update={"api_key": api_key})
    return config.model_copy(update={"model_server": model_server})


def _print_dry_run(config: BenchmarkConfig, config_path: str) -> None:
    settings = config.open_loop
    print("DRY RUN: open-loop load test was not executed.")
    print("Validated config and planned arrival-rate sweep:")
    print(f"  config: {config_path}")
    print(f"  effective base URL: {config.model_server.base_url}")
    print(f"  workload: {config.workloads[0]}")
    print(f"  arrival rates (req/s): {settings.arrival_rates_rps}")
    print(f"  requests per rate: {settings.requests_per_rate}")
    slo = f"p99 TTFT <= {settings.slo_ttft_ms:g} ms"
    if settings.slo_latency_ms is not None:
        slo += f", latency <= {settings.slo_latency_ms:g} ms"
    print(f"  SLO: {slo}")
    print(f"  output path: {config.output_path}")
    print()
    confirmed = (
        f"python scripts/run_openloop.py --config {config_path} "
        f"--base-url {config.model_server.base_url} --confirm-paid-run"
    )
    print("Would run:")
    print(f"  {confirmed}")
    print()
    print("Open-loop fires Poisson-timed requests with NO concurrency cap, so it can drive the")
    print("server into overload (intended). To execute, rerun with --confirm-paid-run on the pod.")


async def _run_confirmed(config: BenchmarkConfig, config_path: str) -> int:
    print("CONFIRMED PAID RUN: endpoint check, manifest, open-loop sweep, validation.")
    report = await check_openai_compatible_endpoint(
        base_url=config.model_server.base_url,
        api_key=config.model_server.api_key,
        model=config.model_server.model,
        timeout_seconds=config.model_server.timeout_seconds,
        stream=config.stream,
    )
    if not report.ok:
        print("Endpoint check failed:", file=sys.stderr)
        for error in report.errors:
            print(f"  {error}", file=sys.stderr)
        return 1

    manifest_path = write_run_manifest(
        config_path=config_path, notes="Confirmed open-loop load test"
    )
    print(f"Wrote manifest: {manifest_path}")

    result = await run_open_loop_sweep(config)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(
        f"{json.dumps(result, indent=2, sort_keys=True)}\n", encoding="utf-8"
    )
    issues = validate_open_loop_result(result)
    if issues:
        print("WARNING: open-loop result sanity issues:", file=sys.stderr)
        for issue in issues:
            print(f"  {issue}", file=sys.stderr)
    print(f"Wrote result: {config.output_path}")
    _print_summary(result)
    print(f"Next: copy off, terminate, then: python scripts/plot_openloop.py {config.output_path}")
    return 0


def _fmt(value: Any, scale: float = 1.0, digits: int = 2) -> str:
    return f"{value * scale:.{digits}f}" if isinstance(value, int | float) else "-"


def _print_summary(result: dict[str, Any]) -> None:
    cols = ["rate", "ok/err", "achiev", "p50lat", "p99lat", "p99ttft", "SLOviol"]
    widths = [6, 10, 10, 9, 9, 9, 9]
    print()
    print("".join(f"{c:>{w}}" for c, w in zip(cols, widths, strict=False)))
    for point in result["arrival_rate_points"]:
        viol = point.get("slo_violation_rate")
        viol_s = f"{viol * 100:.0f}%" if isinstance(viol, int | float) else "-"
        cells = [
            f"{point['target_rate_rps']:g}",
            f"{point['success_count']}/{point['error_count']}",
            _fmt(point.get("achieved_throughput_rps")),
            _fmt(point.get("p50_latency_seconds")),
            _fmt(point.get("p99_latency_seconds")),
            _fmt(point.get("p99_ttft_seconds"), 1.0, 3),
            viol_s,
        ]
        print("".join(f"{cell:>{w}}" for cell, w in zip(cells, widths, strict=False)))


if __name__ == "__main__":
    raise SystemExit(main())
