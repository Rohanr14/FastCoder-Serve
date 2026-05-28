"""Dry-run-gated orchestration for the future paid H100 FP16 baseline."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastcoder.benchmark.config import BenchmarkConfig, load_benchmark_config  # noqa: E402
from fastcoder.benchmark.endpoint import check_openai_compatible_endpoint  # noqa: E402
from fastcoder.benchmark.manifest import write_run_manifest  # noqa: E402
from fastcoder.benchmark.result_validation import validate_result_file  # noqa: E402
from fastcoder.benchmark.runner import run_benchmark_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or dry-run the H100 FP16 baseline workflow.")
    parser.add_argument("--config", default="configs/baseline_fp16.yaml")
    parser.add_argument("--base-url", default=None, help="Override config model_server.base_url.")
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional gateway or upstream bearer token.",
    )
    parser.add_argument("--confirm-paid-run", action="store_true")
    args = parser.parse_args()

    config = _load_effective_config(args.config, args.base_url, args.api_key)
    if not args.confirm_paid_run:
        _print_dry_run(config, args.config, args.base_url, args.api_key)
        return 0

    return asyncio.run(_run_confirmed(config, args.config))


def _load_effective_config(
    config_path: str,
    base_url: str | None,
    api_key: str | None,
) -> BenchmarkConfig:
    config = load_benchmark_config(config_path)
    model_server = config.model_server
    if base_url is not None:
        model_server = model_server.model_copy(update={"base_url": base_url})
    if api_key is not None:
        model_server = model_server.model_copy(update={"api_key": api_key})
    return config.model_copy(update={"model_server": model_server})


def _print_dry_run(
    config: BenchmarkConfig,
    config_path: str,
    base_url: str | None,
    api_key: str | None,
) -> None:
    print("DRY RUN: H100 FP16 baseline was not executed.")
    print("Validated config and planned paid-run steps:")
    print(f"  config: {config_path}")
    print(f"  effective base URL: {config.model_server.base_url}")
    print(f"  model: {config.model_server.model}")
    print(f"  workloads: {', '.join(config.workloads)}")
    print(f"  concurrency: {config.concurrency}")
    print(f"  output path: {config.output_path}")
    print()
    print("Would run:")
    confirmed_command = (
        f"python scripts/run_h100_baseline.py --config {config_path} "
        f"--base-url {config.model_server.base_url}"
    )
    if api_key is not None:
        confirmed_command += " --api-key <redacted>"
    confirmed_command += " --confirm-paid-run"
    print(f"  {confirmed_command}")
    print()
    print("Confirmed run would perform these internal steps:")
    print(
        "  python scripts/check_endpoint.py "
        f"--base-url {config.model_server.base_url} "
        f"--model {config.model_server.model} --stream"
    )
    print(f"  python scripts/create_run_manifest.py --config {config_path}")
    print(f"  python scripts/run_bench.py --config {config_path}")
    print(f"  python scripts/validate_results.py {config.output_path}")
    print()
    print(
        "To execute the paid benchmark, rerun with --confirm-paid-run after the H100 pod "
        "is live."
    )
    if base_url is None:
        print("Note: --base-url was not provided; using the config endpoint.")
    if api_key is not None:
        print("Note: API key was provided and will be used, but is not printed.")


async def _run_confirmed(config: BenchmarkConfig, config_path: str) -> int:
    print("CONFIRMED PAID RUN: proceeding with endpoint check, manifest, benchmark, validation.")
    endpoint_report = await check_openai_compatible_endpoint(
        base_url=config.model_server.base_url,
        api_key=config.model_server.api_key,
        model=config.model_server.model,
        timeout_seconds=config.model_server.timeout_seconds,
        stream=config.stream,
    )
    if not endpoint_report.ok:
        print("Endpoint check failed:", file=sys.stderr)
        for error in endpoint_report.errors:
            print(f"  {error}", file=sys.stderr)
        return 1

    manifest_path = write_run_manifest(
        config_path=config_path,
        notes="Confirmed H100 FP16 baseline run",
    )
    print(f"Wrote manifest: {manifest_path}")

    result = await run_benchmark_config(config)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(
        f"{json.dumps(result, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    validate_result_file(config.output_path)
    print(f"Wrote and validated result: {config.output_path}")
    print(
        "Next steps: copy results off the pod, shut the pod down, and commit measured JSON "
        "with methodology."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
