"""Dry-run-gated HumanEval pass@1 evaluation (pod-only execution).

Mirrors scripts/run_h100_baseline.py: dry-run by default (validates config and prints the plan,
imports no GPU/eval dependency, runs nothing), and only with --confirm-paid-run does it hit the
live endpoint and execute scoring. Scoring runs model-generated code via human_eval's sandboxed
subprocess and requires the eval extra, which exists only on the pod.
"""

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
from fastcoder.benchmark.humaneval_runner import run_humaneval_eval  # noqa: E402
from fastcoder.benchmark.manifest import write_run_manifest  # noqa: E402
from fastcoder.benchmark.result_validation import validate_result_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run or dry-run the HumanEval pass@1 evaluation workflow."
    )
    parser.add_argument("--config", default="configs/humaneval_fp16.yaml")
    parser.add_argument("--base-url", default=None, help="Override config model_server.base_url.")
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional gateway or upstream bearer token.",
    )
    parser.add_argument(
        "--humaneval-path",
        default=None,
        help="Optional JSONL(.gz) of HumanEval problems; else $FASTCODER_HUMANEVAL_PATH or the "
        "installed human_eval package.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=10.0,
        help="Per-problem execution timeout passed to the sandboxed scorer.",
    )
    parser.add_argument("--confirm-paid-run", action="store_true")
    args = parser.parse_args()

    config = _load_effective_config(args.config, args.base_url, args.api_key)

    if not args.confirm_paid_run:
        _print_dry_run(config, args.config, args.base_url, args.api_key, args.humaneval_path)
        return 0

    return asyncio.run(
        _run_confirmed(config, args.config, args.humaneval_path, args.timeout_seconds)
    )


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
    humaneval_path: str | None,
) -> None:
    print("DRY RUN: HumanEval pass@1 evaluation was not executed.")
    print("Validated config and planned pod-only eval steps:")
    print(f"  config: {config_path}")
    print(f"  effective base URL: {config.model_server.base_url}")
    print(f"  model: {config.model_server.model}")
    print(f"  workloads: {', '.join(config.workloads)}")
    print(f"  single-pass concurrency: {min(config.concurrency)}")
    print(f"  max tokens: {config.max_tokens}")
    print(f"  output path: {config.output_path}")
    source = humaneval_path or "$FASTCODER_HUMANEVAL_PATH or the installed human_eval package"
    print(f"  humaneval source: {source}")
    print()
    print("Would run:")
    confirmed_command = (
        f"python scripts/run_humaneval_eval.py --config {config_path} "
        f"--base-url {config.model_server.base_url}"
    )
    if humaneval_path is not None:
        confirmed_command += f" --humaneval-path {humaneval_path}"
    if api_key is not None:
        confirmed_command += " --api-key <redacted>"
    confirmed_command += " --confirm-paid-run"
    print(f"  {confirmed_command}")
    print()
    print(
        "Confirmed run would: check the endpoint, write a manifest, run all HumanEval problems "
        "once, execute each candidate in human_eval's sandboxed subprocess, compute pass@1, then "
        "write and validate the result JSON."
    )
    print(
        "Scoring executes model-generated code and requires the 'human_eval' package "
        "(pip install -e '.[eval]'); it is intentionally unavailable locally and in CI."
    )
    print()
    print(
        "To execute the paid evaluation, rerun with --confirm-paid-run after the H100 pod is live."
    )
    if base_url is None:
        print("Note: --base-url was not provided; using the config endpoint.")
    if api_key is not None:
        print("Note: API key was provided and will be used, but is not printed.")


async def _run_confirmed(
    config: BenchmarkConfig,
    config_path: str,
    humaneval_path: str | None,
    timeout_seconds: float,
) -> int:
    print(
        "CONFIRMED PAID RUN: proceeding with endpoint check, manifest, HumanEval eval, scoring, "
        "validation."
    )
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
        notes="Confirmed HumanEval pass@1 eval run",
    )
    print(f"Wrote manifest: {manifest_path}")

    result = await run_humaneval_eval(
        config,
        humaneval_path=humaneval_path,
        timeout_seconds=timeout_seconds,
    )
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(
        f"{json.dumps(result, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    validate_result_file(config.output_path)
    print(f"Wrote and validated result: {config.output_path}")

    aggregate = result.get("aggregate_metrics", {})
    pass_at_1 = aggregate.get("humaneval_pass_at_1")
    if pass_at_1 is not None:
        print(
            f"HumanEval pass@1: {pass_at_1:.4f} "
            f"({aggregate.get('success_count')} ok / {aggregate.get('request_count')} requests)"
        )
    else:
        print("WARNING: pass@1 was not populated; check the eval run.")

    captured_version = result.get("software", {}).get("vllm_version")
    if captured_version:
        print(f"Captured live vLLM version: {captured_version}")
    else:
        print("WARNING: could not capture vLLM version from /version; field left null.")
    print(
        "Next steps: copy results off the pod, shut the pod down, and commit measured JSON "
        "with methodology."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
