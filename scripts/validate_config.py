"""Validate and summarize a FastCoder benchmark YAML config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastcoder.benchmark.config import load_benchmark_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a FastCoder benchmark config.")
    parser.add_argument("--config", required=True, help="Path to benchmark YAML config.")
    args = parser.parse_args()

    try:
        config = load_benchmark_config(args.config)
    except (OSError, ValueError, ValidationError) as exc:
        print(f"Invalid config: {exc}", file=sys.stderr)
        return 1

    print("Config valid")
    print(f"  name: {config.experiment_name}")
    print(f"  model: {config.model_server.model}")
    print(f"  endpoint: {config.model_server.base_url}")
    print(f"  chat URL: {config.model_server.chat_completions_url}")
    print(f"  workloads: {', '.join(config.workloads)}")
    print(f"  concurrency: {config.concurrency}")
    print(f"  streaming: {config.stream}")
    print(f"  output path: {config.output_path}")
    print(f"  provider: {config.hardware.provider}")
    print(f"  gpu: {config.hardware.gpu_name}")
    print(f"  gpu memory GB: {config.hardware.gpu_memory_gb}")
    print(f"  serving backend: {config.software.serving_backend}")
    print(f"  vLLM version: {config.software.vllm_version}")
    print(f"  backend commit: {config.software.backend_commit}")
    print(f"  speculative method: {config.speculation.method}")
    print(f"  speculative model: {config.speculation.draft_model}")
    hourly_cost = config.hardware.hourly_cost_usd or config.cost.accelerator_hourly_cost_usd
    print(f"  hourly cost USD: {hourly_cost}")
    print(f"  measurement hypothesis: {config.measurement_hypothesis}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
