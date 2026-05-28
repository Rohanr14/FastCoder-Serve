"""Validate a FastCoder benchmark result JSON file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastcoder.benchmark.result_validation import (  # noqa: E402
    ResultValidationError,
    validate_result_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a FastCoder benchmark result JSON file.")
    parser.add_argument("result", help="Path to result JSON.")
    args = parser.parse_args()

    try:
        result = validate_result_file(args.result)
    except (OSError, ResultValidationError) as exc:
        print(f"Invalid result: {exc}", file=sys.stderr)
        return 1

    metrics = result.aggregate_metrics
    print("Result valid")
    print(f"  schema: {result.schema_version}")
    print(f"  run id: {result.run_id}")
    print(f"  config: {result.config_name}")
    print(f"  model: {result.model.name}")
    print(f"  provider: {result.hardware.provider}")
    print(f"  requests: {metrics.request_count}")
    print(f"  errors: {metrics.error_count}")
    print(f"  p50 TTFT: {metrics.p50_ttft_seconds}")
    print(f"  p50 ITL: {metrics.p50_itl_seconds}")
    print(f"  output token throughput: {metrics.output_token_throughput_tps}")
    print(f"  cost per 1M output tokens: {metrics.cost_per_1m_output_tokens_usd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
