"""Fetch vLLM /metrics and report speculative-decoding acceptance (best-effort, read-only).

Run this WHILE the speculative vLLM server is still up (right after the spec benchmark). It prints
the raw spec_decode metric lines and the derived accepted/draft acceptance so you can record it in
the result/methodology. No GPU, no model — just an HTTP GET against the server's /metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastcoder.benchmark.spec import spec_decode_acceptance, spec_decode_lines  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Report vLLM speculative-decoding acceptance.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/v1",
        help="vLLM OpenAI base URL; /metrics is read from the server root.",
    )
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    root = args.base_url.rstrip("/").removesuffix("/v1")
    metrics_url = f"{root}/metrics"
    headers = {}
    if args.api_key:
        headers["authorization"] = f"Bearer {args.api_key}"

    try:
        response = httpx.get(metrics_url, headers=headers, timeout=10.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"Failed to read {metrics_url}: {exc}", file=sys.stderr)
        return 1

    lines = spec_decode_lines(response.text)
    if not lines:
        print(f"No spec_decode metrics found at {metrics_url}.")
        print("Is the server running with speculative decoding? Metric names also vary by version.")
        return 0

    print("spec_decode metrics:")
    for line in lines:
        print(f"  {line}")

    acceptance = spec_decode_acceptance(response.text)
    if acceptance is not None:
        print(f"\nDerived drafted-token acceptance: {acceptance:.4f}")
    else:
        print("\nCould not derive accepted/draft acceptance — read the raw lines above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
