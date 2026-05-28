"""Check an OpenAI-compatible endpoint before benchmark execution."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastcoder.benchmark.endpoint import check_openai_compatible_endpoint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Check an OpenAI-compatible endpoint.")
    parser.add_argument(
        "--base-url",
        required=True,
        help="OpenAI-compatible API base URL, usually ending in /v1.",
    )
    parser.add_argument("--api-key", default=None, help="Optional bearer token.")
    parser.add_argument(
        "--model",
        default=None,
        help="Model name to send in the tiny chat request.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--stream", action="store_true", help="Verify SSE streaming behavior.")
    args = parser.parse_args()

    report = asyncio.run(
        check_openai_compatible_endpoint(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
            stream=args.stream,
        )
    )

    print("Endpoint check")
    print(f"  base URL: {report.base_url}")
    print(f"  chat URL: {report.chat_url}")
    print(f"  model: {report.model}")
    print(f"  stream: {report.stream}")
    print(f"  connectivity: {report.connectivity_ok} ({report.connectivity_status_code})")
    print(f"  /health available: {report.health_available} ({report.health_status_code})")
    print(f"  /v1/models available: {report.models_available} ({report.models_status_code})")
    print(f"  chat completions: {report.chat_ok} ({report.chat_status_code})")
    if report.stream:
        print(f"  stream chunks: {report.stream_chunk_count}")
        print(f"  stream terminated: {report.stream_terminated}")
    for note in report.notes:
        print(f"  note: {note}")
    for error in report.errors:
        print(f"  error: {error}", file=sys.stderr)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
