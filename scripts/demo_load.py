"""Continuous load generator for the FastCoder observability demo.

Sends a steady mix of streaming and non-streaming chat-completion requests at the gateway so the
Prometheus/Grafana dashboard shows live traffic. CPU-only: it drives the gateway, which proxies the
fake OpenAI server. This is NOT a benchmark — it exists purely to populate the demo dashboard, so it
makes no latency claims and writes no result files.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import time

import httpx

_PROMPTS = (
    "In one sentence, what does a Python dict do?",
    "Write a function that reverses a linked list.",
    "How would you design a rate limiter for an API?",
    "Explain async/await in Python briefly.",
    "What is the time complexity of binary search?",
    "Complete: def add(a: int, b: int) -> int:",
)


async def _worker(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    stop_at: float,
    stream_ratio: float,
) -> None:
    while time.monotonic() < stop_at:
        stream = random.random() < stream_ratio
        payload = {
            "model": "fake-coder",
            "messages": [{"role": "user", "content": random.choice(_PROMPTS)}],
            "max_tokens": 48,
            "stream": stream,
        }
        try:
            if stream:
                async with client.stream("POST", url, json=payload, headers=headers) as response:
                    async for _ in response.aiter_lines():
                        pass
            else:
                await client.post(url, json=payload, headers=headers)
        except httpx.HTTPError:
            await asyncio.sleep(0.2)


async def _run(args: argparse.Namespace) -> None:
    url = f"{args.base_url.rstrip('/')}/chat/completions"
    headers = {"content-type": "application/json"}
    if args.api_key:
        headers["authorization"] = f"Bearer {args.api_key}"
    forever = args.duration_seconds <= 0
    stop_at = time.monotonic() + (1e9 if forever else args.duration_seconds)
    async with httpx.AsyncClient(timeout=args.request_timeout) as client:
        await asyncio.gather(
            *(
                _worker(client, url, headers, stop_at, args.stream_ratio)
                for _ in range(args.concurrency)
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Continuous load generator for the demo.")
    parser.add_argument("--base-url", default=os.environ.get("DEMO_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--api-key", default=os.environ.get("DEMO_API_KEY", "dev-token"))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--duration-seconds", type=float, default=0.0, help="0 (default) runs forever."
    )
    parser.add_argument("--stream-ratio", type=float, default=0.5)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    args = parser.parse_args()

    horizon = "forever" if args.duration_seconds <= 0 else f"{args.duration_seconds:g}s"
    print(
        f"demo load -> {args.base_url} (concurrency={args.concurrency}, "
        f"stream_ratio={args.stream_ratio}, {horizon})"
    )
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
