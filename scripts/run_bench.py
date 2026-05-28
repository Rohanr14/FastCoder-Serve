"""CLI entrypoint for the FastCoder benchmark harness."""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastcoder.benchmark.config import load_benchmark_config  # noqa: E402
from fastcoder.benchmark.runner import run_benchmark_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a FastCoder benchmark config.")
    parser.add_argument("--config", required=True, help="Path to a benchmark YAML config.")
    parser.add_argument(
        "--start-fake-server",
        action="store_true",
        help="Start the local fake OpenAI-compatible server for this run.",
    )
    args = parser.parse_args()

    config = load_benchmark_config(args.config)
    if args.start_fake_server:
        with _fake_server_for_config(config.model_server.base_url):
            result = asyncio.run(run_benchmark_config(config))
    else:
        result = asyncio.run(run_benchmark_config(config))

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(_json_dumps(result), encoding="utf-8")
    print(f"Wrote benchmark results to {config.output_path}")


def _json_dumps(data: object) -> str:
    import json

    return f"{json.dumps(data, indent=2, sort_keys=True)}\n"


@contextmanager
def _fake_server_for_config(base_url: str) -> Iterator[None]:
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 9000
    bind_host = "127.0.0.1" if host in {"localhost", "127.0.0.1"} else host
    health_url = f"{parsed.scheme or 'http'}://{host}:{port}/health"
    script_path = ROOT / "scripts" / "fake_openai_server.py"
    process = subprocess.Popen(
        [sys.executable, str(script_path), "--host", bind_host, "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        asyncio.run(_wait_for_health(health_url, process))
        yield
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


async def _wait_for_health(
    url: str,
    process: subprocess.Popen[bytes],
    timeout_seconds: float = 8.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    async with httpx.AsyncClient(timeout=1.0) as client:
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    return
            except httpx.HTTPError as exc:
                last_error = exc
            if process.poll() is not None and last_error is not None:
                raise RuntimeError("fake server exited before becoming healthy") from last_error
            await asyncio.sleep(0.1)
    raise TimeoutError(f"fake server did not become healthy at {url}")


if __name__ == "__main__":
    main()
