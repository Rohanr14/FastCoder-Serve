from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi import FastAPI

from fastcoder.benchmark.config import BenchmarkConfig
from fastcoder.benchmark.endpoint import fetch_vllm_version
from fastcoder.benchmark.gpu import GPUSnapshot
from fastcoder.benchmark.manifest import REDACTED_VALUE, create_run_manifest, hash_file
from fastcoder.benchmark.metrics import TOKEN_SOURCE_APPROXIMATE, RequestMetric
from fastcoder.benchmark.results import build_benchmark_result
from fastcoder.benchmark.workloads import get_workloads
from scripts.fake_openai_server import app as fake_app


def test_validate_config_cli_accepts_valid_and_rejects_invalid(tmp_path: Path) -> None:
    valid = _run(["scripts/validate_config.py", "--config", "configs/local_smoke.yaml"])
    assert valid.returncode == 0
    assert "Config valid" in valid.stdout

    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text(
        """
experiment_name: invalid
concurrency: []
""",
        encoding="utf-8",
    )
    invalid = _run(["scripts/validate_config.py", "--config", str(invalid_config)])
    assert invalid.returncode != 0
    assert "Invalid config" in invalid.stderr


@pytest.mark.asyncio
async def test_check_endpoint_cli_non_streaming_and_streaming() -> None:
    port = _free_port()
    async with _serve(fake_app, port):
        non_streaming = await asyncio.to_thread(
            _run,
            [
                "scripts/check_endpoint.py",
                "--base-url",
                f"http://127.0.0.1:{port}/v1",
                "--model",
                "fake-coder",
                "--timeout-seconds",
                "5",
            ]
        )
        streaming = await asyncio.to_thread(
            _run,
            [
                "scripts/check_endpoint.py",
                "--base-url",
                f"http://127.0.0.1:{port}/v1",
                "--model",
                "fake-coder",
                "--timeout-seconds",
                "5",
                "--stream",
            ]
        )

    assert non_streaming.returncode == 0
    assert "chat completions: True" in non_streaming.stdout
    assert streaming.returncode == 0
    assert "stream chunks:" in streaming.stdout


def test_check_endpoint_cli_failure_path() -> None:
    port = _free_port()
    failure = _run(
        [
            "scripts/check_endpoint.py",
            "--base-url",
            f"http://127.0.0.1:{port}/v1",
            "--model",
            "fake-coder",
            "--timeout-seconds",
            "0.2",
        ]
    )

    assert failure.returncode != 0
    assert "error:" in failure.stderr


def test_validate_results_cli_accepts_valid_and_rejects_invalid(tmp_path: Path) -> None:
    valid_path = tmp_path / "valid_result.json"
    invalid_path = tmp_path / "invalid_result.json"
    payload = _valid_result_payload()
    valid_path.write_text(json.dumps(payload), encoding="utf-8")

    valid = _run(["scripts/validate_results.py", str(valid_path)])
    assert valid.returncode == 0
    assert "Result valid" in valid.stdout

    payload["aggregate_metrics"]["p50_latency_seconds"] = 3.0
    payload["aggregate_metrics"]["p95_latency_seconds"] = 2.0
    payload["aggregate_metrics"]["p99_latency_seconds"] = 1.0
    invalid_path.write_text(json.dumps(payload), encoding="utf-8")

    invalid = _run(["scripts/validate_results.py", str(invalid_path)])
    assert invalid.returncode != 0
    assert "failed result sanity checks" in invalid.stderr


def test_manifest_hash_is_stable_and_secrets_are_redacted() -> None:
    config_path = Path("configs/baseline_fp16.yaml")

    assert hash_file(config_path) == hash_file(config_path)
    manifest = create_run_manifest(
        config_path=config_path,
        environ={
            "VLLM_BASE_URL": "http://localhost:8000/v1",
            "FASTCODER_API_KEY": "super-secret",
        },
    )

    assert manifest.config_hash_sha256 == hash_file(config_path)
    assert manifest.environment["VLLM_BASE_URL"] == "http://localhost:8000/v1"
    assert manifest.environment["FASTCODER_API_KEY"] == REDACTED_VALUE
    assert "super-secret" not in manifest.model_dump_json()


def test_baseline_dry_run_does_not_run_benchmark() -> None:
    result = _run(
        [
            "scripts/run_h100_baseline.py",
            "--config",
            "configs/baseline_fp16.yaml",
            "--base-url",
            "http://127.0.0.1:9/v1",
        ]
    )

    assert result.returncode == 0
    assert "DRY RUN" in result.stdout
    assert "Would run:" in result.stdout
    assert "--confirm-paid-run" in result.stdout


def test_humaneval_eval_dry_run_does_not_run_or_require_human_eval() -> None:
    result = _run(
        [
            "scripts/run_humaneval_eval.py",
            "--config",
            "configs/humaneval_fp16.yaml",
            "--base-url",
            "http://127.0.0.1:9/v1",
        ]
    )

    assert result.returncode == 0
    assert "DRY RUN" in result.stdout
    assert "Would run:" in result.stdout
    assert "--confirm-paid-run" in result.stdout
    assert "human_eval" in result.stdout


def test_openloop_dry_run_does_not_fire_load() -> None:
    result = _run(
        [
            "scripts/run_openloop.py",
            "--config",
            "configs/openloop_fp8.yaml",
            "--base-url",
            "http://127.0.0.1:9/v1",
        ]
    )

    assert result.returncode == 0
    assert "DRY RUN" in result.stdout
    assert "Would run:" in result.stdout
    assert "--confirm-paid-run" in result.stdout
    assert "arrival rates" in result.stdout


@pytest.mark.asyncio
async def test_fetch_vllm_version_reads_version_from_live_route() -> None:
    port = _free_port()
    async with _serve(_version_app({"version": "0.21.0"}), port):
        version = await fetch_vllm_version(
            base_url=f"http://127.0.0.1:{port}/v1",
            timeout_seconds=5.0,
        )

    assert version == "0.21.0"


@pytest.mark.asyncio
async def test_fetch_vllm_version_returns_none_when_route_missing() -> None:
    port = _free_port()
    async with _serve(fake_app, port):
        version = await fetch_vllm_version(
            base_url=f"http://127.0.0.1:{port}/v1",
            timeout_seconds=5.0,
        )

    assert version is None


@pytest.mark.asyncio
async def test_fetch_vllm_version_returns_none_on_malformed_payload() -> None:
    port = _free_port()
    async with _serve(_version_app({"build": "abc123"}), port):
        version = await fetch_vllm_version(
            base_url=f"http://127.0.0.1:{port}/v1",
            timeout_seconds=5.0,
        )

    assert version is None


def test_workload_preflight_warns_for_humaneval_and_errors_for_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastcoder.benchmark import workloads as workloads_module
    from scripts import run_h100_baseline

    # Force the no-local-source path so HumanEval resolves to a warning, not an error,
    # deterministically even where the pod-only human_eval package is installed.
    monkeypatch.delenv("FASTCODER_HUMANEVAL_PATH", raising=False)
    monkeypatch.setattr(workloads_module, "_load_humaneval_problems_from_package", lambda: None)

    warnings, errors = run_h100_baseline._resolve_workloads_preflight(
        ["short_chat_256_256", "humaneval", "does_not_exist"]
    )

    assert any("humaneval" in warning for warning in warnings)
    assert any("does_not_exist" in error for error in errors)
    assert not any("short_chat_256_256" in warning for warning in warnings)


def _version_app(version_payload: dict[str, object]) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/version")
    async def version() -> dict[str, object]:
        return version_payload

    return app


def _valid_result_payload() -> dict[str, object]:
    config = BenchmarkConfig(experiment_name="schema_test", workloads=["short_chat"])
    workloads = get_workloads(config.workloads)
    result = build_benchmark_result(
        config=config,
        workloads=workloads,
        records=[
            RequestMetric(
                experiment="schema_test",
                workload="short_chat",
                concurrency=1,
                request_id="req-1",
                ok=True,
                status_code=200,
                start_time=0.0,
                end_time=1.0,
                first_token_time=None,
                output_tokens=2,
                output_token_source=TOKEN_SOURCE_APPROXIMATE,
                response_chars=10,
            )
        ],
        gpu_snapshot=GPUSnapshot(),
    )
    return result.model_dump(mode="json")


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        check=False,
        capture_output=True,
        text=True,
    )


@asynccontextmanager
async def _serve(app: object, port: int) -> AsyncIterator[None]:
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    try:
        await _wait_for_health(port)
        yield
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=5)


async def _wait_for_health(port: int) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + 5.0
    async with httpx.AsyncClient(timeout=1.0) as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.05)
    raise TimeoutError(f"server did not become healthy at {url}")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
