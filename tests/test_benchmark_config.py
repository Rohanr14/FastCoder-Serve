from __future__ import annotations

from pathlib import Path

from fastcoder.benchmark.config import load_benchmark_config
from fastcoder.benchmark.workloads import get_workloads


def test_load_local_smoke_config() -> None:
    config = load_benchmark_config(Path("configs/local_smoke.yaml"))

    assert config.experiment_name == "local_smoke"
    assert config.model_server.chat_completions_url == "http://localhost:9000/v1/chat/completions"
    assert config.concurrency == [1, 2]
    assert config.output_path == Path("results/local_smoke.json")


def test_configured_workloads_exist() -> None:
    config = load_benchmark_config(Path("configs/local_smoke.yaml"))

    workloads = get_workloads(config.workloads)

    assert [workload.name for workload in workloads] == [
        "short_chat",
        "code_completion",
        "long_context_tiny",
    ]


def test_load_streaming_smoke_config() -> None:
    config = load_benchmark_config(Path("configs/local_smoke_streaming.yaml"))

    assert config.experiment_name == "local_smoke_streaming"
    assert config.stream is True
    assert config.output_path == Path("results/local_smoke_streaming.json")


def test_load_baseline_fp16_config_is_latency_sweep_without_humaneval() -> None:
    config = load_benchmark_config(Path("configs/baseline_fp16.yaml"))

    assert config.experiment_name == "baseline_fp16"
    assert config.model_server.model == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert config.model_server.dtype == "fp16"
    assert config.hardware.gpu_name == "NVIDIA H100 80GB HBM3"
    assert config.hardware.hourly_cost_usd == 2.20
    assert config.serving_backend == "vllm"
    assert config.vllm_version is None
    assert config.speculative_method is None
    assert config.speculative_model is None
    assert config.software.serving_backend == "vllm"
    assert config.software.vllm_version is None
    assert config.speculation.method is None
    assert config.speculation.draft_model is None
    assert config.measurement_hypothesis is not None
    # pass@1 is measured by the dedicated single-pass eval config, not this latency sweep.
    assert config.workloads == ["short_chat_256_256", "long_context_4k_512"]
    assert "humaneval" not in config.workloads


def test_load_humaneval_fp16_config_is_single_pass_eval() -> None:
    config = load_benchmark_config(Path("configs/humaneval_fp16.yaml"))

    assert config.experiment_name == "humaneval_fp16"
    assert config.model_server.model == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert config.model_server.dtype == "fp16"
    assert config.workloads == ["humaneval"]
    assert config.max_tokens == 512
    assert config.temperature == 0.0
    assert config.output_path == Path("results/humaneval_fp16.json")
    assert config.measurement_hypothesis is not None
