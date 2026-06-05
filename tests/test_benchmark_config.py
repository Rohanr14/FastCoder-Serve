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


def test_load_week2_quantization_configs() -> None:
    latency = ["short_chat_256_256", "long_context_4k_512"]
    expected = {
        "baseline_fp8": ("fp8", latency, "baseline_fp8"),
        "humaneval_fp8": ("fp8", ["humaneval"], "humaneval_fp8"),
        "baseline_awq": ("awq_marlin", latency, "baseline_awq"),
        "humaneval_awq": ("awq_marlin", ["humaneval"], "humaneval_awq"),
    }
    for name, (quantization, workloads, stem) in expected.items():
        config = load_benchmark_config(Path(f"configs/{name}.yaml"))
        assert config.experiment_name == name
        assert config.model_server.model.startswith("Qwen/Qwen2.5-Coder-7B-Instruct")
        assert config.model_server.quantization == quantization
        assert config.model_server.dtype == "fp16"
        assert config.workloads == workloads
        assert config.output_path == Path(f"results/{stem}.json")
        assert config.measurement_hypothesis is not None


def test_load_prefix_cache_configs_and_shared_prefix_workload() -> None:
    for name in ("prefix_cache_on", "prefix_cache_off"):
        config = load_benchmark_config(Path(f"configs/{name}.yaml"))
        assert config.experiment_name == name
        assert config.model_server.quantization == "fp8"
        assert config.workloads == ["shared_prefix_4k_512"]
        assert config.output_path == Path(f"results/{name}.json")

    [workload] = get_workloads(["shared_prefix_4k_512"])
    contents = [sample.messages[-1]["content"] for sample in workload.samples]
    assert len(contents) >= 8
    # Every sample diverges only in the trailing question, so they share one long prefix.
    shared_prefix = contents[0].split("Question")[0]
    assert len(shared_prefix) > 4000
    assert all(content.startswith(shared_prefix) for content in contents)
    assert len(set(contents)) == len(contents)


def test_load_speculative_decoding_configs() -> None:
    expected = {
        "spec_draft_fp8": ("draft", "Qwen/Qwen2.5-Coder-0.5B-Instruct"),
        "spec_ngram_fp8": ("ngram", None),
    }
    for name, (method, draft_model) in expected.items():
        config = load_benchmark_config(Path(f"configs/{name}.yaml"))
        assert config.experiment_name == name
        assert config.model_server.quantization == "fp8"
        assert config.workloads == ["short_chat_256_256", "long_context_4k_512"]
        assert config.output_path == Path(f"results/{name}.json")
        assert config.speculation.method == method
        assert config.speculation.draft_model == draft_model
        assert config.speculation.num_speculative_tokens == 5


def test_load_openloop_config() -> None:
    config = load_benchmark_config(Path("configs/openloop_fp8.yaml"))
    assert config.experiment_name == "openloop_fp8"
    assert config.workloads == ["short_chat_256_256"]
    assert config.open_loop.arrival_rates_rps == [5, 10, 20, 30, 40, 50, 60, 80]
    assert config.open_loop.requests_per_rate == 120
    assert config.open_loop.slo_ttft_ms == 250
    assert config.output_path == Path("results/openloop_fp8.json")
