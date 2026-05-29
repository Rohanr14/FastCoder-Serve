from __future__ import annotations

import pytest

from fastcoder.benchmark import workloads as workloads_module
from fastcoder.benchmark.gpu import GPUSnapshot, collect_gpu_snapshot
from fastcoder.benchmark.workloads import (
    HumanEvalUnavailableError,
    get_workloads,
    realistic_workloads,
)


def test_gpu_snapshot_is_cpu_safe() -> None:
    snapshot = collect_gpu_snapshot()

    assert isinstance(snapshot, GPUSnapshot)


def test_humaneval_request_fails_with_clear_message(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the no-source path so the test is deterministic even where human_eval is installed.
    monkeypatch.delenv("FASTCODER_HUMANEVAL_PATH", raising=False)
    monkeypatch.setattr(workloads_module, "_load_humaneval_problems_from_package", lambda: None)

    with pytest.raises(HumanEvalUnavailableError, match="HumanEval workload requested"):
        get_workloads(["humaneval"])


def test_unknown_workload_reports_available_names() -> None:
    with pytest.raises(ValueError, match="unknown workload"):
        get_workloads(["does_not_exist"])


def test_realistic_workloads_are_multi_sample_and_token_controlled() -> None:
    workloads = realistic_workloads()
    assert set(workloads) == {"short_chat_256_256", "long_context_4k_512"}

    short = workloads["short_chat_256_256"]
    assert short.max_output_tokens == 256
    assert len(short.samples) >= 8
    short_prompts = [sample.messages[-1]["content"] for sample in short.samples]
    # Distinct prompts so repeated requests don't all hit the same cached prefix.
    assert len(set(short_prompts)) == len(short_prompts)
    assert all(sample.sample_id is not None for sample in short.samples)
    # Nominal ~256-token prompt; loose bound on the whitespace word count.
    assert 120 <= len(short_prompts[0].split()) <= 360

    long = workloads["long_context_4k_512"]
    assert long.max_output_tokens == 512
    assert len(long.samples) >= 8
    long_prompts = [sample.messages[-1]["content"] for sample in long.samples]
    assert len(set(long_prompts)) == len(long_prompts)
    assert all(sample.sample_id is not None for sample in long.samples)
    # Nominal ~4096-token code prompt; code is token-dense, so bound on characters.
    assert len(long_prompts[0]) > 8000


def test_get_workloads_resolves_realistic_names_in_order() -> None:
    workloads = get_workloads(["long_context_4k_512", "short_chat_256_256"])

    assert [workload.name for workload in workloads] == [
        "long_context_4k_512",
        "short_chat_256_256",
    ]
