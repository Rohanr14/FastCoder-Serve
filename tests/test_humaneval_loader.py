from __future__ import annotations

from pathlib import Path

import pytest

from fastcoder.benchmark import workloads
from fastcoder.benchmark.workloads import (
    HumanEvalUnavailableError,
    get_workloads,
    load_humaneval_workloads,
)

_FIXTURE = Path(__file__).parent / "data" / "humaneval_sample.jsonl"


def test_loads_humaneval_from_jsonl_path() -> None:
    result = load_humaneval_workloads(path=str(_FIXTURE))

    assert len(result) == 1
    workload = result[0]
    assert workload.name == "humaneval"
    assert workload.max_output_tokens == 512
    assert [sample.sample_id for sample in workload.samples] == ["HumanEval/0", "HumanEval/1"]

    first = workload.samples[0]
    assert first.messages[0]["role"] == "system"
    assert "def add(a: int, b: int)" in first.messages[-1]["content"]


def test_loads_humaneval_from_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTCODER_HUMANEVAL_PATH", str(_FIXTURE))

    workload = get_workloads(["humaneval"])[0]

    assert workload.name == "humaneval"
    assert len(workload.samples) == 2


def test_explicit_path_overrides_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTCODER_HUMANEVAL_PATH", "/no/such/humaneval.jsonl")

    workload = load_humaneval_workloads(path=str(_FIXTURE))[0]

    assert len(workload.samples) == 2


def test_missing_path_raises_clear_error() -> None:
    with pytest.raises(HumanEvalUnavailableError, match="does not exist"):
        load_humaneval_workloads(path="/no/such/humaneval.jsonl")


def test_no_source_raises_with_clear_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FASTCODER_HUMANEVAL_PATH", raising=False)
    monkeypatch.setattr(workloads, "_load_humaneval_problems_from_package", lambda: None)

    with pytest.raises(HumanEvalUnavailableError, match="HumanEval workload requested"):
        get_workloads(["humaneval"])
