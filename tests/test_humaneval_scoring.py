from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from fastcoder.benchmark import humaneval_scoring
from fastcoder.benchmark.humaneval_scoring import (
    HumanEvalScoringUnavailableError,
    extract_solution_code,
    pass_at_1,
    score_completion,
    score_completions,
)
from fastcoder.benchmark.workloads import load_humaneval_problems_by_task_id

_FIXTURE = Path(__file__).parent / "data" / "humaneval_sample.jsonl"


def _checker_passes_when_return_present(
    problem: dict[str, Any],
    completion: str,
    timeout: float,
    completion_id: int | None = None,
) -> dict[str, Any]:
    """Stand-in for human_eval.execution.check_correctness used in tests (no code execution)."""

    passed = "return" in completion
    return {
        "task_id": problem["task_id"],
        "passed": passed,
        "result": "passed" if passed else "failed: no return",
        "completion_id": completion_id,
    }


def test_extract_solution_code_prefers_python_fence() -> None:
    completion = (
        "Here is the solution:\n"
        "```\nprint('untagged block, should be ignored')\n```\n"
        "and the real one:\n"
        "```python\ndef f():\n    return 1\n```\n"
    )

    assert extract_solution_code(completion) == "def f():\n    return 1"


def test_extract_solution_code_accepts_py_and_python3_tags() -> None:
    assert extract_solution_code("```py\nreturn 1\n```") == "return 1"
    assert extract_solution_code("```python3\nreturn 2\n```") == "return 2"


def test_extract_solution_code_falls_back_to_first_generic_fence() -> None:
    completion = "prose\n```\ndef g():\n    return 2\n```\nmore prose"

    assert extract_solution_code(completion) == "def g():\n    return 2"


def test_extract_solution_code_falls_back_to_raw_text() -> None:
    assert extract_solution_code("\n    return a + b\n") == "    return a + b"


def test_extract_solution_code_preserves_internal_indentation() -> None:
    completion = "```python\ndef f():\n    if True:\n        return 1\n```"

    assert extract_solution_code(completion) == "def f():\n    if True:\n        return 1"


def test_extract_solution_code_empty() -> None:
    assert extract_solution_code("") == ""


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([], 0.0),
        ([True, True], 1.0),
        ([False, False], 0.0),
        ([True, False, True], 2 / 3),
    ],
)
def test_pass_at_1(flags: list[bool], expected: float) -> None:
    assert pass_at_1(flags) == pytest.approx(expected)


def test_load_humaneval_problems_by_task_id_carries_scoring_metadata() -> None:
    problems = load_humaneval_problems_by_task_id(path=str(_FIXTURE))

    assert set(problems) == {"HumanEval/0", "HumanEval/1"}
    assert problems["HumanEval/0"]["entry_point"] == "add"
    assert "def check(candidate)" in problems["HumanEval/0"]["test"]


def test_score_completion_uses_injected_checker_and_extracts_code() -> None:
    problem = {"task_id": "HumanEval/0", "entry_point": "add"}

    result = score_completion(
        problem=problem,
        completion_text="```python\ndef add(a, b):\n    return a + b\n```",
        check_correctness=_checker_passes_when_return_present,
    )

    assert result.task_id == "HumanEval/0"
    assert result.passed is True
    assert result.extracted_code == "def add(a, b):\n    return a + b"


def test_score_completions_missing_completion_counts_as_failure() -> None:
    problems = {
        "HumanEval/0": {"task_id": "HumanEval/0"},
        "HumanEval/1": {"task_id": "HumanEval/1"},
        "HumanEval/2": {"task_id": "HumanEval/2"},
    }
    completions = {
        "HumanEval/0": "```python\ndef add(a, b):\n    return a + b\n```",
        "HumanEval/1": "no code, just prose",
        # HumanEval/2 intentionally absent -> empty completion -> failure
    }

    score = score_completions(
        problems_by_task_id=problems,
        completions_by_task_id=completions,
        check_correctness=_checker_passes_when_return_present,
    )

    assert score.total == 3
    assert score.passed == 1
    assert score.pass_at_1 == pytest.approx(1 / 3)
    assert [case.task_id for case in score.cases] == [
        "HumanEval/0",
        "HumanEval/1",
        "HumanEval/2",
    ]


def test_score_completions_sorts_task_ids_numerically() -> None:
    problems = {f"HumanEval/{i}": {"task_id": f"HumanEval/{i}"} for i in (0, 2, 10)}

    score = score_completions(
        problems_by_task_id=problems,
        completions_by_task_id={},
        check_correctness=_checker_passes_when_return_present,
    )

    assert [case.task_id for case in score.cases] == [
        "HumanEval/0",
        "HumanEval/2",
        "HumanEval/10",
    ]


def test_scoring_requires_human_eval_package(monkeypatch: pytest.MonkeyPatch) -> None:
    # A None entry in sys.modules forces ImportError on the lazy import, deterministically
    # regardless of whether the pod-only human_eval package happens to be installed.
    monkeypatch.setitem(sys.modules, "human_eval", None)
    monkeypatch.setitem(sys.modules, "human_eval.execution", None)

    with pytest.raises(HumanEvalScoringUnavailableError, match="human_eval"):
        humaneval_scoring._load_check_correctness()

    with pytest.raises(HumanEvalScoringUnavailableError):
        score_completion(problem={"task_id": "HumanEval/0"}, completion_text="return 1")
