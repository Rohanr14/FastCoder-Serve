"""HumanEval pass@1 scoring.

Responsibilities are deliberately split by safety:

* ``extract_solution_code`` and ``pass_at_1`` are pure, deterministic helpers. They never execute
  model output and are fully unit tested locally.
* ``score_completion`` / ``score_completions`` execute model-generated code and therefore run ONLY
  where the optional ``human_eval`` package is installed (the eval pod; ``pip install -e .[eval]``).
  Execution is delegated to ``human_eval.execution.check_correctness``, which runs each candidate in
  a guarded subprocess with a hard timeout — we deliberately never roll our own ``exec``. The import
  is lazy, so importing this module never imports ``human_eval`` and local CI stays download- and
  execution-free.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_TIMEOUT_SECONDS = 10.0

# Matches a fenced code block, capturing an optional language tag and the body.
_FENCED_BLOCK = re.compile(
    r"```[ \t]*(?P<lang>[A-Za-z0-9_+-]*)[ \t]*\r?\n(?P<body>.*?)```",
    re.DOTALL,
)
_PYTHON_LANGS = {"python", "py", "python3"}


class HumanEvalScoringUnavailableError(RuntimeError):
    """Raised when pass@1 scoring is attempted without the pod-only ``human_eval`` package."""


@dataclass(frozen=True, slots=True)
class HumanEvalCaseResult:
    """Outcome of scoring one completion against its HumanEval problem."""

    task_id: str
    passed: bool
    detail: str
    extracted_code: str


@dataclass(frozen=True, slots=True)
class HumanEvalScore:
    """Aggregate pass@1 result over a set of HumanEval problems."""

    pass_at_1: float
    total: int
    passed: int
    cases: tuple[HumanEvalCaseResult, ...]


def extract_solution_code(completion_text: str) -> str:
    """Extract the Python solution from a model completion.

    Preference order: a ```python fenced block, then the first fenced block of any language, then
    the raw text. Only surrounding blank lines are trimmed so internal indentation survives — the
    official HumanEval prompt stub is prepended at scoring time, so both a full function definition
    and a bare indented body remain valid.
    """

    if not completion_text:
        return ""
    blocks = list(_FENCED_BLOCK.finditer(completion_text))
    for match in blocks:
        if match.group("lang").lower() in _PYTHON_LANGS:
            return match.group("body").strip("\n")
    if blocks:
        return blocks[0].group("body").strip("\n")
    return completion_text.strip("\n")


def pass_at_1(passed_flags: Sequence[bool]) -> float:
    """Fraction of problems whose single sampled completion passed (the n=1, k=1 estimator)."""

    total = len(passed_flags)
    if total == 0:
        return 0.0
    return sum(1 for flag in passed_flags if flag) / total


def score_completion(
    *,
    problem: Mapping[str, Any],
    completion_text: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    check_correctness: Any | None = None,
) -> HumanEvalCaseResult:
    """Score one completion against its HumanEval problem in a guarded subprocess.

    Executes model-generated code via ``human_eval.execution.check_correctness`` (pod-only). The
    ``check_correctness`` argument is an injection seam for tests; production passes ``None`` so the
    real, sandboxed implementation is imported lazily.
    """

    runner = check_correctness or _load_check_correctness()
    code = extract_solution_code(completion_text)
    task_id = str(problem.get("task_id", ""))
    outcome = runner(problem, code, timeout_seconds)
    passed, detail = _interpret_outcome(outcome)
    return HumanEvalCaseResult(
        task_id=task_id,
        passed=passed,
        detail=detail,
        extracted_code=code,
    )


def score_completions(
    *,
    problems_by_task_id: Mapping[str, Mapping[str, Any]],
    completions_by_task_id: Mapping[str, str],
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    check_correctness: Any | None = None,
) -> HumanEvalScore:
    """Score every problem and compute pass@1 over the full problem set.

    A problem with no captured completion is scored as a failure (empty completion) so the
    denominator is the full problem set rather than only the answered problems.
    """

    runner = check_correctness or _load_check_correctness()
    cases: list[HumanEvalCaseResult] = []
    for task_id in sorted(problems_by_task_id, key=_task_id_sort_key):
        cases.append(
            score_completion(
                problem=problems_by_task_id[task_id],
                completion_text=completions_by_task_id.get(task_id, ""),
                timeout_seconds=timeout_seconds,
                check_correctness=runner,
            )
        )
    passed_flags = [case.passed for case in cases]
    return HumanEvalScore(
        pass_at_1=pass_at_1(passed_flags),
        total=len(cases),
        passed=sum(1 for flag in passed_flags if flag),
        cases=tuple(cases),
    )


def _interpret_outcome(outcome: Any) -> tuple[bool, str]:
    if isinstance(outcome, Mapping):
        passed = bool(outcome.get("passed", False))
        detail = str(outcome.get("result", "passed" if passed else "failed"))
        return passed, detail
    passed = bool(outcome)
    return passed, "passed" if passed else "failed"


def _task_id_sort_key(task_id: str) -> tuple[str, int, str]:
    prefix, _, suffix = task_id.rpartition("/")
    if suffix.isdigit():
        return (prefix, int(suffix), "")
    return (task_id, -1, task_id)


def _load_check_correctness() -> Any:
    try:
        from human_eval.execution import check_correctness
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatched sys.modules
        msg = (
            "HumanEval scoring requires the 'human_eval' package, which is intentionally not "
            "installed locally or in CI. Install the eval extra on the pod "
            "(pip install -e .[eval]). Scoring executes model-generated code and must run only in "
            "the sandboxed pod environment."
        )
        raise HumanEvalScoringUnavailableError(msg) from exc
    return check_correctness
