"""Small benchmark workload definitions for local smoke tests."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

ChatMessage = dict[str, str]


class HumanEvalUnavailableError(RuntimeError):
    """Raised when HumanEval-style prompts are requested before data is configured."""


@dataclass(frozen=True, slots=True)
class Workload:
    name: str
    description: str
    messages: tuple[ChatMessage, ...]


def smoke_workloads() -> dict[str, Workload]:
    """Return tiny CPU-safe prompt workloads for local validation."""

    return {
        "short_chat": Workload(
            name="short_chat",
            description="One-turn factual coding assistant smoke prompt.",
            messages=(
                {"role": "system", "content": "You are a concise coding assistant."},
                {"role": "user", "content": "In one sentence, what does a Python dict do?"},
            ),
        ),
        "code_completion": Workload(
            name="code_completion",
            description="Tiny code-completion style prompt.",
            messages=(
                {"role": "system", "content": "Complete the Python function."},
                {"role": "user", "content": "def add(a: int, b: int) -> int:\n    "},
            ),
        ),
        "long_context_tiny": Workload(
            name="long_context_tiny",
            description="Small synthetic context prompt, not a real long-context benchmark.",
            messages=(
                {"role": "system", "content": "Summarize only the requested signal."},
                {
                    "role": "user",
                    "content": (
                        "Context: alpha beta gamma. alpha beta gamma. alpha beta gamma. "
                        "Question: Which word appears between alpha and gamma?"
                    ),
                },
            ),
        ),
        "long_context": Workload(
            name="long_context",
            description=(
                "Placeholder for the future H100 long-context workload. This local prompt is "
                "only a structural stand-in until the methodology defines the full 4K input."
            ),
            messages=(
                {"role": "system", "content": "Answer from the provided context only."},
                {
                    "role": "user",
                    "content": (
                        "Context: FastCoder-Serve measures inference latency, throughput, "
                        "quality, memory, and cost for code LLM serving. "
                        "Question: What category of system is FastCoder-Serve?"
                    ),
                },
            ),
        ),
    }


def get_workloads(names: Iterable[str]) -> list[Workload]:
    """Resolve workload names to workload objects."""

    registry = smoke_workloads()
    workloads: list[Workload] = []
    for name in names:
        if name in {"humaneval", "human_eval"}:
            workloads.extend(load_humaneval_workloads())
            continue
        try:
            workloads.append(registry[name])
        except KeyError as exc:
            available = ", ".join(sorted([*registry, "humaneval"]))
            msg = f"unknown workload '{name}'. Available workloads: {available}"
            raise ValueError(msg) from exc
    return workloads


def load_humaneval_workloads(path: str | None = None) -> list[Workload]:
    """Skeleton loader for future HumanEval prompts.

    The dependency and prompt data are intentionally not part of local CI. Future H100/eval work can
    wire this interface to `lm-evaluation-harness` or a checked-in prompt extraction artifact.
    """

    if path is None:
        msg = (
            "HumanEval workload requested, but no HumanEval prompt source is configured. "
            "Install/configure the future HumanEval data path before running this workload; "
            "local smoke tests intentionally do not require it."
        )
        raise HumanEvalUnavailableError(msg)
    msg = (
        "HumanEval prompt loading from a path is not implemented yet. "
        "This skeleton exists so future H100 eval work has a stable interface."
    )
    raise HumanEvalUnavailableError(msg)
