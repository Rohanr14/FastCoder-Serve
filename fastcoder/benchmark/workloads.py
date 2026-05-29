"""Benchmark workload definitions.

Two families live here:

* ``smoke_workloads`` — tiny, CPU-safe single-sample prompts used by local smoke runs.
* ``realistic_workloads`` — token-controlled, multi-sample workloads for the H100 baseline.

Realistic workloads carry many distinct samples so repeated requests do not all share an
identical prompt; that keeps a baseline from accidentally measuring prefix-cache best case.
Input/output token targets in the workload names are *nominal* — the cited numbers always come
from the API ``usage`` fields recorded per request, never from these targets.

The ``shared_prefix_4k_512`` workload is the deliberate exception: every sample shares one long
prefix (only the trailing question varies) so a prefix-caching ablation can measure KV reuse.
"""

from __future__ import annotations

import gzip
import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ChatMessage = dict[str, str]

_APPROX_TOKENS_PER_WORD = 1.3
_APPROX_CHARS_PER_TOKEN = 4
_REALISTIC_SAMPLE_POOL = 64

_TARGET_SHORT_PROMPT_TOKENS = 256
_TARGET_SHORT_OUTPUT_TOKENS = 256
_TARGET_LONG_PROMPT_TOKENS = 4096
_TARGET_LONG_OUTPUT_TOKENS = 512

_TARGET_HUMANEVAL_OUTPUT_TOKENS = 512
_HUMANEVAL_ENV_VAR = "FASTCODER_HUMANEVAL_PATH"


class HumanEvalUnavailableError(RuntimeError):
    """Raised when HumanEval-style prompts are requested before data is configured."""


@dataclass(frozen=True, slots=True)
class WorkloadSample:
    """One concrete prompt within a workload."""

    messages: tuple[ChatMessage, ...]
    sample_id: str | None = None


@dataclass(frozen=True, slots=True)
class Workload:
    name: str
    description: str
    samples: tuple[WorkloadSample, ...]
    max_output_tokens: int | None = None


def _single_sample_workload(
    *,
    name: str,
    description: str,
    messages: tuple[ChatMessage, ...],
    max_output_tokens: int | None = None,
) -> Workload:
    return Workload(
        name=name,
        description=description,
        samples=(WorkloadSample(messages=messages, sample_id=None),),
        max_output_tokens=max_output_tokens,
    )


def smoke_workloads() -> dict[str, Workload]:
    """Return tiny CPU-safe prompt workloads for local validation."""

    return {
        "short_chat": _single_sample_workload(
            name="short_chat",
            description="One-turn factual coding assistant smoke prompt.",
            messages=(
                {"role": "system", "content": "You are a concise coding assistant."},
                {"role": "user", "content": "In one sentence, what does a Python dict do?"},
            ),
        ),
        "code_completion": _single_sample_workload(
            name="code_completion",
            description="Tiny code-completion style prompt.",
            messages=(
                {"role": "system", "content": "Complete the Python function."},
                {"role": "user", "content": "def add(a: int, b: int) -> int:\n    "},
            ),
        ),
        "long_context_tiny": _single_sample_workload(
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
        "long_context": _single_sample_workload(
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


_SHORT_CHAT_SYSTEM = (
    "You are a senior software engineer. Give concise, correct, production-ready answers."
)
_LONG_CONTEXT_SYSTEM = (
    "You are a meticulous code reviewer. Answer only from the provided module."
)

_SHORT_CHAT_DESCRIPTION = (
    "Realistic single-turn coding Q&A. Nominal ~256 prompt / ~256 output tokens; the cited "
    "token counts come from API usage, not from this target. Each request uses a distinct "
    "prompt to avoid unintended prefix-cache reuse."
)
_LONG_CONTEXT_DESCRIPTION = (
    "Long-context code comprehension over a synthetic Python module. Nominal ~4096 prompt / "
    "~512 output tokens; the cited token counts come from API usage, not from this target. "
    "Each request uses a distinct module to avoid unintended prefix-cache reuse."
)

_SHORT_CHAT_QUESTIONS = (
    "How would you design a rate limiter that allows {n} requests per minute per API key?",
    "Our endpoint must stay under {n} ms p99 latency. What caching strategy do you recommend?",
    "Explain how to safely paginate a query that returns about {n} rows in Postgres.",
    "What is the cleanest way to retry a flaky HTTP call up to {n} times with backoff in Python?",
    "We deduplicate a stream of roughly {n} events per second. Which approach scales best?",
    "How should I shard a worker pool across {n} partitions without creating hot spots?",
    "Describe a memory-efficient way to stream a {n} MB CSV file and aggregate one column.",
    "What indexing strategy speeds up a lookup table with about {n} distinct keys?",
    "How do I implement an LRU cache holding {n} entries with O(1) get and put in Python?",
    "Our queue backs up at {n} messages. How do I add backpressure without dropping data?",
    "What is a correct way to batch {n} database inserts inside a single transaction?",
    "How would you test a function that must stay correct across {n} concurrent callers?",
)

_SENTENCE_BANK = (
    "The service runs in a single region behind a load balancer with autoscaling enabled.",
    "Traffic is bursty during business hours and nearly idle overnight and on weekends.",
    "Downstream dependencies include a relational database, an object store, and a message broker.",
    "We track p50, p95, and p99 latency along with error rate on every deployment.",
    "The team values readable code, small pull requests, and thorough test coverage.",
    "Configuration is injected through environment variables and validated at startup.",
    "Observability is provided through structured logs, metrics, and distributed traces.",
    "Backwards compatibility matters because several internal clients call this endpoint.",
    "The codebase is primarily Python with a few performance-critical paths in Rust.",
    "Deployments are automated through continuous integration with required status checks.",
    "Secrets live in a managed vault and are never committed to the repository.",
    "Capacity planning assumes steady linear growth over the next four quarters.",
    "Incident response is documented in runbooks that on-call engineers follow closely.",
    "Data retention rules keep aggregates around while raw events expire quickly.",
)

_WORD_BANK = (
    "ledger",
    "cache",
    "router",
    "parser",
    "scheduler",
    "encoder",
    "buffer",
    "session",
    "worker",
    "queue",
    "index",
    "matrix",
    "token",
    "stream",
    "packet",
    "record",
    "cursor",
    "handler",
    "planner",
    "monitor",
    "sampler",
    "builder",
    "tracker",
    "adapter",
)


def _count_words(text: str) -> int:
    return len(text.split())


def _filler_paragraph(start: int, target_words: int) -> str:
    """Deterministic, realistic-sounding filler of an approximate word count."""

    if target_words <= 0:
        return ""
    words: list[str] = []
    offset = 0
    while len(words) < target_words:
        sentence = _SENTENCE_BANK[(start + offset) % len(_SENTENCE_BANK)]
        words.extend(sentence.split())
        offset += 1
    return " ".join(words[:target_words])


def _fake_code_module(seed: int) -> tuple[str, str, int]:
    """Build a deterministic, seed-varied Python module plus its needle function."""

    target_chars = _TARGET_LONG_PROMPT_TOKENS * _APPROX_CHARS_PER_TOKEN
    lines: list[str] = []
    functions: list[tuple[str, int]] = []
    char_count = 0
    index = 0
    while char_count < target_chars:
        word = _WORD_BANK[(seed * 31 + index) % len(_WORD_BANK)]
        name = f"transform_{word}_{index}"
        const = (seed * 97 + index * 7) % 900 + 100
        block = [
            f"def {name}(value: int) -> int:",
            f"    # deterministic {word} stage {index}",
            f"    return value * {const} + {index}",
            "",
        ]
        lines.extend(block)
        char_count += sum(len(line) + 1 for line in block)
        functions.append((name, const))
        index += 1
    needle_name, needle_const = functions[(seed * 13) % len(functions)]
    return "\n".join(lines), needle_name, needle_const


def _build_short_chat_256_256() -> Workload:
    system_message: ChatMessage = {"role": "system", "content": _SHORT_CHAT_SYSTEM}
    target_total_words = round(_TARGET_SHORT_PROMPT_TOKENS / _APPROX_TOKENS_PER_WORD)
    total_user_words = max(1, target_total_words - _count_words(_SHORT_CHAT_SYSTEM))
    samples: list[WorkloadSample] = []
    for i in range(_REALISTIC_SAMPLE_POOL):
        question = _SHORT_CHAT_QUESTIONS[i % len(_SHORT_CHAT_QUESTIONS)].format(n=1000 + i)
        filler_words = max(1, total_user_words - _count_words(question))
        context = _filler_paragraph(start=i, target_words=filler_words)
        content = f"{question}\n\nProject context:\n{context}"
        user_message: ChatMessage = {"role": "user", "content": content}
        samples.append(
            WorkloadSample(
                messages=(system_message, user_message),
                sample_id=f"short_chat_256_256/{i}",
            )
        )
    return Workload(
        name="short_chat_256_256",
        description=_SHORT_CHAT_DESCRIPTION,
        samples=tuple(samples),
        max_output_tokens=_TARGET_SHORT_OUTPUT_TOKENS,
    )


def _build_long_context_4k_512() -> Workload:
    system_message: ChatMessage = {"role": "system", "content": _LONG_CONTEXT_SYSTEM}
    samples: list[WorkloadSample] = []
    for i in range(_REALISTIC_SAMPLE_POOL):
        module, needle_name, _needle_const = _fake_code_module(seed=i)
        content = (
            "Review the module below and answer the question.\n\n"
            f"Question: which integer constant does `{needle_name}` multiply its input by?\n\n"
            f"```python\n{module}\n```"
        )
        user_message: ChatMessage = {"role": "user", "content": content}
        samples.append(
            WorkloadSample(
                messages=(system_message, user_message),
                sample_id=f"long_context_4k_512/{i}",
            )
        )
    return Workload(
        name="long_context_4k_512",
        description=_LONG_CONTEXT_DESCRIPTION,
        samples=tuple(samples),
        max_output_tokens=_TARGET_LONG_OUTPUT_TOKENS,
    )


_SHARED_PREFIX_DESCRIPTION = (
    "Prefix-caching workload: one fixed ~4K-token Python module is shared verbatim across all "
    "samples, with only a short trailing question varying per sample. This is the realistic "
    "shared-context pattern (same codebase/system prompt, many questions) where vLLM automatic "
    "prefix caching can reuse the long prefix's KV. Token counts come from API usage."
)

_SHARED_PREFIX_QUESTIONS = (
    "Summarize in two sentences what this module is doing.",
    "Which functions look like near-duplicates that could be merged?",
    "Name one function whose constant multiplier looks suspiciously large.",
    "How would you add unit tests for the transforms in this module?",
    "Which stage would you refactor first for readability, and why?",
    "Is there any obvious dead code or unreachable branch here?",
    "How would you parallelize applying every transform to a list of inputs?",
    "What naming convention do these functions follow, and is it consistent?",
    "Which function would you benchmark first if throughput mattered?",
    "How would you document this module for a new teammate?",
    "What is a safe way to add input validation to these functions?",
    "Which transform would you cache results for, and how?",
)


def _build_shared_prefix_4k_512() -> Workload:
    system_message: ChatMessage = {"role": "system", "content": _LONG_CONTEXT_SYSTEM}
    module, _needle_name, _needle_const = _fake_code_module(seed=0)
    shared_context = (
        "Review the module below and answer the question that follows.\n\n"
        f"```python\n{module}\n```\n\n"
    )
    samples: list[WorkloadSample] = []
    for i in range(_REALISTIC_SAMPLE_POOL):
        question = _SHARED_PREFIX_QUESTIONS[i % len(_SHARED_PREFIX_QUESTIONS)]
        # shared_context is byte-identical for every sample; only the trailing question varies,
        # so the multi-thousand-token prefix is a true shared prefix for vLLM's KV cache.
        content = f"{shared_context}Question {i}: {question}"
        user_message: ChatMessage = {"role": "user", "content": content}
        samples.append(
            WorkloadSample(
                messages=(system_message, user_message),
                sample_id=f"shared_prefix_4k_512/{i}",
            )
        )
    return Workload(
        name="shared_prefix_4k_512",
        description=_SHARED_PREFIX_DESCRIPTION,
        samples=tuple(samples),
        max_output_tokens=_TARGET_LONG_OUTPUT_TOKENS,
    )


_REALISTIC_BUILDERS: dict[str, Callable[[], Workload]] = {
    "short_chat_256_256": _build_short_chat_256_256,
    "long_context_4k_512": _build_long_context_4k_512,
    "shared_prefix_4k_512": _build_shared_prefix_4k_512,
}


def realistic_workloads() -> dict[str, Workload]:
    """Build all token-controlled multi-sample workloads (used by the H100 baseline)."""

    return {name: build() for name, build in _REALISTIC_BUILDERS.items()}


def get_workloads(names: Iterable[str]) -> list[Workload]:
    """Resolve workload names to workload objects."""

    registry = smoke_workloads()
    workloads: list[Workload] = []
    for name in names:
        if name in {"humaneval", "human_eval"}:
            workloads.extend(load_humaneval_workloads())
            continue
        if name in registry:
            workloads.append(registry[name])
            continue
        builder = _REALISTIC_BUILDERS.get(name)
        if builder is not None:
            workloads.append(builder())
            continue
        available = ", ".join(sorted([*registry, *_REALISTIC_BUILDERS, "humaneval"]))
        msg = f"unknown workload '{name}'. Available workloads: {available}"
        raise ValueError(msg)
    return workloads


_HUMANEVAL_SYSTEM = (
    "You are an expert Python programmer. Write correct, self-contained code."
)
_HUMANEVAL_DESCRIPTION = (
    "HumanEval code-generation benchmark (164 problems) used for pass@1 quality. The prompt asks "
    "for the full function in a single Python code block; output cap is nominal ~512 tokens and "
    "actual token counts come from API usage. Prompt data is loaded lazily and is not bundled "
    "in CI."
)


def _humaneval_messages(prompt: str) -> tuple[ChatMessage, ...]:
    user_content = (
        "Complete the following Python function. Respond with the full function definition, "
        "including the signature, inside a single ```python code block and nothing else.\n\n"
        f"```python\n{prompt}\n```"
    )
    return (
        {"role": "system", "content": _HUMANEVAL_SYSTEM},
        {"role": "user", "content": user_content},
    )


def _humaneval_samples_from_problems(
    problems: Iterable[dict[str, Any]],
) -> tuple[WorkloadSample, ...]:
    samples: list[WorkloadSample] = []
    for record in problems:
        task_id = record.get("task_id")
        prompt = record.get("prompt")
        if not isinstance(task_id, str) or not isinstance(prompt, str):
            msg = "each HumanEval problem must have string 'task_id' and 'prompt' fields"
            raise HumanEvalUnavailableError(msg)
        samples.append(
            WorkloadSample(messages=_humaneval_messages(prompt), sample_id=task_id)
        )
    if not samples:
        msg = "HumanEval problem source contained no problems"
        raise HumanEvalUnavailableError(msg)
    return tuple(samples)


def _read_humaneval_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        msg = f"HumanEval JSONL path does not exist: {path}"
        raise HumanEvalUnavailableError(msg)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            lines = handle.readlines()
    else:
        with path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()

    problems: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            msg = f"invalid JSON on line {line_number} of {path}: {exc}"
            raise HumanEvalUnavailableError(msg) from exc
        if not isinstance(record, dict):
            msg = f"line {line_number} of {path} is not a JSON object"
            raise HumanEvalUnavailableError(msg)
        problems.append(record)
    return problems


def _load_humaneval_problems_from_package() -> list[dict[str, Any]] | None:
    try:
        from human_eval.data import read_problems
    except ImportError:
        return None
    return list(read_problems().values())


def _load_humaneval_problems(path: str | None) -> list[dict[str, Any]]:
    resolved = path or os.environ.get(_HUMANEVAL_ENV_VAR)
    if resolved:
        return _read_humaneval_jsonl(Path(resolved))
    from_package = _load_humaneval_problems_from_package()
    if from_package is not None:
        return from_package
    msg = (
        "HumanEval workload requested, but no HumanEval problem source is available. Provide a "
        f"JSONL(.gz) path (argument or ${_HUMANEVAL_ENV_VAR}), or install the eval extra "
        "(pip install -e '.[eval]'). Local CI intentionally does not bundle HumanEval data."
    )
    raise HumanEvalUnavailableError(msg)


def load_humaneval_problems_by_task_id(path: str | None = None) -> dict[str, dict[str, Any]]:
    """Return HumanEval problems keyed by ``task_id`` for scoring.

    Uses the same source resolution as :func:`load_humaneval_workloads` (explicit ``path`` →
    ``FASTCODER_HUMANEVAL_PATH`` → ``human_eval`` package). Unlike the workload loader, this keeps
    the full problem record (including ``entry_point`` and ``test``) so pass@1 scoring can rebuild
    the check program by task id. Raises ``HumanEvalUnavailableError`` when no source is available.
    """

    problems = _load_humaneval_problems(path)
    by_task_id: dict[str, dict[str, Any]] = {}
    for record in problems:
        task_id = record.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            msg = "each HumanEval problem must have a non-empty string 'task_id'"
            raise HumanEvalUnavailableError(msg)
        by_task_id[task_id] = record
    if not by_task_id:
        msg = "HumanEval problem source contained no problems"
        raise HumanEvalUnavailableError(msg)
    return by_task_id


def load_humaneval_workloads(path: str | None = None) -> list[Workload]:
    """Load the HumanEval problems into a single multi-sample workload.

    Sources are tried in order: an explicit JSONL(.gz) ``path``, the ``FASTCODER_HUMANEVAL_PATH``
    environment variable, then the optional ``human_eval`` package (``read_problems``). Each
    problem becomes a sample whose ``sample_id`` is the HumanEval ``task_id`` (e.g.
    ``HumanEval/0``); scoring metadata (``entry_point``/``test``) is intentionally not carried here
    and is reloaded by task id at scoring time. Raises ``HumanEvalUnavailableError`` when no source
    is available, which keeps local CI download-free.
    """

    problems = _load_humaneval_problems(path)
    samples = _humaneval_samples_from_problems(problems)
    return [
        Workload(
            name="humaneval",
            description=_HUMANEVAL_DESCRIPTION,
            samples=samples,
            max_output_tokens=_TARGET_HUMANEVAL_OUTPUT_TOKENS,
        )
    ]
