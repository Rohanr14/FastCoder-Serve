from __future__ import annotations

import pytest

from fastcoder.benchmark.gpu import GPUSnapshot, collect_gpu_snapshot
from fastcoder.benchmark.workloads import HumanEvalUnavailableError, get_workloads


def test_gpu_snapshot_is_cpu_safe() -> None:
    snapshot = collect_gpu_snapshot()

    assert isinstance(snapshot, GPUSnapshot)


def test_humaneval_request_fails_with_clear_message() -> None:
    with pytest.raises(HumanEvalUnavailableError, match="HumanEval workload requested"):
        get_workloads(["humaneval"])
