from __future__ import annotations

from itertools import pairwise

import pytest

from fastcoder.benchmark.openloop import (
    OPEN_LOOP_SCHEMA_VERSION,
    OpenLoopRequest,
    poisson_arrival_offsets,
    summarize_open_loop,
    validate_open_loop_result,
)


def test_poisson_offsets_are_deterministic_increasing_and_well_scaled() -> None:
    a = poisson_arrival_offsets(10.0, 500, seed=7)
    b = poisson_arrival_offsets(10.0, 500, seed=7)
    assert a == b  # deterministic for a fixed seed
    assert len(a) == 500
    assert all(later > earlier for earlier, later in pairwise(a))  # strictly increasing
    # mean inter-arrival ~ 1/rate = 0.1s (loose bound — it's a random process)
    mean_gap = a[-1] / len(a)
    assert 0.07 < mean_gap < 0.13


def test_poisson_rejects_nonpositive_rate() -> None:
    with pytest.raises(ValueError, match="rate_rps"):
        poisson_arrival_offsets(0.0, 10, seed=1)


def _req(
    arrival: float, latency: float | None, ttft: float | None, ok: bool = True
) -> OpenLoopRequest:
    return OpenLoopRequest(
        arrival_offset=arrival, latency_seconds=latency, ttft_seconds=ttft, ok=ok
    )


def test_summarize_computes_slo_violation_and_percentiles() -> None:
    # 4 requests: two meet TTFT<=0.25s, one misses TTFT, one failed (timeout/overload).
    requests = [
        _req(0.0, 1.0, 0.10),
        _req(0.1, 1.2, 0.20),
        _req(0.2, 3.0, 0.40),           # TTFT 0.40 > 0.25 -> violation
        _req(0.3, None, None, ok=False),  # failure -> violation
    ]
    summary = summarize_open_loop(
        target_rate_rps=10.0, requests=requests, slo_ttft_seconds=0.25, slo_latency_seconds=5.0
    )
    assert summary["request_count"] == 4
    assert summary["success_count"] == 3
    assert summary["error_count"] == 1
    assert summary["slo_violation_rate"] == 0.5  # 2 of 4 violate
    assert summary["p50_latency_seconds"] is not None
    assert summary["achieved_throughput_rps"] is not None


def test_summarize_handles_total_overload_all_failed() -> None:
    requests = [_req(float(i) * 0.1, None, None, ok=False) for i in range(10)]
    summary = summarize_open_loop(target_rate_rps=200.0, requests=requests, slo_ttft_seconds=0.25)
    assert summary["success_count"] == 0
    assert summary["slo_violation_rate"] == 1.0
    assert summary["achieved_throughput_rps"] is None
    assert summary["p99_latency_seconds"] is None


def test_validate_open_loop_result_flags_bad_points() -> None:
    good = {
        "schema_version": OPEN_LOOP_SCHEMA_VERSION,
        "arrival_rate_points": [
            {
                "target_rate_rps": 10.0,
                "request_count": 2,
                "success_count": 2,
                "error_count": 0,
                "slo_violation_rate": 0.0,
                "p50_latency_seconds": 1.0,
                "p95_latency_seconds": 1.5,
                "p99_latency_seconds": 2.0,
            }
        ],
    }
    assert validate_open_loop_result(good) == []

    bad = {
        "schema_version": OPEN_LOOP_SCHEMA_VERSION,
        "arrival_rate_points": [
            {
                "target_rate_rps": 10.0,
                "request_count": 2,
                "success_count": 1,
                "error_count": 0,  # 1 + 0 != 2
                "slo_violation_rate": 1.5,  # outside [0, 1]
                "p50_latency_seconds": 2.0,
                "p95_latency_seconds": 1.0,  # not monotonic
                "p99_latency_seconds": 1.5,
            }
        ],
    }
    issues = validate_open_loop_result(bad)
    assert len(issues) == 3
