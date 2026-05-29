from __future__ import annotations

from fastcoder.benchmark.slo import (
    OperatingPoint,
    max_throughput_under_slo,
    operating_points_from_result,
)


def _points() -> list[OperatingPoint]:
    # Mirrors the shape of the real data, including a non-monotonic p99 TTFT (c8 > c32) and
    # long-context saturation (throughput peaks at c32, collapses at c64).
    return [
        OperatingPoint("FP8", "short", 8, 0.116, 1.254, 1605.0, 0.381),
        OperatingPoint("FP8", "short", 32, 0.142, 1.371, 4757.0, 0.128),
        OperatingPoint("FP8", "short", 64, 0.449, 1.767, 9204.0, 0.066),
        OperatingPoint("FP8", "long", 32, 0.553, 0.652, 1156.0, 0.529),
        OperatingPoint("FP8", "long", 64, 1.872, 2.026, 757.0, 0.807),
    ]


def test_strict_slo_maximises_over_qualifying_points_not_first_violation() -> None:
    # Under 250 ms p99 TTFT: c64 (0.449) is excluded, but c32 (0.142) qualifies and beats c8 even
    # though the intervening c8->c32 tail is non-monotonic.
    best = max_throughput_under_slo(_points(), slo_seconds=0.250)
    assert best is not None
    assert (best.workload, best.concurrency) == ("short", 32)
    assert best.output_token_throughput_tps == 4757.0


def test_relaxed_slo_admits_higher_concurrency() -> None:
    best = max_throughput_under_slo(_points(), slo_seconds=1.0)
    assert best is not None
    assert (best.workload, best.concurrency) == ("short", 64)


def test_no_point_meets_impossible_slo_returns_none() -> None:
    assert max_throughput_under_slo(_points(), slo_seconds=0.001) is None


def test_workload_filter_and_unconstrained_peak_exposes_saturation() -> None:
    # Unconstrained peak for long-context is c32 (1156 tok/s), not c64 (757) — saturation.
    peak = max_throughput_under_slo(_points(), slo_seconds=float("inf"), workload="long")
    assert peak is not None
    assert (peak.workload, peak.concurrency) == ("long", 32)


def test_operating_points_from_result_parses_per_workload_metrics() -> None:
    result = {
        "per_workload_metrics": [
            {
                "workload": "short",
                "concurrency": 8,
                "metrics": {
                    "p99_ttft_seconds": 0.1,
                    "p99_latency_seconds": 1.0,
                    "output_token_throughput_tps": 1600.0,
                    "cost_per_1m_output_tokens_usd": 0.38,
                },
            }
        ]
    }
    points = operating_points_from_result(result, "FP8")
    assert len(points) == 1
    assert points[0].label == "FP8"
    assert points[0].concurrency == 8
    assert points[0].output_token_throughput_tps == 1600.0
