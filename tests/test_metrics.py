from __future__ import annotations

import pytest

from fastcoder.benchmark.metrics import (
    TOKEN_SOURCE_APPROXIMATE,
    TOKEN_SOURCE_USAGE,
    RequestMetric,
    compute_summary,
    cost_per_1m_output_tokens,
    estimate_token_count,
    inter_token_latencies,
    percentile,
)


def test_percentile_linear_interpolation() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([10.0], 99) == 10.0
    assert percentile([], 50) is None


def test_estimate_token_count_is_whitespace_based() -> None:
    assert estimate_token_count("alpha beta gamma") == 3
    assert estimate_token_count("   ") == 0


def test_compute_summary_counts_success_errors_and_cost() -> None:
    records = [
        RequestMetric(
            experiment="test",
            workload="short_chat",
            concurrency=1,
            request_id="a",
            ok=True,
            status_code=200,
            start_time=0.0,
            end_time=1.0,
            first_token_time=0.25,
            output_tokens=10,
            output_token_source=TOKEN_SOURCE_USAGE,
        ),
        RequestMetric(
            experiment="test",
            workload="short_chat",
            concurrency=1,
            request_id="b",
            ok=False,
            status_code=500,
            start_time=0.0,
            end_time=0.5,
            first_token_time=None,
            output_tokens=0,
            output_token_source=TOKEN_SOURCE_USAGE,
            error="boom",
        ),
    ]

    summary = compute_summary(records, accelerator_hourly_cost_usd=3.60)

    assert summary["request_count"] == 2
    assert summary["success_count"] == 1
    assert summary["error_count"] == 1
    assert summary["output_tokens"] == 10
    assert summary["request_throughput_rps"] == 1.0
    assert summary["output_token_throughput_tps"] == 10.0
    assert summary["cost_per_1m_output_tokens_usd"] == pytest.approx(100.0)
    assert summary["p50_ttft_seconds"] == 0.25


def test_streaming_itl_calculation() -> None:
    assert inter_token_latencies([10.0, 10.2, 10.5]) == pytest.approx([0.2, 0.3])
    assert inter_token_latencies([10.0]) == []


def test_missing_cost_inputs_return_none() -> None:
    assert cost_per_1m_output_tokens(100.0, None) is None
    assert cost_per_1m_output_tokens(None, 2.20) is None
    assert cost_per_1m_output_tokens(100.0, 0.0) is None


def test_estimated_token_path_is_marked() -> None:
    records = [
        RequestMetric(
            experiment="test",
            workload="short_chat",
            concurrency=1,
            request_id="a",
            ok=True,
            status_code=200,
            start_time=0.0,
            end_time=1.0,
            first_token_time=None,
            output_tokens=3,
            output_token_source=TOKEN_SOURCE_APPROXIMATE,
        )
    ]

    summary = compute_summary(records)

    assert summary["estimated_output_tokens"] == 3
    assert summary["output_token_count_estimated"] is True
    assert summary["cost_per_1m_output_tokens_usd"] is None
