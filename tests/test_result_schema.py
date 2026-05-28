from __future__ import annotations

from fastcoder.benchmark.config import BenchmarkConfig
from fastcoder.benchmark.gpu import GPUSnapshot
from fastcoder.benchmark.metrics import TOKEN_SOURCE_APPROXIMATE, RequestMetric
from fastcoder.benchmark.results import BenchmarkResult, build_benchmark_result
from fastcoder.benchmark.workloads import get_workloads


def test_result_schema_has_required_top_level_fields() -> None:
    config = BenchmarkConfig(experiment_name="schema_test", workloads=["short_chat"])
    workloads = get_workloads(config.workloads)
    result = build_benchmark_result(
        config=config,
        workloads=workloads,
        records=[
            RequestMetric(
                experiment="schema_test",
                workload="short_chat",
                concurrency=1,
                request_id="req-1",
                ok=True,
                status_code=200,
                start_time=0.0,
                end_time=1.0,
                first_token_time=None,
                output_tokens=2,
                output_token_source=TOKEN_SOURCE_APPROXIMATE,
                response_chars=10,
            )
        ],
        gpu_snapshot=GPUSnapshot(),
    )
    payload = result.model_dump(mode="json")

    assert BenchmarkResult.model_validate(payload)
    assert set(payload) >= {
        "schema_version",
        "run_id",
        "created_at",
        "config_name",
        "model",
        "hardware",
        "software",
        "speculation",
        "workloads",
        "aggregate_metrics",
        "per_workload_metrics",
        "per_request_metrics",
    }
    assert payload["schema_version"] == "0.1"
    assert payload["hardware"]["gpu_name"] is None
    assert payload["software"]["serving_backend"] is None
    assert payload["speculation"]["method"] is None
    assert payload["aggregate_metrics"]["cost_per_1m_output_tokens_usd"] is None
