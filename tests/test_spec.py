from __future__ import annotations

from fastcoder.benchmark.spec import spec_decode_acceptance, spec_decode_lines

_METRICS = """\
# HELP vllm:spec_decode_num_accepted_tokens_total Number of accepted tokens.
# TYPE vllm:spec_decode_num_accepted_tokens_total counter
vllm:spec_decode_num_accepted_tokens_total{model_name="qwen"} 750.0
vllm:spec_decode_num_draft_tokens_total{model_name="qwen"} 1000.0
vllm:num_requests_running{model_name="qwen"} 2.0
"""


def test_spec_decode_acceptance_is_accepted_over_draft() -> None:
    assert spec_decode_acceptance(_METRICS) == 0.75


def test_spec_decode_lines_filters_to_spec_decode_samples() -> None:
    lines = spec_decode_lines(_METRICS)
    assert lines
    assert all("spec_decode" in line for line in lines)
    assert not any(line.startswith("#") for line in lines)
    assert any("accepted" in line for line in lines)


def test_acceptance_is_none_when_metrics_absent() -> None:
    assert spec_decode_acceptance("vllm:num_requests_running 2.0\n") is None


def test_acceptance_is_none_when_draft_is_zero() -> None:
    text = (
        "vllm:spec_decode_num_accepted_tokens_total 0.0\n"
        "vllm:spec_decode_num_draft_tokens_total 0.0\n"
    )
    assert spec_decode_acceptance(text) is None
