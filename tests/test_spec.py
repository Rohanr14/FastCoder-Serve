from __future__ import annotations

from fastcoder.benchmark.spec import spec_decode_acceptance, spec_decode_lines

# Mirrors real vLLM v1 output: *_total counters PLUS *_created (epoch timestamps) and *_per_pos.
_REAL_SHAPE = """\
vllm:spec_decode_num_drafts_total{engine="0"} 200.0
vllm:spec_decode_num_drafts_created{engine="0"} 1.7802746209391222e+09
vllm:spec_decode_num_draft_tokens_total{engine="0"} 1000.0
vllm:spec_decode_num_draft_tokens_created{engine="0"} 1.7802746209391582e+09
vllm:spec_decode_num_accepted_tokens_total{engine="0"} 750.0
vllm:spec_decode_num_accepted_tokens_created{engine="0"} 1.7802746209391844e+09
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="0"} 300.0
vllm:spec_decode_num_accepted_tokens_per_pos_created{engine="0",position="0"} 1.7802746209392188e+09
"""

# Exactly the shape the user captured: every *_total is zero, *_created are epoch timestamps.
_ALL_ZERO = """\
vllm:spec_decode_num_draft_tokens_total{engine="0"} 0.0
vllm:spec_decode_num_draft_tokens_created{engine="0"} 1.7802746209391582e+09
vllm:spec_decode_num_accepted_tokens_total{engine="0"} 0.0
vllm:spec_decode_num_accepted_tokens_created{engine="0"} 1.7802746209391844e+09
"""


def test_acceptance_uses_total_counters_only() -> None:
    # 750 accepted / 1000 draft = 0.75 — must NOT be polluted by _created epochs (~1.78e9) or the
    # _per_pos breakdown (300). Regression guard for the bug that produced ~6.0.
    assert spec_decode_acceptance(_REAL_SHAPE) == 0.75


def test_acceptance_none_when_counters_zero_despite_created_timestamps() -> None:
    assert spec_decode_acceptance(_ALL_ZERO) is None


def test_acceptance_none_when_absent() -> None:
    assert spec_decode_acceptance("vllm:num_requests_running 2.0\n") is None


def test_spec_decode_lines_filters_to_spec_decode_samples() -> None:
    lines = spec_decode_lines(_REAL_SHAPE)
    assert lines
    assert all("spec_decode" in line for line in lines)
    assert not any(line.startswith("#") for line in lines)
