# Methodology

Reproducible methodology for the measured H100 runs. The committed milestones are the **FP16
baseline** and the **FP8 / AWQ-Marlin INT4** quantization ablations for
`Qwen/Qwen2.5-Coder-7B-Instruct`.

## Hardware & environment

- GPU: single **NVIDIA H100 80GB HBM3**, provider RunPod (assumed `$2.20/hr`).
- Serving: **vLLM 0.21.0** OpenAI-compatible server, served model `Qwen/Qwen2.5-Coder-7B-Instruct`
  (`--dtype float16`). FP8 via `--quantization fp8` (online dynamic quant, reuses the FP16 weights);
  AWQ-INT4 via a pre-quantized `-AWQ` checkpoint with `--quantization awq_marlin`.
- Measured **in-pod** against `http://127.0.0.1:8000/v1` (direct vLLM) to avoid WAN/proxy latency.
- Pre-flight manifests (git commit, config hash, platform; secrets redacted) under
  `results/manifests/`.
- Reproduce from scratch: [runpod_setup.md](runpod_setup.md). Run procedure:
  [h100_baseline_runbook.md](h100_baseline_runbook.md).

## Workloads & sweep

- Latency/throughput — `configs/baseline_{fp16,fp8,awq}.yaml`: workloads `short_chat_256_256` and
  `long_context_4k_512`; concurrency sweep `{1, 8, 32, 64}`; streaming. Held identical across
  precisions so results are directly comparable.
- Quality — `configs/humaneval_{fp16,fp8,awq}.yaml`: all 164 HumanEval problems, one greedy sample
  each (temperature 0, `max_tokens` 512), scored by `human_eval.execution.check_correctness` in its
  sandboxed subprocess (pod-only; never local or CI).

## Metrics & token counting

- TTFT, ITL, end-to-end latency (p50/p95/p99), output-token throughput, request throughput, peak
  GPU memory, cost per 1M output tokens, HumanEval pass@1.
- Output-token counts are **exact** (`usage.completion_tokens`, requested via
  `stream_options.include_usage` on streaming requests). The initial FP16 baseline was re-run after
  enabling this; the earlier whitespace estimate undercounted code tokens by ~1.6x, so the
  throughput/cost reported here supersede that first run.

## Results (committed, validated)

Aggregate across the sweep; per-operating-point detail in each file's `per_workload_metrics`, and
the throughput-vs-latency frontier in `results/pareto.png`. 428 latency requests each (0 errors);
164 HumanEval problems each (0 errors).

| precision | lat p50 (s) | lat p95 (s) | lat p99 (s) | TTFT p50 (s) | ITL p50 (ms) | tok/s | $/1M out | pass@1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FP16 | 1.63 | 2.52 | 2.61 | 0.174 | 6.3 | 516 | 1.18 | 0.878 (144/164) |
| FP8 | 1.11 | 1.98 | 2.02 | 0.136 | 4.6 | 737 | 0.83 | 0.878 (144/164) |
| AWQ-INT4 | 1.16 | 2.71 | 2.73 | 0.175 | 4.7 | 692 | 0.88 | 0.872 (143/164) |

### Findings

- **FP8 dominates on H100:** +43% throughput, −30% $/1M, −32% p50 latency, and lower ITL, at
  identical pass@1. Recommended serving precision for this model/GPU.
- **AWQ-INT4 < FP8 here:** higher throughput than FP16 but worse p95/p99 tail latency and −0.6pp
  quality. INT4's advantage is weight footprint, which does not bind for a 7B model on 80 GB; at
  this scale FP8's native Hopper tensor-core path wins.
- **Peak GPU memory is ~73.6 GB across all three** — vLLM reserves its `gpu_memory_utilization`
  target (~0.90×80) regardless of weight precision, so peak *reserved* memory does not vary.
  Quantization's freed weight memory becomes KV-cache headroom (higher max concurrency / longer
  context), which is the right metric for the memory benefit — not peak reserved.
- pass@1 tracks the model's published ~88.4% HumanEval, a sanity check on the scorer.

## Honesty

No measured-performance claim is published unless it comes from committed result JSON that passes
`scripts/validate_results.py`. Unmeasured fields are `null`. Cross-precision comparisons hold the
workloads, sweep, and token-counting method constant.
