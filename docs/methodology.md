# Methodology

Reproducible methodology for the measured H100 runs. The first committed milestone is the
**FP16 baseline** for `Qwen/Qwen2.5-Coder-7B-Instruct`.

## Hardware & environment (FP16 baseline)

- GPU: single **NVIDIA H100 80GB HBM3**, provider RunPod (assumed `$2.20/hr`).
- Serving: **vLLM 0.21.0** OpenAI-compatible server, `--dtype float16`, served model
  `Qwen/Qwen2.5-Coder-7B-Instruct`, `--max-model-len 4096`, `--gpu-memory-utilization 0.90`.
- Measured **in-pod** against `http://127.0.0.1:8000/v1` (direct vLLM) to avoid WAN/proxy latency.
- Pre-flight manifests (git commit, config hash, platform; secrets redacted) under
  `results/manifests/`.
- Reproduce from scratch: [runpod_setup.md](runpod_setup.md). Run procedure:
  [h100_baseline_runbook.md](h100_baseline_runbook.md).

## Workloads & sweep

- Latency/throughput — `configs/baseline_fp16.yaml`: workloads `short_chat_256_256` and
  `long_context_4k_512`; concurrency sweep `{1, 8, 32, 64}`; streaming.
- Quality — `configs/humaneval_fp16.yaml`: all **164 HumanEval problems, one greedy sample each**
  (temperature 0, `max_tokens` 512), scored by `human_eval.execution.check_correctness` in its
  sandboxed subprocess (pod-only; never local or CI).

## Metrics

- TTFT, ITL, end-to-end latency (p50/p95/p99), output-token throughput, request throughput, peak
  GPU memory, cost per 1M output tokens, HumanEval pass@1.
- **Token counting:** uses API `usage.completion_tokens` when present, otherwise a whitespace-based
  estimate (flagged in the JSON as `output_token_count_estimated: true`). For this committed baseline
  the streaming responses carried no usage, so **throughput and cost are estimate-based; latency,
  TTFT, ITL, and pass@1 are exact.**

## Committed results (FP16 baseline)

From committed JSON that passes `scripts/validate_results.py`.

**Latency / throughput** — `results/baseline_fp16.json` (428 requests, 0 errors, streaming):

- TTFT p50/p95/p99 = 0.171 / 0.776 / 2.77 s
- end-to-end latency p50/p95/p99 = 1.50 / 2.17 / 2.96 s
- inter-token latency p50/p95/p99 = 6.1 / 8.9 / 11.9 ms
- output throughput 324 tok/s; request throughput 3.79 rps
- peak GPU memory 73.6 / 80 GB; cost $1.89 / 1M output tokens (estimated)
- percentiles are aggregate across the sweep; per-operating-point detail is in
  `per_workload_metrics` and `results/pareto.png`.

**Quality** — `results/humaneval_fp16.json` (164 problems):

- HumanEval pass@1 = 0.878 (144/164), consistent with the model's published ~88.4% (a sanity check
  that the scorer grades correctly).

## Honesty

No measured-performance claim is published unless it comes from committed result JSON that passes
`scripts/validate_results.py`. Unmeasured fields are `null`. Comparative claims (e.g., FP8 vs FP16)
wait until both sides are measured on the declared hardware.
