# FastCoder-Serve

FastCoder-Serve is a production-grade inference serving and measurement project for code LLMs. The target system will benchmark Qwen2.5-Coder-7B-Instruct on a single H100 with vLLM across FP16, FP8, AWQ-Marlin INT4, and later conditional speculative-decoding or prefix-caching ablations, then publish latency, throughput, quality, memory, and cost Pareto curves.

**Current status:** the first measured **FP16 baseline is committed** — single H100, vLLM 0.21.0, Qwen2.5-Coder-7B-Instruct. See [Results (measured)](#results-measured). FP8 and AWQ-Marlin INT4 ablations are next. Smoke paths in this repo are harness tests only; do not infer serving performance from them.

## Results (measured)

First validated FP16 baseline for `Qwen/Qwen2.5-Coder-7B-Instruct` on a single RunPod H100 80GB
(vLLM 0.21.0, `--dtype float16`), measured in-pod against direct vLLM. All numbers come from
committed JSON that passes `scripts/validate_results.py`.

**Latency / throughput** — [`results/baseline_fp16.json`](results/baseline_fp16.json); 428 requests, 0 errors, streaming, concurrency sweep `{1, 8, 32, 64}`:

| metric | p50 | p95 | p99 |
| --- | --- | --- | --- |
| TTFT (s) | 0.171 | 0.776 | 2.77 |
| end-to-end latency (s) | 1.50 | 2.17 | 2.96 |
| inter-token latency (ms) | 6.1 | 8.9 | 11.9 |

- output throughput **324 tok/s**, request throughput 3.79 rps
- peak GPU memory **73.6 / 80 GB**
- cost **$1.89 / 1M output tokens** *(token counts estimated — see caveat)*
- percentiles are aggregate across the sweep; per-operating-point detail is in `per_workload_metrics` and [`results/pareto.png`](results/pareto.png)

**Quality** — [`results/humaneval_fp16.json`](results/humaneval_fp16.json); 164 problems, single greedy sample each:

- **HumanEval pass@1 = 0.878 (144/164)** — consistent with the model's published ~88.4% (a sanity check that the sandboxed scorer grades correctly).

> **Caveat:** throughput and $/1M output tokens rest on a whitespace-based token estimate because the streaming responses returned no `usage.completion_tokens`. Latency, TTFT, ITL, and pass@1 are exact. Reproduce from scratch via [docs/runpod_setup.md](docs/runpod_setup.md).

## What Exists Now

- Typed Python package scaffold for benchmark, gateway, and plotting code.
- CPU-safe fake OpenAI-compatible server for local non-streaming and SSE streaming smoke tests.
- Config-driven benchmark harness targeting any OpenAI-compatible `/v1/chat/completions` endpoint, including streamed TTFT and ITL timing.
- FastAPI gateway skeleton with bearer auth, local in-memory rate limiting, true streaming pass-through, structured logs, and Prometheus metrics.
- Stable benchmark result schema version `0.1`.
- Result metadata for serving backend, backend commit, vLLM version, and speculative-decoding settings.
- H100 FP16 baseline + HumanEval eval, executed on an H100 and committed under `results/`.
- Placeholder Docker, Prometheus, Grafana, CI, and methodology files.

## Architecture

```text
benchmark runner -> OpenAI-compatible endpoint
                         ^
                         |
                 FastAPI gateway
                         ^
                         |
              vLLM or fake local server

Prometheus scrapes gateway /metrics, and Grafana loads checked-in dashboard JSON.
```

For local development, the fake server replaces vLLM. It downloads no models and requires no GPU.

## Local Quickstart

1. Create and activate a virtual environment:

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   make install
   ```

3. Run the fake OpenAI-compatible server:

   ```bash
   make fake-server
   ```

4. In another shell, run the gateway against the fake server:

   ```bash
   FASTCODER_AUTH_BYPASS=true VLLM_BASE_URL=http://localhost:9000/v1 make gateway
   ```

5. Run the smoke benchmark:

   ```bash
   make bench-smoke
   ```

6. Run the streaming smoke benchmark:

   ```bash
   make bench-smoke-streaming
   ```

7. Generate the placeholder Pareto plot:

   ```bash
   make plot-smoke
   ```

8. Run local CI checks:

   ```bash
   make ci
   ```

Useful local commands:

```bash
make ci
make fake-server
make gateway
make bench-smoke
make bench-smoke-streaming
make plot-smoke
```

## Pre-Flight Tools

The FP16 baseline has been measured and committed (see [Results (measured)](#results-measured)). These pre-flight tools remain for re-runs and the upcoming FP8/AWQ ablations:

```bash
make validate-baseline-config
make baseline-dry-run
make manifest-baseline
make validate-smoke-results
make validate-streaming-results
```

Additional validation commands:

```bash
make validate-smoke-config
make validate-streaming-config
make check-fake-endpoint
```

All future performance numbers must come from committed result JSON files that pass `scripts/validate_results.py`. Do not copy numbers from logs, screenshots, papers, or expectations into the README.

## Result Schema

Benchmark outputs under `results/*.json` use schema version `0.1` with these top-level fields:

- `run_id`, `created_at`, and `config_name`
- `model` and `hardware` metadata
- `software` and `speculation` metadata
- `workloads`
- `aggregate_metrics`
- `per_workload_metrics`
- `per_request_metrics`

Unavailable local-only metrics such as GPU memory and HumanEval pass@1 are `null`. Estimated token counts are explicitly marked in the metrics.

## Methodology Stub

The final benchmark will report p50/p95/p99 TTFT, inter-token latency, throughput, GPU memory, HumanEval pass@1, and cost per million output tokens across concurrency levels `{1, 8, 32, 64}`. Local smoke tests only validate harness mechanics and do not represent real serving performance.

See [docs/methodology.md](docs/methodology.md) for the evolving methodology notes.
See [docs/h100_baseline_runbook.md](docs/h100_baseline_runbook.md) for the future FP16 H100 baseline procedure.

## Roadmap

- Tier 1 core: H100 vLLM FP16 baseline (done), FP8, AWQ-Marlin INT4, benchmark JSON, Pareto plot, FastAPI gateway, Prometheus/Grafana, Docker, CI, and writeup.
- Tier 2 ablations: draft-model speculative decoding, conditional EAGLE 3.1 if compatible support exists, prefix caching, acceptance-rate metrics, and side-by-side demo.
- Tier 3 stretch: SGLang head-to-head and a small upstream contribution to vLLM or SGLang.

## Honesty Policy

FastCoder-Serve is measurement-first. Do not claim latency improvements, throughput gains, HumanEval retention, GPU memory savings, or cost reductions until the numbers are measured on the declared hardware and committed with reproducible configuration.

EAGLE 3.1 is conditional/stretch for this repository. It may require vLLM nightly, current main, or an upcoming v0.22.0 release line plus compatible Qwen2.5-Coder support. If that support is unavailable, draft-model speculative decoding or a documented incompatibility is the correct outcome.

The FP16 baseline milestone is complete and committed under `results/`. The next milestone is the FP8 ablation, then AWQ-Marlin INT4. GPU runs are intentionally not part of local CI or smoke testing.
