# FastCoder-Serve

FastCoder-Serve is a production-grade inference serving and measurement project for code LLMs. The target system will benchmark Qwen2.5-Coder-7B-Instruct on a single H100 with vLLM across FP16, FP8, AWQ-Marlin INT4, and later conditional speculative-decoding or prefix-caching ablations, then publish latency, throughput, quality, memory, and cost Pareto curves.

**Current status:** FP16, FP8, and AWQ-Marlin INT4 are all measured and committed on a single H100 (vLLM 0.21.0, Qwen2.5-Coder-7B-Instruct) with exact token counts. See [Results (measured)](#results-measured) — headline: **FP8 gives ~+43% throughput and −30% cost at identical HumanEval pass@1**. Smoke paths in this repo are harness tests only; do not infer serving performance from them.

## Results (measured)

Validated results for `Qwen/Qwen2.5-Coder-7B-Instruct` on a single RunPod H100 80GB (vLLM 0.21.0),
measured in-pod against direct vLLM across three precisions. All numbers come from committed JSON
that passes `scripts/validate_results.py`; output-token counts are **exact**
(`usage.completion_tokens`).

**Latency / throughput / cost** — `results/baseline_{fp16,fp8,awq}.json`; 428 requests each, 0 errors, streaming, concurrency sweep `{1, 8, 32, 64}` (aggregate across the sweep):

| precision | lat p50 (s) | lat p95 (s) | lat p99 (s) | TTFT p50 (s) | ITL p50 (ms) | throughput (tok/s) | $/1M out |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP16 | 1.63 | 2.52 | 2.61 | 0.174 | 6.3 | 516 | $1.18 |
| **FP8** | **1.11** | **1.98** | **2.02** | **0.136** | **4.6** | **737** | **$0.83** |
| AWQ-INT4 | 1.16 | 2.71 | 2.73 | 0.175 | 4.7 | 692 | $0.88 |

**Quality** — `results/humaneval_{fp16,fp8,awq}.json`; HumanEval pass@1, 164 problems, single greedy sample each:

| precision | pass@1 | passed |
| --- | --- | --- |
| FP16 | 0.878 | 144/164 |
| FP8 | 0.878 | 144/164 |
| AWQ-INT4 | 0.872 | 143/164 |

**Takeaways**
- **FP8 is a near-free win on H100:** +43% throughput, −30% $/1M, −32% p50 latency vs FP16, at **identical** pass@1 (144/164). The recommended serving precision here.
- **AWQ-INT4 trails FP8** on this hardware: +34% throughput but worse p95/p99 *tail* latency than FP16 and −0.6pp quality. INT4's edge is weight footprint, which doesn't bind for a 7B on 80 GB.
- **Prefix caching** (shared-context workload, FP8) cut tail TTFT 85% and gave **+163% throughput at concurrency 64** — see the writeup's production recommendations.
- **Peak GPU memory is ~73.6 GB for all three** — vLLM reserves its `gpu_memory_utilization` target regardless of weight size, so quantization's freed memory becomes KV-cache headroom (concurrency/context), not lower peak. See [methodology](docs/methodology.md).
- pass@1 tracks the model's published ~88.4% HumanEval (sanity check on the scorer).

Per-operating-point detail is in each file's `per_workload_metrics`; the throughput-vs-latency frontier is plotted in [`results/pareto.png`](results/pareto.png). Full analysis — capacity under SLO, FP8-vs-INT4, and deployment guidance — is in [docs/writeup.md](docs/writeup.md). Reproduce from scratch via [docs/runpod_setup.md](docs/runpod_setup.md).

## Live demo (observability)

One command brings up a **CPU-only** demo of the serving + observability stack — the FastCoder
gateway in front of a fake OpenAI server, Prometheus scraping it, and a provisioned Grafana
dashboard, driven by a continuous load generator:

```bash
docker compose -f docker/docker-compose.yml up --build
# Grafana (anonymous): http://localhost:3000  ·  Prometheus: http://localhost:9090
```

Live latency / throughput / error dashboards under load, no GPU. It demonstrates the *system*, not
H100 performance (the measured numbers live in [results/](results) and [docs/writeup.md](docs/writeup.md)).
See [docs/observability.md](docs/observability.md).

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

- Tier 1 core: H100 vLLM FP16 baseline (done), FP8 (done), AWQ-Marlin INT4 (done), benchmark JSON, Pareto plot, FastAPI gateway, Prometheus/Grafana, Docker, CI, and writeup.
- Tier 2 ablations: draft-model speculative decoding, conditional EAGLE 3.1 if compatible support exists, prefix caching, acceptance-rate metrics, and side-by-side demo.
- Tier 3 stretch: SGLang head-to-head and a small upstream contribution to vLLM or SGLang.

## Honesty Policy

FastCoder-Serve is measurement-first. Do not claim latency improvements, throughput gains, HumanEval retention, GPU memory savings, or cost reductions until the numbers are measured on the declared hardware and committed with reproducible configuration.

EAGLE 3.1 is conditional/stretch for this repository. It may require vLLM nightly, current main, or an upcoming v0.22.0 release line plus compatible Qwen2.5-Coder support. If that support is unavailable, draft-model speculative decoding or a documented incompatibility is the correct outcome.

FP16, FP8, and AWQ-Marlin INT4 are measured and committed under `results/` with exact token counts. Next is a writeup of the precision frontier and optional Tier-2 ablations (speculative decoding, prefix caching). GPU runs are intentionally not part of local CI or smoke testing.
