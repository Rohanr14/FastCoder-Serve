# FastCoder-Serve

FastCoder-Serve is a production-grade inference serving and measurement project for code LLMs. The target system will benchmark Qwen2.5-Coder-7B-Instruct on a single H100 with vLLM across FP16, FP8, AWQ-Marlin INT4, speculative decoding, and prefix caching, then publish latency, throughput, quality, memory, and cost Pareto curves.

**Current status:** this repo is currently at local foundation stage. H100 benchmark results are not measured yet. No speedups, quality deltas, or cost claims should be inferred from the smoke paths in this repository.

## What Exists Now

- Typed Python package scaffold for benchmark, gateway, and plotting code.
- CPU-safe fake OpenAI-compatible server for local non-streaming and SSE streaming smoke tests.
- Config-driven benchmark harness targeting any OpenAI-compatible `/v1/chat/completions` endpoint, including streamed TTFT and ITL timing.
- FastAPI gateway skeleton with bearer auth, local in-memory rate limiting, true streaming pass-through, structured logs, and Prometheus metrics.
- Stable benchmark result schema version `0.1`.
- H100 FP16 baseline config and runbook scaffolding, not executed yet.
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

The repo is ready for the first measured FP16 baseline run once an H100 pod is explicitly live, but H100 FP16 has not been measured yet.

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

- Tier 1 core: H100 vLLM FP16 baseline, FP8, AWQ-Marlin INT4, benchmark JSON, Pareto plot, FastAPI gateway, Prometheus/Grafana, Docker, CI, and writeup.
- Tier 2 ablations: EAGLE/speculative decoding, prefix caching, acceptance-rate metrics, and side-by-side demo.
- Tier 3 stretch: SGLang head-to-head and a small upstream contribution to vLLM or SGLang.

## Honesty Policy

FastCoder-Serve is measurement-first. Do not claim latency improvements, throughput gains, HumanEval retention, GPU memory savings, or cost reductions until the numbers are measured on the declared hardware and committed with reproducible configuration.

The next milestone is a real FP16 vLLM baseline on a RunPod H100 using `configs/baseline_fp16.yaml`. That milestone is intentionally not part of local CI or smoke testing.
