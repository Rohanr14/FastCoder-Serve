# Contributing

FastCoder-Serve is an LLM inference serving and measurement project. The core target is Qwen2.5-Coder-7B-Instruct on a single H100, measured through reproducible benchmark configs and checked-in result artifacts.

## Project Scope

- Benchmark OpenAI-compatible inference endpoints.
- Measure latency, TTFT, inter-token latency, throughput, GPU memory, quality, and cost.
- Keep experiments config-driven and reproducible.
- Maintain a small FastAPI gateway with observability suitable for serving experiments.

## Anti-Scope

- No fine-tuning or training.
- No RAG, vector databases, LangChain-style orchestration, or unrelated application features.
- No closed-model quality comparisons.
- No consumer-GPU benchmark claims for headline results.
- No user accounts, SSO, bots, or frontend-heavy product work.

## Local Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
make install
```

## Local Checks

```bash
make ci
make bench-smoke
make bench-smoke-streaming
make validate-smoke-results
make validate-streaming-results
```

Local checks must not require CUDA, H100 access, RunPod, vLLM, paid APIs, or large model downloads.

## Benchmark Honesty

- Do not publish speedups, cost reductions, quality deltas, or GPU-memory numbers until they are measured and backed by committed result JSON.
- Record exact hardware, backend version, backend commit when available, model revision, config file, and run manifest.
- Use `null` for unavailable metrics rather than filling placeholders.
- Keep local smoke results clearly labeled as plumbing checks, not performance measurements.

## Secrets and Large Artifacts

- Do not commit API keys, provider tokens, SSH keys, `.env` files, or private endpoint URLs.
- Do not commit model weights, model caches, large raw logs, or large benchmark dumps.
- Keep generated result artifacts small and schema-valid.
