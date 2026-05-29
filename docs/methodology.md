# Methodology

This file will hold the reproducible benchmark methodology for the H100 runs.

Current local foundation scope:

- Validate config parsing, request orchestration, aggregation, gateway proxying, and plotting.
- Use the fake local OpenAI-compatible server only.
- Avoid model downloads, CUDA, vLLM, RunPod, H100 access, and paid APIs.

Future H100 methodology will document:

- Hardware, region, provider, and hourly cost.
- vLLM version, serving backend, backend commit, image digest, model revision, quantization flags, and launch commands.
- Speculative-decoding method, draft model, token settings, acceptance rate, and notes when applicable.
- Workload definitions and prompt/token length distributions.
- Concurrency sweep `{1, 8, 32, 64}`.
- TTFT, ITL, throughput, p50/p95/p99 latency, GPU memory, HumanEval pass@1, and cost per million output tokens.
- Warmup policy, run count, error handling, and result file schema.

No measured-performance claims belong in this repository until the result files and methodology are committed together.
