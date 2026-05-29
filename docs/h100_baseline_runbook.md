# H100 FP16 Baseline Runbook

This runbook is for the first paid FP16 baseline on a single H100. It is command-driven, but it is still scaffold until the pod is intentionally live.

Do not claim speedups, throughput, HumanEval retention, GPU memory savings, or cost improvements until measured result JSON files and methodology notes are committed.

The first measured run is **FP16 Qwen2.5-Coder-7B-Instruct**. FP8, AWQ-Marlin, prefix caching, and speculative-decoding ablations come later.

## RunPod Pod Selection

- Select a single H100 80GB pod.
- Attach sufficient disk volume for Qwen2.5-Coder-7B-Instruct weights, Python packages, logs, and result files.
- Expose only the ports required for the run:
  - vLLM OpenAI-compatible API, default `8000`
  - optional FastCoder gateway, default `8080`
  - optional SSH or notebook access as needed
- Confirm the hourly price before starting.
- Record pod start time and intended shutdown time.
- Shut down the pod immediately after results are copied. Never leave RunPod running overnight.

## Environment Setup

```bash
git clone <repo-url> FastCoder-Serve
cd FastCoder-Serve
python3.11 -m venv .venv
source .venv/bin/activate
make install
make validate-baseline-config
make baseline-dry-run
```

Set environment variables if using the gateway:

```bash
export FASTCODER_API_KEY=<token>
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
```

The run manifest redacts `FASTCODER_API_KEY`.

## vLLM Install Notes

Install vLLM in the H100 environment only. Do not install or run vLLM as part of local CI.

Record these details in the run notes:

- vLLM version from `vllm --version`
- Python version
- CUDA image or base container
- GPU details from `nvidia-smi`
- model revision if pinned
- exact vLLM launch command

## FP16 vLLM Command Template

```bash
python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --dtype float16 \
  --served-model-name Qwen/Qwen2.5-Coder-7B-Instruct
```

This command may trigger the model download on the H100 pod. It is not run locally and has not produced benchmark numbers yet.

## Direct vLLM Path

Prefer running the benchmark from inside the RunPod pod against `http://127.0.0.1:8000/v1`. That avoids measuring local-WAN, SSH tunnel, provider proxy, or browser-path latency. If any result is measured remotely through an exposed port or proxy, disclose that in the result notes and methodology.

Use this path when checking vLLM directly:

```bash
python scripts/check_endpoint.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --timeout-seconds 120 \
  --stream
```

Then run the dry-run scaffold:

```bash
python scripts/run_h100_baseline.py \
  --config configs/baseline_fp16.yaml \
  --base-url http://127.0.0.1:8000/v1
```

## Gateway-Mediated Path

Start the gateway in front of vLLM:

```bash
FASTCODER_API_KEY=$FASTCODER_API_KEY \
VLLM_BASE_URL=http://127.0.0.1:8000/v1 \
python scripts/run_gateway.py --host 0.0.0.0 --port 8080
```

Check the gateway endpoint:

```bash
python scripts/check_endpoint.py \
  --base-url http://127.0.0.1:8080/v1 \
  --api-key "$FASTCODER_API_KEY" \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --timeout-seconds 120 \
  --stream
```

Use the gateway URL in the baseline runner:

```bash
python scripts/run_h100_baseline.py \
  --config configs/baseline_fp16.yaml \
  --base-url http://127.0.0.1:8080/v1 \
  --api-key "$FASTCODER_API_KEY"
```

Gateway-mediated numbers should be labeled separately from direct vLLM numbers because the gateway can add latency and backpressure.

## Manifest

Create a reproducibility manifest before the paid run:

```bash
make manifest-baseline
```

Manifest files are written under:

```text
results/manifests/baseline_fp16_<timestamp>.json
```

They include git state, config hash, Python/platform information, selected environment variables, and the intended output result path. Secrets are redacted.

## Baseline Dry Run

Run this before any paid benchmark:

```bash
make baseline-dry-run
```

It validates `configs/baseline_fp16.yaml` and prints the planned endpoint check, manifest, benchmark, and result-validation commands. It does not run the benchmark.

## Real Baseline Command

Only run this after the H100 pod is live, vLLM is serving, endpoint checks pass, and the spend window is approved:

```bash
python scripts/run_h100_baseline.py \
  --config configs/baseline_fp16.yaml \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key "$FASTCODER_API_KEY" \
  --confirm-paid-run
```

The `--confirm-paid-run` flag is required. Without it, the script exits in dry-run mode.

## Result Validation

After a local smoke run:

```bash
make validate-smoke-results
make validate-streaming-results
```

After the real H100 baseline:

```bash
python scripts/validate_results.py results/baseline_fp16.json
```

Expected baseline artifacts:

- `results/baseline_fp16.json`
- `results/manifests/baseline_fp16_<timestamp>.json`
- command logs or notes captured separately for the methodology writeup

Before copying numbers into any public document, confirm the result JSON includes:

- `software.vllm_version`
- `software.serving_backend`
- `software.backend_commit` when available
- `speculation.method: null`
- `speculation.draft_model: null`

## Troubleshooting

- Model download slow: keep the pod alive only for the planned window, verify disk volume, and avoid restarting the download unnecessarily.
- CUDA/vLLM install mismatch: record the exact error and image details; fix the environment before benchmarking.
- OOM: stop the benchmark, capture the config, and reduce scope only after documenting the failure.
- Endpoint reachable but chat completions fail: run `scripts/check_endpoint.py` and inspect `/v1/models`, model name, auth header, and vLLM logs.
- Streaming TTFT null: confirm `stream: true`, verify SSE chunks with `scripts/check_endpoint.py --stream`, and avoid buffered proxies.
- p99 latency looks wrong: validate result JSON, inspect per-request timings, rerun a small sanity sweep before spending on the full run.

## Cost Controls

- Start with `make validate-baseline-config` and endpoint checks.
- Run `make baseline-dry-run` before the confirmed command.
- Run a small sanity benchmark first if the confirmed baseline config is adjusted.
- Copy results and manifests off the pod immediately.
- Shut down the RunPod pod after results are secured.
- Never leave RunPod running overnight.
