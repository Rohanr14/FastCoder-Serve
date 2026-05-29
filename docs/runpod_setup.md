# RunPod H100 Setup & Reproduction Guide

End-to-end guide to stand up a **fresh** single-H100 RunPod pod, connect to it from the
Windsurf IDE terminal over SSH, serve `Qwen/Qwen2.5-Coder-7B-Instruct` with vLLM, and run the
FastCoder-Serve benchmarks (latency sweep + HumanEval pass@1).

It is written for the **terminate-and-recreate** workflow: stopping a pod releases the H100, and
the GPU is often gone when you come back, so the practical move is to terminate the old pod and
spin up a new one with a fresh GPU. The cost of that is re-downloading ~15 GB of model weights
every time — section 1 shows how a Network Volume removes that cost so a fresh pod is cheap to
recreate.

This guide owns **pod creation, SSH, and the run sequence**. For the *why* behind each benchmark
flag, troubleshooting, and the methodology/measurement rules, it hands off to
[h100_baseline_runbook.md](h100_baseline_runbook.md) rather than duplicating it.

> **Safety / honesty rules (unchanged):**
> - HumanEval scoring executes model-generated code. Run it **only** on the disposable pod, never
>   locally or in CI.
> - Do not publish any speedup/throughput/latency/pass@1/memory/cost number unless it comes from a
>   committed result JSON that passes `scripts/validate_results.py`. Unmeasured fields stay `null`.
> - The paid runs are gated behind `--confirm-paid-run`. Without that flag every script is a dry run.

---

## At a glance (the happy path)

1. One-time: add your SSH **public** key to RunPod, and (recommended) create a Network Volume.
2. Create a fresh **1× H100 80GB** pod from a CUDA/PyTorch template, attaching the volume.
3. SSH in from the Windsurf integrated terminal.
4. Bootstrap: clone repo → venv → `make install` → install vLLM → install `.[eval]`.
5. Launch vLLM under `tmux` (FP16).
6. `check_endpoint.py` to confirm it serves.
7. Latency baseline: dry-run, then `--confirm-paid-run`.
8. HumanEval pass@1: dry-run, then `--confirm-paid-run`.
9. Validate result JSONs, copy them off the pod, **terminate the pod**.

A copy-paste cheat sheet for steps 4–9 is at the bottom.

---

## 1. One-time account setup

### 1a. SSH public key

RunPod injects an **account-level** public key into every new pod's `authorized_keys`. Do this
once and every future fresh pod is reachable with the same key.

On your laptop, create a key if you don't have one (Ed25519 recommended):

```bash
ls ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519 -C "runpod" -f ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

Paste the **public** key (the `.pub` contents — never the private key) into
RunPod → **Settings → SSH Public Keys**.

### 1b. Network Volume (recommended — this is what makes recreate cheap)

A Network Volume persists across pod terminations. Cache the Hugging Face weights on it once and
every fresh pod skips the ~15 GB download.

- RunPod → **Storage → Network Volumes → New**.
- Size it for the model cache plus headroom (the 7B FP16 weights are ~15 GB; ~60 GB leaves room
  for future variants and logs).
- **Pick the region where H100s are actually available to you** — a Network Volume is region-locked,
  and you can only attach it to pods in the same region. If H100 supply in that region dries up,
  you fall back to re-downloading weights on a pod elsewhere.
- Confirm the per-GB monthly price in the RunPod console before creating it.

> Persist only the **model cache** (and optionally the repo) on the volume. Do **not** try to reuse
> a `.venv` across pods — vLLM wheels are tied to the image's CUDA/driver, so rebuild the Python
> environment on each fresh pod. The slow part you're caching is the weight download, not the venv.

---

## 2. Create a fresh H100 pod

RunPod → **Pods → Deploy**:

- **GPU:** 1× H100 80GB (SXM or PCIe). This matches `configs/baseline_fp16.yaml`
  (`NVIDIA H100 80GB HBM3`, assumed `$2.20/hr` — confirm the live price before deploying).
- **Template / image:** a CUDA 12.x PyTorch base (e.g. a `runpod/pytorch` CUDA 12.x image). Any
  image with a recent CUDA runtime and Python 3.11 works; vLLM is installed in step 4, not baked in.
- **Container / volume disk:** enough for weights + Python packages + logs. If you attached a
  Network Volume it mounts at **`/workspace`**; otherwise give the container disk ~60 GB.
- **Network Volume:** attach the one from step 1b (mounts at `/workspace`).
- **Exposed ports:**
  - **SSH** — required (RunPod exposes this by default for SSH-enabled templates).
  - **TCP 8000** — only if you want to reach vLLM from your laptop/browser. For the recommended
    in-pod benchmark you do **not** need it exposed (the benchmark hits `127.0.0.1:8000`).
  - **TCP 8080** — only if you'll run the optional FastCoder gateway and reach it externally.
- Confirm the hourly price, deploy, and **record the start time** so you can cap spend.

---

## 3. Connect via SSH from the Windsurf IDE terminal

Once the pod is **Running**, open its **Connect** panel. Use the **"SSH over exposed TCP"** command
(direct, supports `scp`) — it looks like:

```bash
ssh root@<POD_PUBLIC_IP> -p <POD_SSH_PORT> -i ~/.ssh/id_ed25519
```

In Windsurf, open the integrated terminal (**Terminal → New Terminal**, or `` Ctrl+` ``) and paste
that command. That terminal is now a shell *on the pod*.

**Optional — an alias so you don't hunt for IP/port each time.** Add to `~/.ssh/config` on your
laptop (update IP/port after each recreate):

```sshconfig
Host runpod-h100
    HostName <POD_PUBLIC_IP>
    Port <POD_SSH_PORT>
    User root
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 30
```

Then just `ssh runpod-h100`.

**Optional — edit pod files in the IDE.** If you want Windsurf's editor (not just the terminal) on
the pod, use its Remote-SSH support to open the remote folder (e.g. `/workspace/FastCoder-Serve`)
using the same `runpod-h100` host. The terminal path above is enough for everything in this guide.

---

## 4. Bootstrap the project on the pod

Work under `/workspace` so the repo lives on the Network Volume (survives recreate). All commands
below run **on the pod**.

```bash
cd /workspace

# Get the code (clone fresh, or see the rsync note below to push local uncommitted work).
git clone https://github.com/Rohanr14/FastCoder-Serve.git
cd FastCoder-Serve

# Cache HF weights on the Network Volume so future fresh pods skip the download.
export HF_HOME=/workspace/hf
# Optional: avoids HF rate limits. Qwen2.5-Coder-7B-Instruct is public, so this is not required.
# export HF_TOKEN=<your_hf_token>

# Python env + project (dev extra: ruff/mypy/pytest).
python3.11 -m venv .venv
source .venv/bin/activate
make install

# vLLM is pod-only and intentionally NOT a project dependency. Pin for reproducibility;
# the runner captures whatever /version reports into the result JSON regardless.
pip install vllm==0.21.0

# HumanEval prompts + scorer (pod-only; pulls human-eval from GitHub).
pip install -e ".[eval]"
```

### 4a. Enable HumanEval code execution (one-time, easy to miss)

`human-eval` ships with the actual `exec(...)` call **commented out** as a safety measure — until
you uncomment it, `check_correctness` runs nothing and **every problem silently fails (pass@1 = 0)**.
On the disposable pod, enable it:

```bash
python - <<'PY'
import pathlib, human_eval.execution as e
p = pathlib.Path(e.__file__)
s = p.read_text()
needle = "#         exec(check_program, exec_globals)"
repl = "        exec(check_program, exec_globals)"
if needle in s:
    p.write_text(s.replace(needle, repl, 1))
    print("enabled exec in", p)
else:
    print("already enabled (or layout changed — open and check manually):", p)
PY
```

If it prints "already enabled (or layout changed)", open `human_eval/execution.py`, find the
`exec(check_program, exec_globals)` line near the safety warning, and make sure it is uncommented.
This only executes untrusted code inside human_eval's guarded subprocess on a throwaway pod — which
is exactly why scoring is pod-only.

### 4b. (Alternative to clone) push local uncommitted work

If you have local changes not yet on GitHub, rsync from your **laptop** instead of cloning:

```bash
rsync -av --exclude .venv --exclude .git --exclude results/manifests \
  ./ "root@<POD_PUBLIC_IP>:/workspace/FastCoder-Serve/" -e "ssh -p <POD_SSH_PORT> -i ~/.ssh/id_ed25519"
```

---

## 5. Start vLLM (FP16) under tmux

Run the server in `tmux` so it survives a dropped SSH connection.

```bash
tmux new -s vllm
# inside tmux:
source .venv/bin/activate
export HF_HOME=/workspace/hf   # same cache dir as bootstrap

python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --dtype float16 \
  --served-model-name Qwen/Qwen2.5-Coder-7B-Instruct
```

First launch downloads the weights into `HF_HOME` (slow once; fast on every later fresh pod that
reuses the volume). Wait for vLLM to log that startup is complete and it's serving on `:8000`.
Detach with `Ctrl-b d`; reattach later with `tmux attach -t vllm`.

> FP16 is the first baseline. FP8 / AWQ-Marlin / prefix-caching / speculative-decoding launch
> variants come later and will be documented alongside their configs.

---

## 6. Verify the endpoint

In a second shell on the pod (new Windsurf terminal `ssh runpod-h100`, then
`cd /workspace/FastCoder-Serve && source .venv/bin/activate`):

```bash
python scripts/check_endpoint.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --timeout-seconds 120 \
  --stream
```

Expect a successful chat completion and streamed chunks. If this fails, fix it before spending on a
run — see the troubleshooting section of [h100_baseline_runbook.md](h100_baseline_runbook.md).

---

## 7. Run the benchmarks

Run from **inside the pod** against `127.0.0.1:8000` so you measure the model, not WAN/proxy latency.
Every runner is a dry run until you add `--confirm-paid-run`.

### 7a. Latency / throughput sweep (FP16 baseline)

```bash
# Dry run first — validates config, writes nothing, runs nothing:
python scripts/run_h100_baseline.py \
  --config configs/baseline_fp16.yaml \
  --base-url http://127.0.0.1:8000/v1

# Confirmed paid run (only after endpoint checks pass and the spend window is approved):
python scripts/run_h100_baseline.py \
  --config configs/baseline_fp16.yaml \
  --base-url http://127.0.0.1:8000/v1 \
  --confirm-paid-run
```

Writes `results/baseline_fp16.json` and a manifest under `results/manifests/`. See the runbook for
the gateway-mediated variant and the full flag rationale.

### 7b. HumanEval pass@1 (single greedy sample per problem)

This is the dedicated single-pass eval (separate from the latency sweep). It runs all 164 problems
once, captures completions, and scores pass@1 in human_eval's sandbox. Requires step 4a.

```bash
# Dry run — validates config and prints the plan; imports no human_eval, runs nothing:
python scripts/run_humaneval_eval.py \
  --config configs/humaneval_fp16.yaml \
  --base-url http://127.0.0.1:8000/v1

# Confirmed paid run:
python scripts/run_humaneval_eval.py \
  --config configs/humaneval_fp16.yaml \
  --base-url http://127.0.0.1:8000/v1 \
  --confirm-paid-run
```

If above causes error due to no detected dataset, download manually and run from that:
```bash
# Download HumanEval dataset manually:
wget https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz -O HumanEval.jsonl.gz

FASTCODER_HUMANEVAL_PATH=$(pwd)/HumanEval.jsonl.gz python scripts/run_humaneval_eval.py \
  --config configs/humaneval_fp16.yaml \
  --base-url http://127.0.0.1:8000/v1 \
  --confirm-paid-run
```

Writes `results/humaneval_fp16.json` with `aggregate_metrics.humaneval_pass_at_1` populated and
prints the pass@1 and the captured live vLLM version.

Useful flags:
- `--humaneval-path <file.jsonl(.gz)>` — score against a local problem set instead of the installed
  package (also settable via `FASTCODER_HUMANEVAL_PATH`).
- `--timeout-seconds <float>` — per-problem execution timeout in the sandbox (default 10).
- `--api-key <token>` — bearer token if you're going through the gateway instead of direct vLLM.

---

## 8. Validate and copy results off the pod

Validate **on the pod** before you tear it down:

```bash
python scripts/validate_results.py results/baseline_fp16.json
python scripts/validate_results.py results/humaneval_fp16.json
```

Then get the artifacts to your laptop (run **on the laptop**):

```bash
scp -P <POD_SSH_PORT> -i ~/.ssh/id_ed25519 \
  "root@<POD_PUBLIC_IP>:/workspace/FastCoder-Serve/results/*.json" ./results/
scp -P <POD_SSH_PORT> -i ~/.ssh/id_ed25519 -r \
  "root@<POD_PUBLIC_IP>:/workspace/FastCoder-Serve/results/manifests" ./results/
```

(Or commit + push from the pod if you prefer git transport.) Re-validate on the laptop, then commit
the measured JSON with methodology notes.

---

## 9. Safe shutdown

- Confirm both result JSONs are copied off **and** validated locally.
- Stop vLLM (`tmux attach -t vllm`, `Ctrl-c`) — optional, since you're terminating anyway.
- **Terminate** the pod (your workflow). The Network Volume keeps the weight cache for next time.
- Never leave an H100 idle — record the actual stop time against the start time from step 2.

Full cost-control checklist: see [h100_baseline_runbook.md](h100_baseline_runbook.md#cost-controls).

---

## Cheat sheet (fresh pod, steps 4–9)

After SSHing into a fresh pod with the Network Volume attached:

```bash
# --- bootstrap (on pod) ---
cd /workspace
git clone https://github.com/Rohanr14/FastCoder-Serve.git    # or rsync from laptop
cd FastCoder-Serve
export HF_HOME=/workspace/hf
python3.11 -m venv .venv && source .venv/bin/activate
make install
pip install vllm==0.21.0
pip install -e ".[eval]"
# enable human_eval exec (see step 4a) — REQUIRED for pass@1

# --- serve (in tmux) ---
tmux new -s vllm
source .venv/bin/activate && export HF_HOME=/workspace/hf
python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8000 \
  --model Qwen/Qwen2.5-Coder-7B-Instruct --dtype float16 \
  --served-model-name Qwen/Qwen2.5-Coder-7B-Instruct
# Ctrl-b d to detach

# --- verify + run (second shell) ---
python scripts/check_endpoint.py --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-Coder-7B-Instruct --timeout-seconds 120 --stream

python scripts/run_h100_baseline.py --config configs/baseline_fp16.yaml \
  --base-url http://127.0.0.1:8000/v1                      # dry run
python scripts/run_h100_baseline.py --config configs/baseline_fp16.yaml \
  --base-url http://127.0.0.1:8000/v1 --confirm-paid-run   # paid

python scripts/run_humaneval_eval.py --config configs/humaneval_fp16.yaml \
  --base-url http://127.0.0.1:8000/v1                      # dry run
python scripts/run_humaneval_eval.py --config configs/humaneval_fp16.yaml \
  --base-url http://127.0.0.1:8000/v1 --confirm-paid-run   # paid

# --- validate + ship ---
python scripts/validate_results.py results/baseline_fp16.json
python scripts/validate_results.py results/humaneval_fp16.json
# scp results off (see step 8), then TERMINATE the pod.
```

---

# Week 2 — Quantization ablations (FP8, then AWQ-Marlin INT4)

Week 2 measures **the same sweep at lower precision** and compares against the committed FP16
baseline. Reuse the **same pod, SSH, bootstrap, and second shell** from sections 1–6 — only the
**vLLM launch flag** and the **config files** change. The Week-2 configs are exact mirrors of the
FP16 ones (identical workloads, concurrency `{1, 8, 32, 64}`, requests, `max_tokens`), so
`results/baseline_fp8.json` and `results/baseline_awq.json` are directly comparable to
`results/baseline_fp16.json`.

Configs already in the repo: `configs/baseline_fp8.yaml`, `configs/humaneval_fp8.yaml`,
`configs/baseline_awq.yaml`, `configs/humaneval_awq.yaml`.

> Run the precisions **one at a time** — one model per GPU, so stop the previous vLLM server before
> launching the next. Same honesty rules: dry-run first, `--confirm-paid-run` to spend, validate
> before publishing.

## FP8 (W8A8) — no new download

FP8 here is vLLM **online dynamic quantization** of the FP16 weights, so it reuses the checkpoint
already cached on your Network Volume — nothing new to download. Relaunch vLLM with
`--quantization fp8`:

```bash
tmux attach -t vllm     # Ctrl-c to stop FP16; or: tmux kill-session -t vllm && tmux new -s vllm
source .venv/bin/activate && export HF_HOME=/workspace/hf
python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8000 \
  --model Qwen/Qwen2.5-Coder-7B-Instruct --quantization fp8 \
  --served-model-name Qwen/Qwen2.5-Coder-7B-Instruct
# Ctrl-b d to detach
```

Then, in the second shell:

```bash
python scripts/check_endpoint.py --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-Coder-7B-Instruct --timeout-seconds 120 --stream

# latency / throughput
python scripts/run_h100_baseline.py --config configs/baseline_fp8.yaml \
  --base-url http://127.0.0.1:8000/v1                      # dry run
python scripts/run_h100_baseline.py --config configs/baseline_fp8.yaml \
  --base-url http://127.0.0.1:8000/v1 --confirm-paid-run   # paid

# HumanEval pass@1 (prefix FASTCODER_HUMANEVAL_PATH=... if the dataset isn't auto-detected — see 7b)
python scripts/run_humaneval_eval.py --config configs/humaneval_fp8.yaml \
  --base-url http://127.0.0.1:8000/v1                      # dry run
python scripts/run_humaneval_eval.py --config configs/humaneval_fp8.yaml \
  --base-url http://127.0.0.1:8000/v1 --confirm-paid-run   # paid

python scripts/validate_results.py results/baseline_fp8.json
python scripts/validate_results.py results/humaneval_fp8.json
```

## AWQ-Marlin INT4 — needs a pre-quantized checkpoint

Unlike FP8, AWQ loads a **separate, pre-quantized INT4 checkpoint** (~5–6 GB, downloaded once to the
Network Volume). **Confirm the model id on Hugging Face before spending** — Qwen publishes official
AWQ variants (conventionally a `-AWQ` suffix); if one isn't available for this model, self-quantize
with AutoAWQ. The configs default to `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ`; edit `model:` in both
`configs/baseline_awq.yaml` and `configs/humaneval_awq.yaml` if the real id differs.

```bash
tmux attach -t vllm     # stop the FP8 server first
source .venv/bin/activate && export HF_HOME=/workspace/hf
python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8000 \
  --model Qwen/Qwen2.5-Coder-7B-Instruct-AWQ --quantization awq_marlin --dtype float16 \
  --served-model-name Qwen/Qwen2.5-Coder-7B-Instruct-AWQ
# Ctrl-b d to detach
```

Then run the same flow against the AWQ configs (the `--model` matches the AWQ id):

```bash
python scripts/check_endpoint.py --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-Coder-7B-Instruct-AWQ --timeout-seconds 120 --stream

python scripts/run_h100_baseline.py --config configs/baseline_awq.yaml \
  --base-url http://127.0.0.1:8000/v1                      # dry run
python scripts/run_h100_baseline.py --config configs/baseline_awq.yaml \
  --base-url http://127.0.0.1:8000/v1 --confirm-paid-run   # paid

python scripts/run_humaneval_eval.py --config configs/humaneval_awq.yaml \
  --base-url http://127.0.0.1:8000/v1                      # dry run
python scripts/run_humaneval_eval.py --config configs/humaneval_awq.yaml \
  --base-url http://127.0.0.1:8000/v1 --confirm-paid-run   # paid

python scripts/validate_results.py results/baseline_awq.json
python scripts/validate_results.py results/humaneval_awq.json
```

## Copy off, then compare

Copy the four new JSONs off the pod (same `scp` as step 8) and **terminate the pod**. Back on the
laptop the three precisions are directly comparable (identical sweep): FP16 vs FP8 vs AWQ-INT4 on
latency, throughput, peak GPU memory, $/1M tokens, and HumanEval pass@1 — that quality-vs-cost
frontier is the Week-2 deliverable. Regenerate the Pareto plot over all three:

```bash
python scripts/generate_pareto.py \
  results/baseline_fp16.json results/baseline_fp8.json results/baseline_awq.json \
  --output results/pareto.png
```

Commit the validated JSONs + manifests the same way as Week 1 (allowlist them in `.gitignore`).

## Week 2 cheat sheet (precision switch only)

Bootstrap is identical to the Week-1 cheat sheet. Per precision: stop the previous vLLM, relaunch
with the flag(s) below, then run the matching configs (dry → `--confirm-paid-run` → validate).

| precision | vLLM flag(s) | model | configs |
| --- | --- | --- | --- |
| FP8 | `--quantization fp8` | `Qwen/Qwen2.5-Coder-7B-Instruct` (reuses FP16 weights) | `baseline_fp8.yaml`, `humaneval_fp8.yaml` |
| AWQ-INT4 | `--quantization awq_marlin --dtype float16` | `*-AWQ` checkpoint (confirm id) | `baseline_awq.yaml`, `humaneval_awq.yaml` |
