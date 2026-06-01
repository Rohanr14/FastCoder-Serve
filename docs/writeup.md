# Precision frontier for Qwen2.5-Coder-7B on a single H100

**Question:** to serve `Qwen/Qwen2.5-Coder-7B-Instruct` on one H100, which weight precision should
you pick — FP16, FP8, or AWQ-Marlin INT4? This maps the latency / throughput / quality / cost
frontier across all three, on identical workloads, and turns it into a serving recommendation.

Every number below comes from a committed result JSON that passes `scripts/validate_results.py`.
Output-token counts are exact (`usage.completion_tokens`); see [methodology](methodology.md) and
reproduce from scratch with [runpod_setup.md](runpod_setup.md).

## TL;DR

- **FP8 is a near-free win on H100:** **+43% throughput, −30% cost/token, −32% p50 latency** vs
  FP16, at **identical** HumanEval pass@1 (144/164). Use it.
- **AWQ-INT4 is not worth it here:** +34% throughput but **worse p95/p99 tail latency than even
  FP16**, and −0.6 pp quality. INT4's advantage is weight footprint, which doesn't bind for a 7B on
  80 GB.
- **Under an interactive SLO (p99 TTFT ≤ 250 ms), FP8 sustains ~32% more throughput than FP16 at
  25% lower cost** — and at a looser 500 ms SLO the gap widens dramatically because FP16 sits right
  at the latency cliff (below).
- **Long-context (4k) saturates at concurrency 32** for all three precisions — pushing to 64
  *collapses* throughput and spikes TTFT 4–5×. Cap concurrency per workload shape.
- **Prefix caching is the high-concurrency fix for shared context:** on a shared-4K-prefix workload
  (FP8), enabling it cut tail TTFT 85% and *more than doubled* throughput at concurrency 64 (+163%).
- **Speculative decoding made it *worse*, not better:** draft-model and n-gram both *raised* latency
  (ITL 7× worse even at concurrency 1) and cut throughput up to 10×. A fast 7B on a compute-rich H100
  has no latency to recover — speculation is for slow/large models at low QPS.

## Setup

| | |
| --- | --- |
| Model | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Hardware | single NVIDIA H100 80GB HBM3 (RunPod, assumed $2.20/hr) |
| Server | vLLM 0.21.0, OpenAI-compatible, measured **in-pod** vs `127.0.0.1:8000` (no WAN/proxy) |
| Precisions | FP16 (`--dtype float16`); FP8 (`--quantization fp8`, online); AWQ-INT4 (`--quantization awq_marlin`, pre-quantized `-AWQ` checkpoint) |
| Workloads | `short_chat_256_256`, `long_context_4k_512` |
| Sweep | concurrency `{1, 8, 32, 64}`, 50 requests each, streaming |
| Quality | HumanEval pass@1, 164 problems, one greedy sample each, scored in `human_eval`'s sandbox |

Configs are `configs/{baseline,humaneval}_{fp16,fp8,awq}.yaml` — identical across precisions so the
results are directly comparable.

## The frontier (aggregate across the sweep)

| precision | lat p50 (s) | lat p95 (s) | lat p99 (s) | TTFT p50 (s) | ITL p50 (ms) | throughput (tok/s) | $/1M out | pass@1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FP16 | 1.63 | 2.52 | 2.61 | 0.174 | 6.3 | 516 | $1.18 | 0.878 (144/164) |
| **FP8** | **1.11** | **1.98** | **2.02** | **0.136** | **4.6** | **737** | **$0.83** | **0.878 (144/164)** |
| AWQ-INT4 | 1.16 | 2.71 | 2.73 | 0.175 | 4.7 | 692 | $0.88 | 0.872 (143/164) |

Throughput-vs-latency operating points are plotted in
[`results/pareto.png`](../results/pareto.png) (color = precision, marker = workload).

FP8 improves *every* axis at once: faster per-token decode (ITL 4.6 vs 6.3 ms), lower first-token
latency, higher throughput, lower cost — and HumanEval pass@1 is bit-for-identical to FP16. AWQ-INT4
matches FP8 on median latency and decode but has the **worst tail latency of the three** and drops
one HumanEval problem.

## Capacity under an SLO

Aggregate throughput mixes fast and slow operating points, so it understates what the server can do
*at a target latency*. The benchmark is closed-loop (concurrency = in-flight requests, not an
arrival rate), so "max sustained throughput under an SLO" = the highest-throughput measured
operating point whose p99 TTFT still meets the SLO. Reproduce with `python scripts/slo_analysis.py`.

**p99 TTFT ≤ 250 ms (strict, interactive code assistant):**

| precision | throughput (tok/s) | $/1M out | operating point |
| --- | --- | --- | --- |
| FP16 | 3,604 | $0.170 | short_chat/c32 (TTFT 0.187s) |
| **FP8** | **4,757** | **$0.128** | short_chat/c32 (TTFT 0.142s) |
| AWQ-INT4 | 4,653 | $0.131 | short_chat/c32 (TTFT 0.180s) |

→ **FP8 serves +32% throughput at −25% cost vs FP16 under the same latency SLO.**

**p99 TTFT ≤ 1000 ms (relaxed):**

| precision | throughput (tok/s) | $/1M out | operating point |
| --- | --- | --- | --- |
| FP16 | 7,120 | $0.086 | short_chat/c64 (TTFT 0.503s) |
| **FP8** | **9,204** | **$0.066** | short_chat/c64 (TTFT 0.449s) |
| AWQ-INT4 | 8,712 | $0.070 | short_chat/c64 (TTFT 0.389s) |

→ **FP8 +29% throughput at −23% cost.**

**The 500 ms cliff.** At a p99 TTFT ≤ 500 ms SLO, FP16's best operating point (short_chat/c64) just
*misses* at **503 ms**, so FP16 falls back to c32 (3,604 tok/s) while FP8 (449 ms) and AWQ (389 ms)
clear it at c64 — FP8 then serves **9,204 tok/s, ~2.5× FP16**. This is a real serving phenomenon
(a few ms of TTFT can cross an SLO and unlock a much higher operating point), but note FP16 is right
on the boundary, so treat the 2.5× as an illustration of SLO sensitivity rather than a robust
steady-state ratio — the ±32% / ±29% at the 250 ms and 1000 ms tiers are the dependable numbers.

**Saturation — peak throughput operating point per workload:**

| precision | short_chat peak | long_context peak |
| --- | --- | --- |
| FP16 | 7,120 tok/s @ c64 | 1,007 tok/s @ **c32** |
| FP8 | 9,204 tok/s @ c64 | 1,156 tok/s @ **c32** |
| AWQ-INT4 | 8,712 tok/s @ c64 | 1,029 tok/s @ **c32** |

Long-context throughput **peaks at c32 and drops at c64** for all three (e.g. FP8: 1,156 → 757
tok/s) while p99 TTFT jumps to ~1.9–2.6 s — the 4k-token KV footprint oversubscribes the scheduler.
Short-chat keeps scaling to c64. **Takeaway: set max concurrency by workload shape** — ~32 for
long-context, higher for short-chat.

## Prefix caching: the high-concurrency fix for shared context

The saturation above is redundant prefill — at high concurrency the server recomputes long prompts
over and over. When requests **share** a prefix (a fixed system prompt, a pinned codebase/document,
agent loops — a very common production pattern), vLLM's automatic prefix caching computes that
prefix once and reuses its KV. To measure the upper bound I built `shared_prefix_4k_512`: one fixed
~4K-token module shared verbatim across all requests, only the trailing question varying. Same FP8
server, run with `--enable-prefix-caching` vs `--no-enable-prefix-caching`.

| concurrency | TTFT p50: off → on | throughput: off → on |
| --- | --- | --- |
| 1 | 0.139 → 0.059 s (−58%) | 212 → 227 tok/s (+7%) |
| 8 | 0.154 → 0.071 s (−54%) | 881 → 1,422 tok/s (+61%) |
| 32 | 0.992 → 0.638 s (−36%) | 1,191 → 2,634 tok/s (+121%) |
| **64** | **3.26 → 0.75 s (−77%)** | **1,328 → 3,494 tok/s (+163%)** |

Aggregate: tail TTFT (p99) drops **5.98 s → 0.89 s (−85%)**, throughput **+30%**, cost **−23%** — at
byte-identical outputs (KV reuse doesn't change generation, so HumanEval is unchanged and not
re-run). **The win scales with concurrency**: the more in-flight requests share the prefix, the more
redundant prefill caching removes — turning the high-concurrency regime from a liability into a 2.6×
throughput win.

**Honest scope:** this is the best case (a fully shared 4K prefix). Real gains scale with how much
prefix is actually shared; a distinct-prompt workload like `long_context_4k_512` sees ~no benefit.
The rule is simple — enable it whenever traffic has shared prefixes; it costs nothing when it
doesn't hit.

## Why FP8 beats INT4 here

- **FP8 has native Hopper tensor-core support** — W8A8 FP8 runs close to the hardware's peak, so it
  improves both compute throughput and decode latency with no quality cost.
- **AWQ-INT4 dequantizes 4-bit weights to compute**; at a 7B on an 80 GB H100 the model already fits
  comfortably, so INT4's only real advantage — a smaller weight footprint — doesn't buy anything,
  while the dequant path shows up as worse tail latency.
- INT4's niche is when weights *don't* fit (larger models, smaller GPUs) or when you need maximum
  KV-cache headroom. Neither binds here.

## A note on GPU memory (honest)

Peak GPU memory is ~73.6 GB for **all three** precisions — not a measurement error. vLLM reserves
its `gpu_memory_utilization` target (~0.90 × 80 GB) up front regardless of weight size; quantization
frees *weight* memory that becomes additional **KV-cache headroom** (higher max concurrency / longer
context), not a lower peak reservation. To quantify the memory benefit you would measure max
concurrency / KV blocks at a fixed utilization, not peak reserved bytes — a good follow-up.

## Speculative decoding: a negative result

Speculative decoding — a small draft model (here `Qwen2.5-Coder-0.5B`), or n-gram/prompt-lookup,
proposes tokens the target verifies — is supposed to cut latency. On this setup it **hurt across the
board.** Same FP8 server, same sweep, versus `baseline_fp8`:

| short_chat | ITL p50: FP8 → draft → n-gram | throughput: FP8 → draft → n-gram |
| --- | --- | --- |
| c1 | 4.1 → 29.7 → 5.3 ms | 241 → 101 → 193 tok/s |
| c64 | 5.2 → 62.2 → 92.7 ms | 9,204 → 2,324 → 912 tok/s |

The tell is **concurrency 1**, where speculation is *supposed* to win: draft ITL is **7× worse**
(29.7 vs 4.1 ms). On an H100 the 7B target already decodes at ~4 ms/token, so there's almost no
latency to recover — and the draft + verification add more overhead than they save. At high
concurrency it's worse: batching already saturates the GPU, so drafting/verifying is pure overhead
and throughput collapses up to **10×**.

(Acceptance rate would sharpen the explanation, but I couldn't capture it cleanly — vLLM's
spec-decode counters read zero at capture time, against a by-then-idle server. The latency and
throughput data make the case on their own.)

**Takeaway:** speculative decoding is for **latency-bound, low-QPS serving of slow or large models** —
not a compute-rich H100 batching a fast 7B. Knowing when *not* to reach for a technique is the point.

## Production recommendations

Putting the frontier together into "what would I actually deploy?":

- **Precision — default to FP8.** On Hopper it's the free lunch: +43% throughput, −30% cost, lower
  latency, identical HumanEval pass@1 vs FP16. Use FP16 only to match a reference bit-for-bit or on
  pre-Hopper GPUs without FP8. Reach for **AWQ-INT4 only when weights don't fit** (bigger models,
  smaller GPUs) or you need maximum KV headroom — at 7B on 80 GB it loses to FP8 on tail latency for
  no benefit.
- **Concurrency — cap by workload shape, not globally.** Short, bounded outputs scale to high
  concurrency; long-context (large KV) saturates early — here it peaked at **c32** and *regressed*
  at c64 (throughput down, TTFT up ~5×). Set per-workload limits, or route long-context to its own
  pool, instead of one global max.
- **Prefix caching — on by default if traffic shares prefixes.** For shared-context patterns (fixed
  system prompt, pinned codebase, agent loops) it cut tail TTFT 85% and doubled throughput at high
  concurrency, and costs nothing when prefixes don't overlap.
- **Speculative decoding — skip it for this profile.** Draft-model and n-gram both hurt here (the
  target is already fast and batching saturates compute). Reserve it for latency-bound, low-QPS
  serving of slow or large models.
- **Capacity planning — size to an SLO, not to peak throughput.** Pick the latency target first,
  then read max sustained throughput at it (`scripts/slo_analysis.py`). Under a 250 ms p99 TTFT SLO
  this stack sustains ~4,800 tok/s on FP8 (~$0.13/1M). A few ms of TTFT can cross an SLO threshold
  and unlock a much higher operating point, so measure at *your* target, not at headline peak.

## Limitations

- One model, one GPU, one serving backend (vLLM 0.21.0). No cross-backend (TensorRT-LLM / SGLang)
  comparison yet.
- Closed-loop concurrency sweep, not an open-loop arrival-rate / queueing study.
- Cost assumes $2.20/hr and aggregate throughput; treat $/token as an order-of-magnitude figure.
- AWQ-INT4 used the official `-AWQ` checkpoint; a different INT4 recipe (e.g. GPTQ) could differ.
- The prefix-caching result is a shared-prefix upper bound; production gains depend on real overlap.
- Speculative-decoding acceptance rate was not reliably captured (vLLM counters read zero at capture);
  the negative result rests on the latency/throughput data.

## Reproduce

```bash
# on an H100 pod (see docs/runpod_setup.md), per precision:
#   FP16: --dtype float16 | FP8: --quantization fp8 | AWQ: --quantization awq_marlin
python scripts/run_h100_baseline.py --config configs/baseline_fp8.yaml \
  --base-url http://127.0.0.1:8000/v1 --confirm-paid-run
python scripts/run_humaneval_eval.py --config configs/humaneval_fp8.yaml \
  --base-url http://127.0.0.1:8000/v1 --confirm-paid-run

# locally, on the committed result JSONs:
python scripts/validate_results.py results/baseline_fp8.json
python scripts/slo_analysis.py
python scripts/generate_pareto.py \
  results/baseline_fp16.json results/baseline_fp8.json results/baseline_awq.json \
  --output results/pareto.png
```
