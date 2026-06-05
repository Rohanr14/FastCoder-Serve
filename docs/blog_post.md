# I rented one H100 to answer: which precision should you serve a 7B code model at?

FP16, FP8, or INT4? Everyone has an opinion; I wanted measured numbers. So I built a small,
reproducible benchmark around vLLM serving **Qwen2.5-Coder-7B-Instruct** on a single H100 and ran
the same workloads at all three precisions — latency, throughput, HumanEval quality, and cost.

Here's what I found, and how I kept myself honest.

## The answer: FP8 is close to a free win

Same model, same workloads (a concurrency sweep of short chat + 4K-context code review), same H100,
vLLM 0.21.0:

| precision | throughput | $/1M output tok | p50 latency | HumanEval pass@1 |
| --- | --- | --- | --- | --- |
| FP16 | 516 tok/s | $1.18 | 1.63 s | 0.878 (144/164) |
| **FP8** | **737 tok/s** | **$0.83** | **1.11 s** | **0.878 (144/164)** |
| AWQ-INT4 | 692 tok/s | $0.88 | 1.16 s | 0.872 (143/164) |

**FP8 gives +43% throughput, −30% cost/token, and −32% p50 latency over FP16 — at bit-for-identical
HumanEval pass@1.** On Hopper, FP8 runs on native tensor cores, so you pay essentially nothing in
quality for a large efficiency gain. If you're serving a 7B on an H100, this is the default.

## The surprise: INT4 didn't pay off

I expected 4-bit AWQ to win on efficiency. It didn't. It beat FP16 on throughput (+34%) but had the
**worst tail latency of the three** (p99 latency above FP16) and dropped one HumanEval problem. The
reason: INT4's advantage is a smaller weight *footprint*, and a 7B already fits comfortably in 80 GB
— so the only thing INT4 buys you here is the dequantization overhead. INT4 earns its keep when the
model *doesn't* fit or you need maximum KV-cache headroom. Neither was true here.

## The bigger surprise: speculative decoding made it *worse*

I tried speculative decoding — a small draft model proposing tokens the 7B verifies — expecting
lower latency. It did the opposite: inter-token latency got **~7× worse even at low load**. On an
H100 the 7B already decodes at ~4 ms/token, so there's almost no latency to recover, and the draft +
verification just pile on overhead; under load, throughput collapsed up to 10×. Speculative decoding
is for *slow or large* models at low QPS — not a fast 7B on a compute-rich GPU. Knowing when **not**
to reach for a technique is half the job.

Prefix caching was the optimization that *did* pay off: on a shared-context workload (one fixed 4K
prompt, many questions — think a pinned codebase), enabling it cut tail TTFT 85% and **more than
doubled throughput at high concurrency**. Turn it on whenever your traffic shares prefixes; it costs
nothing when it doesn't.

## Capacity is the number that matters

Aggregate throughput hides what a server can do *at a latency target*. So I computed max sustained
throughput under a p99 time-to-first-token SLO:

- **Under a 250 ms p99 TTFT SLO, FP8 sustains ~4,800 tok/s vs FP16's ~3,600 — +32% throughput at
  −25% cost.**
- Long-context (4K) throughput **peaks at concurrency 32 and collapses at 64** across all precisions
  — the KV footprint oversubscribes the scheduler. Short-chat keeps scaling. Lesson: cap max
  concurrency by workload shape, not globally.

I also load-tested it **open-loop** — Poisson arrivals at a fixed rate with no concurrency cap, the
way real traffic actually hits. The FP8 server holds the 250 ms p99 TTFT SLO up to ~37 req/s of
short-chat, then degrades *gracefully*: latency climbs and ~17% of requests miss the SLO at 2×
capacity, with no crashes. "Sustainable QPS at an SLO" — not peak throughput — is what you actually
provision against.

## The part I'm actually proud of: keeping it honest

Benchmarks are easy to fool yourself with. Some guardrails I built in:

- **No number ships unless it comes from a committed result file that passes a validator.** Nothing
  is typed by hand into the README.
- **I caught my own bug.** My first throughput numbers used a whitespace token estimate because the
  streaming responses didn't return usage — it undercounted code tokens by ~1.6×. I fixed the client
  to request exact token counts and **re-ran the FP16 baseline** so the whole comparison is exact.
- **I report what I *didn't* measure.** Peak GPU memory is identical (~73.6 GB) across precisions —
  not a bug: vLLM reserves its memory target regardless of weight size, so quantization's savings
  become KV-cache headroom, not lower peak. Claiming "FP8 uses less memory" from that number would
  be wrong, so I don't.

## Reproduce it — or point it at your own endpoint

The whole thing is open-source and packaged: `pip install fastcoder-serve`, then aim it at any
OpenAI-compatible endpoint for the same validated latency/throughput/quality/cost Pareto and capacity
report. One H100-hour reproduces the entire study — pod setup, configs, runners, the SLO analysis,
and a from-scratch RunPod guide are all in the repo.

*Stack: vLLM 0.21.0 · Qwen2.5-Coder-7B-Instruct · single H100 80GB · FP16 / FP8 / AWQ-Marlin INT4.*
