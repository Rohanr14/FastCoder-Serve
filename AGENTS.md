# FastCoder-Serve Agent Instructions

- Always read `docs/project_bible.md` before major work and treat it as the source of truth.
- Keep the project focused on LLM inference serving, benchmarking, observability, and measurement.
- Never add fine-tuning, RAG, LangChain, agents, vector databases, closed-model comparisons, consumer-GPU benchmark claims, Discord bots, user accounts, SSO, or unrelated frontend complexity.
- Prefer config-driven experiments over hard-coded benchmark variants.
- Every change should include tests or a clear reason tests are not applicable.
- Do not run large model downloads, paid APIs, CUDA-only paths, H100 benchmarks, RunPod jobs, or long GPU workloads unless explicitly asked.
- Maintain reproducibility and measured-results honesty: do not claim speedups or quality numbers until they are measured and committed with methodology.
- Keep local smoke tests CPU-safe and small.
