# Using FastCoder-Serve as a tool

FastCoder-Serve installs as a CLI so you can point it at **any** OpenAI-compatible endpoint (vLLM,
TGI, a hosted API, your own gateway) and get an **honest, schema-validated** Pareto + SLO report.
The rigor is the selling point: every number comes from a result file that passes
`fastcoder validate`, output-token counts are exact, and each run is captured in a reproducibility
manifest.

## Install

```bash
pip install -e .            # from a checkout
# pip install fastcoder-serve   # once published to PyPI
```

This registers the `fastcoder` command. Subcommands:

| command | does |
| --- | --- |
| `fastcoder endpoint` | preflight an OpenAI-compatible endpoint (chat + streaming) |
| `fastcoder config` | validate a benchmark config YAML |
| `fastcoder bench` | run a benchmark from a config and write a result JSON |
| `fastcoder validate` | schema + sanity-check a result JSON |
| `fastcoder slo` | max sustained throughput under a p99 TTFT SLO, per result |
| `fastcoder pareto` | render the throughput-vs-latency plot |

## Example

```bash
# 1. point a config at your endpoint (copy configs/local_smoke.yaml and edit model_server.base_url),
#    then sanity-check it and the endpoint:
fastcoder config   --config my_endpoint.yaml
fastcoder endpoint --base-url http://localhost:8000/v1 --model my-model --stream

# 2. run, validate, analyze:
fastcoder bench    --config my_endpoint.yaml
fastcoder validate results/my_endpoint.json
fastcoder slo      "mine=results/my_endpoint.json"
fastcoder pareto   results/my_endpoint.json --output results/pareto.png
```

Each subcommand accepts the same flags as the underlying script (`fastcoder <cmd> --help`). Paid /
GPU runs stay behind the same dry-run gating as the in-repo runners — nothing hits an endpoint
without you asking.

## Why it's trustworthy

- **No number without a validated source.** `fastcoder validate` enforces the result schema and
  monotonic-percentile / count-consistency sanity checks; the repo's own published numbers all pass it.
- **Exact tokens.** Streaming requests ask for `usage` so throughput/cost use real token counts, not
  a word-count estimate.
- **Reproducible.** Runs record git commit, config hash, and platform (secrets redacted) under
  `results/manifests/`.
