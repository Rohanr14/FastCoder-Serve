# Observability demo

A one-command, **CPU-only** demo of the FastCoder-Serve serving + observability stack. It runs the
FastCoder gateway in front of the fake OpenAI server, with Prometheus scraping the gateway and a
provisioned Grafana dashboard, driven by a continuous load generator — so you get **live latency,
throughput, and error dashboards under load, with no GPU**.

> This demonstrates the *system* (gateway instrumentation, Prometheus/Grafana wiring, behavior under
> load), not H100 serving performance. The measured H100 numbers live in `results/` and
> [writeup.md](writeup.md).

## Run it

```bash
docker compose -f docker/docker-compose.yml up --build
```

Then open:

| service | URL | notes |
| --- | --- | --- |
| Grafana | http://localhost:3000 | anonymous (no login); the **FastCoder-Serve Gateway** dashboard auto-loads |
| Prometheus | http://localhost:9090 | targets + raw queries |
| Gateway | http://localhost:8000 | `/metrics`, `/v1/chat/completions` |

The `load` service sends ~8 concurrent streaming + non-streaming requests continuously, so the
dashboard fills within ~15 s. Stop with `Ctrl-C`, then `docker compose -f docker/docker-compose.yml down`.

## What the dashboard shows

Every panel is backed by the gateway's real Prometheus metrics (`fastcoder/gateway/metrics.py`):

- **Throughput** — `sum(rate(fastcoder_gateway_requests_total[1m]))`
- **Error rate** — `sum(rate(fastcoder_gateway_errors_total[1m]))`
- **Gateway latency p50 / p95 / p99** — `histogram_quantile` over `fastcoder_gateway_request_latency_seconds_bucket`
- **Request rate by HTTP status**
- **Gateway vs upstream p95 latency** — the proxy overhead the gateway adds over the backend.

A test (`tests/test_observability.py`) asserts the dashboard only references metrics that the gateway
actually defines, so it can't silently rot.

> **Tip:** the demo sets a high gateway rate limit so traffic flows. Lower
> `FASTCODER_RATE_LIMIT_PER_MINUTE` in `docker/docker-compose.yml` to watch the gateway shed load
> with HTTP 429s — the *request rate by HTTP status* panel shows it live (a nice "the rate limiter
> actually works" demo).

## Without Docker

Drive the same load against a locally-run gateway (`make gateway` in front of `make fake-server`),
then scrape `/metrics`:

```bash
python scripts/demo_load.py --base-url http://localhost:8000/v1 --api-key dev-token \
  --concurrency 8 --duration-seconds 20
curl -s http://localhost:8000/metrics | grep fastcoder_gateway_requests_total
```

## Capturing it

Once traffic is flowing, screenshot the Grafana dashboard (or record a short GIF) under load for the
blog post / LinkedIn — a live dashboard is a strong visual that most portfolios lack.
