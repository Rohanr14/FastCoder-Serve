"""Prometheus metrics for the gateway."""

from __future__ import annotations

from fastapi import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUEST_COUNT = Counter(
    "fastcoder_gateway_requests_total",
    "Gateway request count.",
    ["method", "route", "status"],
)
ERROR_COUNT = Counter(
    "fastcoder_gateway_errors_total",
    "Gateway error count.",
    ["route", "kind"],
)
REQUEST_LATENCY = Histogram(
    "fastcoder_gateway_request_latency_seconds",
    "Gateway end-to-end request latency.",
    ["route"],
)
UPSTREAM_LATENCY = Histogram(
    "fastcoder_gateway_upstream_latency_seconds",
    "Gateway upstream request latency.",
    ["route"],
)
TTFT_PLACEHOLDER = Histogram(
    "fastcoder_gateway_ttft_seconds",
    "Placeholder TTFT histogram for future streaming instrumentation.",
    ["route"],
)


def record_request(method: str, route: str, status_code: int, latency_seconds: float) -> None:
    REQUEST_COUNT.labels(method=method, route=route, status=str(status_code)).inc()
    REQUEST_LATENCY.labels(route=route).observe(latency_seconds)


def record_error(route: str, kind: str) -> None:
    ERROR_COUNT.labels(route=route, kind=kind).inc()


def record_upstream_latency(route: str, latency_seconds: float) -> None:
    UPSTREAM_LATENCY.labels(route=route).observe(latency_seconds)


def metrics_response() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
