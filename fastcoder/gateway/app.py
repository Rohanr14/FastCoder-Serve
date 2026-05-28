"""FastAPI gateway app factory."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse

from fastcoder.gateway.auth import require_bearer_token
from fastcoder.gateway.config import GatewaySettings
from fastcoder.gateway.logging import configure_logging
from fastcoder.gateway.metrics import (
    metrics_response,
    record_error,
    record_request,
    record_upstream_latency,
)
from fastcoder.gateway.rate_limit import InMemoryRateLimiter

LOGGER = logging.getLogger("fastcoder.gateway")
CHAT_ROUTE = "/v1/chat/completions"


def create_app(
    settings: GatewaySettings | None = None,
    *,
    upstream_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    """Create the FastAPI gateway app."""

    resolved_settings = settings or GatewaySettings()
    configure_logging(resolved_settings.log_level)
    limiter = InMemoryRateLimiter(limit_per_minute=resolved_settings.rate_limit_per_minute)
    owns_client = upstream_client is None

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        client = upstream_client or httpx.AsyncClient(
            timeout=resolved_settings.request_timeout_seconds,
        )
        _app.state.upstream_client = client
        try:
            yield
        finally:
            if owns_client:
                await client.aclose()

    app = FastAPI(title="FastCoder Gateway", version="0.1.0", lifespan=lifespan)
    if upstream_client is not None:
        app.state.upstream_client = upstream_client

    async def auth_dependency(request: Request) -> str:
        return await require_bearer_token(request, resolved_settings)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        return metrics_response()

    @app.post(CHAT_ROUTE)
    async def chat_completions(
        request: Request,
        token: str = Depends(auth_dependency),
    ) -> Response:
        route = CHAT_ROUTE
        start = time.perf_counter()
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        record_on_exit = True
        try:
            decision = limiter.check(_rate_limit_key(request, token))
            if not decision.allowed:
                status_code = status.HTTP_429_TOO_MANY_REQUESTS
                record_error(route, "rate_limited")
                raise HTTPException(
                    status_code=status_code,
                    detail="rate limit exceeded",
                    headers={"Retry-After": str(int(decision.reset_seconds) + 1)},
                )

            payload = await _json_payload(request)
            upstream_headers = _upstream_headers(resolved_settings)
            client: httpx.AsyncClient = request.app.state.upstream_client
            if payload.get("stream") is True:
                streaming_response = await _stream_upstream_response(
                    client=client,
                    settings=resolved_settings,
                    payload=payload,
                    headers=upstream_headers,
                    route=route,
                    start=start,
                )
                record_on_exit = False
                return streaming_response

            upstream_start = time.perf_counter()
            upstream_response = await client.post(
                resolved_settings.upstream_chat_completions_url,
                json=payload,
                headers=upstream_headers,
            )
            record_upstream_latency(route, time.perf_counter() - upstream_start)
            status_code = upstream_response.status_code
            if upstream_response.is_error:
                record_error(route, "upstream_error")

            content_type = upstream_response.headers.get("content-type", "application/json")
            LOGGER.info(
                "proxied chat completion",
                extra={
                    "fastcoder_status_code": status_code,
                    "fastcoder_upstream_status_code": upstream_response.status_code,
                },
            )
            return Response(
                content=upstream_response.content,
                status_code=status_code,
                media_type=content_type,
            )
        except HTTPException:
            raise
        except httpx.HTTPError as exc:
            status_code = status.HTTP_502_BAD_GATEWAY
            record_error(route, "upstream_http_error")
            LOGGER.exception("upstream request failed", extra={"fastcoder_error": str(exc)})
            raise HTTPException(status_code=status_code, detail="upstream request failed") from exc
        finally:
            if record_on_exit:
                record_request("POST", route, status_code, time.perf_counter() - start)

    return app


async def _json_payload(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="request body must be JSON",
        ) from exc
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="request body must be a JSON object",
        )
    return payload


def _upstream_headers(settings: GatewaySettings) -> dict[str, str]:
    headers = {"content-type": "application/json"}
    if settings.upstream_api_key:
        headers["authorization"] = f"Bearer {settings.upstream_api_key}"
    return headers


async def _stream_upstream_response(
    *,
    client: httpx.AsyncClient,
    settings: GatewaySettings,
    payload: dict[str, Any],
    headers: dict[str, str],
    route: str,
    start: float,
) -> Response:
    upstream_start = time.perf_counter()
    request = client.build_request(
        "POST",
        settings.upstream_chat_completions_url,
        json=payload,
        headers=headers,
    )
    upstream_response = await client.send(request, stream=True)
    record_upstream_latency(route, time.perf_counter() - upstream_start)

    status_code = upstream_response.status_code
    content_type = upstream_response.headers.get("content-type", "text/event-stream")
    if upstream_response.is_error:
        body = await upstream_response.aread()
        await upstream_response.aclose()
        record_error(route, "upstream_error")
        record_request("POST", route, status_code, time.perf_counter() - start)
        return Response(
            content=body,
            status_code=status_code,
            media_type=content_type,
        )

    LOGGER.info(
        "streaming chat completion",
        extra={
            "fastcoder_status_code": status_code,
            "fastcoder_upstream_status_code": upstream_response.status_code,
        },
    )

    async def stream_body() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream_response.aiter_raw():
                if chunk:
                    yield chunk
        except httpx.HTTPError as exc:
            record_error(route, "upstream_stream_error")
            LOGGER.exception("upstream stream failed", extra={"fastcoder_error": str(exc)})
            raise
        finally:
            await upstream_response.aclose()
            record_request("POST", route, status_code, time.perf_counter() - start)

    return StreamingResponse(
        stream_body(),
        status_code=status_code,
        headers={"content-type": content_type},
    )


def _rate_limit_key(request: Request, token: str) -> str:
    if token != "auth-bypass":
        return f"token:{token}"
    if request.client:
        return f"ip:{request.client.host}"
    return "ip:unknown"
