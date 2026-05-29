"""OpenAI-compatible endpoint pre-flight checks."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, Field


class EndpointCheckReport(BaseModel):
    base_url: str
    chat_url: str
    model: str
    stream: bool
    connectivity_ok: bool = False
    connectivity_status_code: int | None = None
    health_available: bool = False
    health_status_code: int | None = None
    models_available: bool = False
    models_status_code: int | None = None
    chat_ok: bool = False
    chat_status_code: int | None = None
    stream_chunk_count: int = 0
    stream_terminated: bool | None = None
    errors: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        if not self.chat_ok:
            return False
        if self.stream:
            return self.stream_chunk_count > 1 and self.stream_terminated is True
        return True


async def check_openai_compatible_endpoint(
    *,
    base_url: str,
    api_key: str | None = None,
    model: str | None = None,
    timeout_seconds: float = 10.0,
    stream: bool = False,
) -> EndpointCheckReport:
    """Probe an OpenAI-compatible endpoint without requiring vLLM-specific routes."""

    api_base = base_url.rstrip("/")
    selected_model = model or "fake-coder"
    report = EndpointCheckReport(
        base_url=api_base,
        chat_url=f"{api_base}/chat/completions",
        model=selected_model,
        stream=stream,
    )
    headers = _headers(api_key)

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        await _check_connectivity(client, report)
        await _check_health(client, report)
        await _check_models(client, report, headers)
        if stream:
            await _check_streaming_chat(client, report, headers)
        else:
            await _check_non_streaming_chat(client, report, headers)

    return report


async def fetch_vllm_version(
    *,
    base_url: str,
    api_key: str | None = None,
    timeout_seconds: float = 10.0,
) -> str | None:
    """Best-effort read of the live server version from vLLM's ``/version`` route.

    Returns the reported version string, or ``None`` when the route is missing,
    unreachable, or does not report a version. Never raises: an absent ``/version``
    endpoint must not fail a benchmark run, only leave the field unmeasured.
    """

    version_url = f"{_origin_url(base_url)}/version"
    headers = _headers(api_key)
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(version_url, headers=headers)
    except httpx.HTTPError:
        return None
    if response.is_error:
        return None
    try:
        body = response.json()
    except json.JSONDecodeError:
        return None
    if not isinstance(body, dict):
        return None
    version = body.get("version")
    if isinstance(version, str) and version.strip():
        return version.strip()
    return None


async def _check_connectivity(client: httpx.AsyncClient, report: EndpointCheckReport) -> None:
    try:
        response = await client.get(_origin_url(report.base_url))
    except httpx.HTTPError as exc:
        report.errors.append(f"basic connectivity failed: {exc}")
        return
    report.connectivity_ok = True
    report.connectivity_status_code = response.status_code
    if response.status_code >= 500:
        report.notes.append(f"base origin returned HTTP {response.status_code}")


async def _check_health(client: httpx.AsyncClient, report: EndpointCheckReport) -> None:
    try:
        response = await client.get(f"{_origin_url(report.base_url)}/health")
    except httpx.HTTPError as exc:
        report.notes.append(f"/health unavailable: {exc}")
        return
    report.health_status_code = response.status_code
    report.health_available = response.status_code < 400
    if not report.health_available:
        report.notes.append(f"/health optional check returned HTTP {response.status_code}")


async def _check_models(
    client: httpx.AsyncClient,
    report: EndpointCheckReport,
    headers: dict[str, str],
) -> None:
    try:
        response = await client.get(f"{report.base_url}/models", headers=headers)
    except httpx.HTTPError as exc:
        report.notes.append(f"/v1/models unavailable: {exc}")
        return
    report.models_status_code = response.status_code
    report.models_available = response.status_code < 400
    if not report.models_available:
        report.notes.append(f"/v1/models optional check returned HTTP {response.status_code}")


async def _check_non_streaming_chat(
    client: httpx.AsyncClient,
    report: EndpointCheckReport,
    headers: dict[str, str],
) -> None:
    try:
        response = await client.post(
            report.chat_url,
            json=_chat_payload(report, False),
            headers=headers,
        )
    except httpx.HTTPError as exc:
        report.errors.append(f"chat completions request failed: {exc}")
        return
    report.chat_status_code = response.status_code
    if response.is_error:
        report.errors.append(
            f"chat completions returned HTTP {response.status_code}: {response.text[:500]}"
        )
        return
    try:
        body = response.json()
    except json.JSONDecodeError as exc:
        report.errors.append(f"chat completions returned invalid JSON: {exc}")
        return
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or not choices:
        report.errors.append("chat completions response did not include choices")
        return
    report.chat_ok = True


async def _check_streaming_chat(
    client: httpx.AsyncClient,
    report: EndpointCheckReport,
    headers: dict[str, str],
) -> None:
    try:
        async with client.stream(
            "POST",
            report.chat_url,
            json=_chat_payload(report, True),
            headers=headers,
        ) as response:
            report.chat_status_code = response.status_code
            if response.is_error:
                body = await response.aread()
                report.errors.append(
                    "streaming chat completions returned HTTP "
                    f"{response.status_code}: {body.decode('utf-8', errors='replace')[:500]}"
                )
                return
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                report.stream_chunk_count += 1
                if _is_done_line(line):
                    report.stream_terminated = True
                    break
                if _has_finish_reason(line):
                    report.stream_terminated = True
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        report.errors.append(f"streaming chat completions request failed: {exc}")
        return

    if report.stream_chunk_count <= 1:
        report.errors.append("streaming response did not return more than one SSE chunk")
        report.stream_terminated = False
        return
    if report.stream_terminated is False:
        report.errors.append("streaming response ended without a valid termination marker")
        return
    if report.stream_terminated is None:
        report.errors.append("stream ended without [DONE] or finish_reason")
        report.stream_terminated = False
        return
    report.chat_ok = True


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    return headers


def _chat_payload(report: EndpointCheckReport, stream: bool) -> dict[str, Any]:
    return {
        "model": report.model,
        "messages": [
            {"role": "system", "content": "You are a concise endpoint health checker."},
            {"role": "user", "content": "Reply with ok."},
        ],
        "max_tokens": 8,
        "temperature": 0,
        "stream": stream,
    }


def _origin_url(base_url: str) -> str:
    parts = urlsplit(base_url)
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


def _is_done_line(line: str) -> bool:
    return line.removeprefix("data:").strip() == "[DONE]"


def _has_finish_reason(line: str) -> bool:
    raw_data = line.removeprefix("data:").strip()
    if not raw_data or raw_data == "[DONE]":
        return raw_data == "[DONE]"
    body = json.loads(raw_data)
    if not isinstance(body, dict):
        return False
    choices = body.get("choices")
    if not isinstance(choices, list):
        return False
    return any(
        isinstance(choice, dict) and choice.get("finish_reason") is not None
        for choice in choices
    )
