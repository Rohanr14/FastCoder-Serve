"""Async OpenAI-compatible benchmark client."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from fastcoder.benchmark.config import ModelServerConfig
from fastcoder.benchmark.metrics import (
    TOKEN_SOURCE_APPROXIMATE,
    TOKEN_SOURCE_USAGE,
    RequestMetric,
    estimate_token_count,
)


class OpenAICompatibleClient:
    """Minimal async client for `/v1/chat/completions` benchmark requests."""

    def __init__(
        self,
        config: ModelServerConfig,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._client = http_client or httpx.AsyncClient(timeout=config.timeout_seconds)
        self._owns_client = http_client is None

    async def __aenter__(self) -> OpenAICompatibleClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def chat_completion(
        self,
        *,
        experiment: str,
        workload: str,
        concurrency: int,
        request_index: int,
        messages: Sequence[Mapping[str, str]],
        max_tokens: int,
        temperature: float,
        stream: bool,
        sample_id: str | None = None,
        capture_text: bool = False,
    ) -> RequestMetric:
        """Send one chat-completion request and return timing metadata.

        When ``capture_text`` is true the assembled completion text is attached to the returned
        metric's ``response_text`` (used by HumanEval scoring). It defaults to false so the latency
        benchmark path never retains model output.
        """

        request_id = f"{workload}-c{concurrency}-{request_index}"
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": [dict(message) for message in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if stream:
            # Ask for a terminal usage chunk so output-token counts are exact. vLLM only emits
            # usage on a stream when include_usage is set; without it the client falls back to the
            # whitespace estimate. Timing is unaffected — the usage chunk carries no content and
            # arrives just before [DONE].
            payload["stream_options"] = {"include_usage": True}
        headers = self._headers()
        if stream:
            return await self._streaming_completion(
                experiment=experiment,
                workload=workload,
                concurrency=concurrency,
                request_id=request_id,
                sample_id=sample_id,
                payload=payload,
                headers=headers,
                capture_text=capture_text,
            )
        return await self._non_streaming_completion(
            experiment=experiment,
            workload=workload,
            concurrency=concurrency,
            request_id=request_id,
            sample_id=sample_id,
            payload=payload,
            headers=headers,
            capture_text=capture_text,
        )

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self._config.api_key:
            headers["authorization"] = f"Bearer {self._config.api_key}"
        return headers

    async def _non_streaming_completion(
        self,
        *,
        experiment: str,
        workload: str,
        concurrency: int,
        request_id: str,
        sample_id: str | None,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        capture_text: bool = False,
    ) -> RequestMetric:
        start = time.perf_counter()
        try:
            response = await self._client.post(
                self._config.chat_completions_url,
                json=payload,
                headers=headers,
            )
            end = time.perf_counter()
        except httpx.HTTPError as exc:
            end = time.perf_counter()
            return self._error_metric(
                experiment,
                workload,
                concurrency,
                request_id,
                start,
                end,
                None,
                str(exc),
                stream=False,
                sample_id=sample_id,
            )

        if response.is_error:
            return self._error_metric(
                experiment,
                workload,
                concurrency,
                request_id,
                start,
                end,
                response.status_code,
                response.text[:500],
                stream=False,
                sample_id=sample_id,
            )

        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            return self._error_metric(
                experiment,
                workload,
                concurrency,
                request_id,
                start,
                end,
                response.status_code,
                f"invalid JSON response: {exc}",
                stream=False,
                sample_id=sample_id,
            )

        text = _extract_non_streaming_text(body)
        usage = body.get("usage") if isinstance(body, dict) else None
        output_tokens, output_source = _completion_tokens_from_usage_or_text(usage, text)
        input_tokens = _prompt_tokens_from_usage(usage)
        model = str(body.get("model", "")) if isinstance(body, dict) else ""

        return RequestMetric(
            experiment=experiment,
            workload=workload,
            concurrency=concurrency,
            request_id=request_id,
            ok=True,
            status_code=response.status_code,
            start_time=start,
            end_time=end,
            first_token_time=None,
            output_tokens=output_tokens,
            output_token_source=output_source,
            input_tokens=input_tokens,
            model=model,
            response_chars=len(text),
            stream=False,
            sample_id=sample_id,
            response_text=text if capture_text else None,
        )

    async def _streaming_completion(
        self,
        *,
        experiment: str,
        workload: str,
        concurrency: int,
        request_id: str,
        sample_id: str | None,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        capture_text: bool = False,
    ) -> RequestMetric:
        start = time.perf_counter()
        chunks: list[str] = []
        chunk_arrival_times: list[float] = []
        first_token_time: float | None = None
        status_code: int | None = None
        usage: Any = None
        try:
            async with self._client.stream(
                "POST",
                self._config.chat_completions_url,
                json=payload,
                headers=headers,
            ) as response:
                status_code = response.status_code
                if response.is_error:
                    error_body = await response.aread()
                    end = time.perf_counter()
                    return self._error_metric(
                        experiment,
                        workload,
                        concurrency,
                        request_id,
                        start,
                        end,
                        response.status_code,
                        error_body.decode("utf-8", errors="replace")[:500],
                        stream=True,
                        sample_id=sample_id,
                    )
                async for line in response.aiter_lines():
                    event = _extract_streaming_event(line)
                    if event is None:
                        continue
                    if event.usage is not None:
                        usage = event.usage
                    if event.done:
                        break
                    content = event.content
                    arrival_time = time.perf_counter()
                    if content and first_token_time is None:
                        first_token_time = arrival_time
                    if content:
                        chunk_arrival_times.append(arrival_time)
                    chunks.append(content)
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            end = time.perf_counter()
            return self._error_metric(
                experiment,
                workload,
                concurrency,
                request_id,
                start,
                end,
                status_code,
                str(exc),
                stream=True,
                sample_id=sample_id,
            )

        end = time.perf_counter()
        text = "".join(chunks)
        output_tokens, output_source = _completion_tokens_from_usage_or_text(usage, text)
        return RequestMetric(
            experiment=experiment,
            workload=workload,
            concurrency=concurrency,
            request_id=request_id,
            ok=True,
            status_code=status_code,
            start_time=start,
            end_time=end,
            first_token_time=first_token_time,
            output_tokens=output_tokens,
            output_token_source=output_source,
            input_tokens=_prompt_tokens_from_usage(usage),
            model=self._config.model,
            response_chars=len(text),
            stream=True,
            chunk_arrival_times=tuple(chunk_arrival_times),
            sample_id=sample_id,
            response_text=text if capture_text else None,
        )

    def _error_metric(
        self,
        experiment: str,
        workload: str,
        concurrency: int,
        request_id: str,
        start: float,
        end: float,
        status_code: int | None,
        error: str,
        *,
        stream: bool = False,
        sample_id: str | None = None,
    ) -> RequestMetric:
        return RequestMetric(
            experiment=experiment,
            workload=workload,
            concurrency=concurrency,
            request_id=request_id,
            ok=False,
            status_code=status_code,
            start_time=start,
            end_time=end,
            first_token_time=None,
            output_tokens=0,
            output_token_source=TOKEN_SOURCE_APPROXIMATE,
            model=self._config.model,
            error=error,
            stream=stream,
            sample_id=sample_id,
        )


def _extract_non_streaming_text(body: Any) -> str:
    if not isinstance(body, dict):
        return ""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    message = first_choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        return content if isinstance(content, str) else ""
    text = first_choice.get("text")
    return text if isinstance(text, str) else ""


class _StreamingEvent:
    def __init__(self, *, content: str = "", done: bool = False, usage: Any = None) -> None:
        self.content = content
        self.done = done
        self.usage = usage


def _extract_streaming_event(line: str) -> _StreamingEvent | None:
    if not line.startswith("data:"):
        return None
    raw_data = line.removeprefix("data:").strip()
    if not raw_data:
        return None
    if raw_data == "[DONE]":
        return _StreamingEvent(done=True)
    body = json.loads(raw_data)
    if not isinstance(body, dict):
        return None
    usage = body.get("usage")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return _StreamingEvent(usage=usage)
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return _StreamingEvent(usage=usage)
    delta = first_choice.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        return _StreamingEvent(content=content if isinstance(content, str) else "", usage=usage)
    text = first_choice.get("text")
    return _StreamingEvent(content=text if isinstance(text, str) else "", usage=usage)


def _completion_tokens_from_usage_or_text(usage: Any, text: str) -> tuple[int, str]:
    if isinstance(usage, dict):
        completion_tokens = usage.get("completion_tokens")
        if isinstance(completion_tokens, int):
            return completion_tokens, TOKEN_SOURCE_USAGE
    return estimate_token_count(text), TOKEN_SOURCE_APPROXIMATE


def _prompt_tokens_from_usage(usage: Any) -> int | None:
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens")
        if isinstance(prompt_tokens, int):
            return prompt_tokens
    return None
