from __future__ import annotations

import httpx
import pytest

from fastcoder.gateway.app import create_app
from fastcoder.gateway.config import GatewaySettings
from scripts.fake_openai_server import app as fake_app


@pytest.mark.asyncio
async def test_gateway_health_metrics_auth_and_proxy() -> None:
    upstream_transport = httpx.ASGITransport(app=fake_app)
    async with httpx.AsyncClient(
        transport=upstream_transport,
        base_url="http://upstream",
    ) as upstream_client:
        gateway = create_app(
            GatewaySettings(
                api_key="test-token",
                upstream_base_url="http://upstream/v1",
                auth_bypass=False,
                rate_limit_per_minute=100,
            ),
            upstream_client=upstream_client,
        )
        gateway_transport = httpx.ASGITransport(app=gateway)
        async with httpx.AsyncClient(
            transport=gateway_transport,
            base_url="http://gateway",
        ) as client:
            health = await client.get("/health")
            metrics = await client.get("/metrics")
            unauthorized = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )
            proxied = await client.post(
                "/v1/chat/completions",
                headers={"authorization": "Bearer test-token"},
                json={
                    "model": "fake-coder",
                    "messages": [{"role": "user", "content": "What does a dict do?"}],
                },
            )

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert metrics.status_code == 200
    assert "fastcoder_gateway_requests_total" in metrics.text
    assert unauthorized.status_code == 401
    assert proxied.status_code == 200
    assert proxied.json()["model"] == "fake-coder"
