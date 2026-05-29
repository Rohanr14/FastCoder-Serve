from __future__ import annotations

import pytest

from fastcoder.gateway.config import GatewaySettings

# Settings aliases that a developer's local .env or shell may define. When present they
# win over explicit init kwargs at validation time (e.g. GatewaySettings(api_key="test-token")
# silently becomes the .env value), producing confusing 401s. Clear them so tests are
# hermetic and behave the same locally as in CI, where no .env exists.
_GATEWAY_ENV_ALIASES = (
    "FASTCODER_API_KEY",
    "VLLM_BASE_URL",
    "VLLM_API_KEY",
    "LOG_LEVEL",
    "FASTCODER_AUTH_BYPASS",
    "FASTCODER_RATE_LIMIT_PER_MINUTE",
    "FASTCODER_REQUEST_TIMEOUT_SECONDS",
)


@pytest.fixture(autouse=True)
def isolate_gateway_settings_from_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(GatewaySettings.model_config, "env_file", None)
    for alias in _GATEWAY_ENV_ALIASES:
        monkeypatch.delenv(alias, raising=False)
