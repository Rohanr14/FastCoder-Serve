"""Gateway runtime settings."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GatewaySettings(BaseSettings):
    """Environment-driven gateway settings."""

    api_key: str = Field(default="dev-token", validation_alias="FASTCODER_API_KEY")
    upstream_base_url: str = Field(
        default="http://localhost:8000/v1",
        validation_alias="VLLM_BASE_URL",
    )
    upstream_api_key: str | None = Field(default=None, validation_alias="VLLM_API_KEY")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    auth_bypass: bool = Field(default=False, validation_alias="FASTCODER_AUTH_BYPASS")
    rate_limit_per_minute: int = Field(
        default=120,
        ge=1,
        validation_alias="FASTCODER_RATE_LIMIT_PER_MINUTE",
    )
    request_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        validation_alias="FASTCODER_REQUEST_TIMEOUT_SECONDS",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @property
    def upstream_chat_completions_url(self) -> str:
        return f"{self.upstream_base_url.rstrip('/')}/chat/completions"
