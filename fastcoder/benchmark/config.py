"""Configuration models for benchmark runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class ModelServerConfig(BaseModel):
    """Connection settings for an OpenAI-compatible model server."""

    base_url: str = Field(default="http://localhost:9000/v1")
    model: str = Field(default="fake-coder")
    api_key: str | None = Field(default=None)
    timeout_seconds: float = Field(default=30.0, gt=0)
    quantization: str | None = Field(default=None)
    dtype: str | None = Field(default=None)
    notes: str | None = Field(default=None)

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"


class CostAssumptions(BaseModel):
    """Cost inputs used for rough benchmark economics."""

    accelerator_hourly_cost_usd: float | None = Field(default=None, ge=0)


class HardwareConfig(BaseModel):
    """Expected or measured hardware metadata for a benchmark run."""

    gpu_name: str | None = Field(default=None)
    gpu_memory_gb: float | None = Field(default=None, ge=0)
    provider: str | None = Field(default=None)
    hourly_cost_usd: float | None = Field(default=None, ge=0)


class BenchmarkConfig(BaseModel):
    """Top-level benchmark configuration."""

    experiment_name: str = Field(default="local_smoke")
    model_server: ModelServerConfig = Field(default_factory=ModelServerConfig)
    workloads: list[str] = Field(default_factory=lambda: ["short_chat"])
    concurrency: list[int] = Field(default_factory=lambda: [1])
    requests_per_workload_per_concurrency: int = Field(default=1, ge=1)
    max_tokens: int = Field(default=64, ge=1)
    temperature: float = Field(default=0.0, ge=0)
    stream: bool = Field(default=False)
    output_path: Path = Field(default=Path("results/local_smoke.json"))
    cost: CostAssumptions = Field(default_factory=CostAssumptions)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    expected_endpoints: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("concurrency")
    @classmethod
    def validate_concurrency(cls, values: list[int]) -> list[int]:
        if not values:
            msg = "concurrency must contain at least one level"
            raise ValueError(msg)
        if any(value < 1 for value in values):
            msg = "all concurrency levels must be >= 1"
            raise ValueError(msg)
        return values

    @field_validator("workloads")
    @classmethod
    def validate_workloads(cls, values: list[str]) -> list[str]:
        if not values:
            msg = "workloads must contain at least one workload name"
            raise ValueError(msg)
        return values


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    """Load and validate a benchmark config from YAML."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    if not isinstance(raw_config, dict):
        msg = f"benchmark config must be a mapping: {config_path}"
        raise ValueError(msg)
    return BenchmarkConfig.model_validate(raw_config)


def dump_config_for_json(config: BenchmarkConfig) -> dict[str, Any]:
    """Return a JSON-serializable representation of a benchmark config."""

    return config.model_dump(mode="json")
