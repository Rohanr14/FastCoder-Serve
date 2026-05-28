"""Configuration models for benchmark runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


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


class SoftwareConfig(BaseModel):
    """Serving software metadata captured with a benchmark config."""

    vllm_version: str | None = Field(default=None)
    serving_backend: str | None = Field(default=None)
    backend_commit: str | None = Field(default=None)


class SpeculationConfig(BaseModel):
    """Speculative-decoding metadata for a benchmark config."""

    method: str | None = Field(default=None)
    draft_model: str | None = Field(default=None)
    num_speculative_tokens: int | None = Field(default=None, ge=1)
    acceptance_rate: float | None = Field(default=None, ge=0, le=1)
    notes: str | None = Field(default=None)


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
    software: SoftwareConfig = Field(default_factory=SoftwareConfig)
    speculation: SpeculationConfig = Field(default_factory=SpeculationConfig)
    vllm_version: str | None = Field(default=None)
    serving_backend: str | None = Field(default=None)
    backend_commit: str | None = Field(default=None)
    speculative_method: str | None = Field(default=None)
    speculative_model: str | None = Field(default=None)
    speculative_notes: str | None = Field(default=None)
    measurement_hypothesis: str | None = Field(default=None)
    expected_endpoints: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def sync_metadata_fields(self) -> BenchmarkConfig:
        """Keep concise config fields and nested result-schema metadata in sync."""

        self.vllm_version = self.vllm_version or self.software.vllm_version
        self.serving_backend = self.serving_backend or self.software.serving_backend
        self.backend_commit = self.backend_commit or self.software.backend_commit
        self.speculative_method = self.speculative_method or self.speculation.method
        self.speculative_model = self.speculative_model or self.speculation.draft_model
        self.speculative_notes = self.speculative_notes or self.speculation.notes

        self.software = self.software.model_copy(
            update={
                "vllm_version": self.vllm_version,
                "serving_backend": self.serving_backend,
                "backend_commit": self.backend_commit,
            }
        )
        self.speculation = self.speculation.model_copy(
            update={
                "method": self.speculative_method,
                "draft_model": self.speculative_model,
                "notes": self.speculative_notes,
            }
        )
        return self

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
