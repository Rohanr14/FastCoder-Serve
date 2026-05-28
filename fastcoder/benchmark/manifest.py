"""Run manifest creation for reproducible benchmark executions."""

from __future__ import annotations

import hashlib
import os
import platform as platform_module
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field

from fastcoder import __version__
from fastcoder.benchmark.config import load_benchmark_config

ALLOWED_ENVIRONMENT_KEYS = ("VLLM_BASE_URL", "FASTCODER_API_KEY")
SECRET_ENVIRONMENT_KEYS = {"FASTCODER_API_KEY"}
REDACTED_VALUE = "***REDACTED***"


class GitManifestInfo(BaseModel):
    commit_hash: str | None = None
    dirty: bool | None = None


class RunManifest(BaseModel):
    schema_version: str = "0.1"
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    git: GitManifestInfo
    config_file_path: str
    config_hash_sha256: str
    python_version: str
    package_version: str | None
    platform: dict[str, str]
    environment: dict[str, str]
    notes: str | None = None
    intended_output_result_path: str


def create_run_manifest(
    *,
    config_path: str | Path,
    notes: str | None = None,
    environ: dict[str, str] | None = None,
) -> RunManifest:
    """Create a run manifest for a benchmark config."""

    resolved_config_path = Path(config_path)
    config = load_benchmark_config(resolved_config_path)
    return RunManifest(
        git=collect_git_info(resolved_config_path.parent),
        config_file_path=str(resolved_config_path),
        config_hash_sha256=hash_file(resolved_config_path),
        python_version=sys.version,
        package_version=__version__,
        platform={
            "system": platform_module.system(),
            "release": platform_module.release(),
            "machine": platform_module.machine(),
            "platform": platform_module.platform(),
        },
        environment=capture_environment(environ or dict(os.environ)),
        notes=notes,
        intended_output_result_path=str(config.output_path),
    )


def write_run_manifest(
    *,
    config_path: str | Path,
    output_dir: str | Path = "results/manifests",
    notes: str | None = None,
) -> Path:
    """Create and write a timestamped run manifest JSON file."""

    manifest = create_run_manifest(config_path=config_path, notes=notes)
    config_name = Path(config_path).stem
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = Path(output_dir) / f"{config_name}_{timestamp}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        f"{manifest.model_dump_json(indent=2)}\n",
        encoding="utf-8",
    )
    return destination


def hash_file(path: str | Path) -> str:
    """Return a stable SHA256 hash for a file's exact bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_environment(environ: dict[str, str]) -> dict[str, str]:
    """Capture allowed environment keys while redacting secrets."""

    captured: dict[str, str] = {}
    for key in ALLOWED_ENVIRONMENT_KEYS:
        if key not in environ:
            continue
        captured[key] = REDACTED_VALUE if key in SECRET_ENVIRONMENT_KEYS else environ[key]
    return captured


def collect_git_info(cwd: str | Path) -> GitManifestInfo:
    """Best-effort git metadata collection."""

    commit_hash = _git(["rev-parse", "HEAD"], cwd)
    status = _git(["status", "--porcelain"], cwd)
    dirty = None if status is None else bool(status.strip())
    return GitManifestInfo(commit_hash=commit_hash, dirty=dirty)


def _git(args: list[str], cwd: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()
