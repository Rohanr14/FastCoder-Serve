"""Optional GPU metric hooks for future H100 runs.

The local smoke path must stay CPU-safe. This module therefore treats GPU telemetry as best-effort
metadata: if NVIDIA tooling is absent, all fields remain ``None`` and callers continue normally.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GPUSnapshot:
    gpu_name: str | None = None
    gpu_memory_gb: float | None = None
    peak_memory_gb: float | None = None


def collect_gpu_snapshot() -> GPUSnapshot:
    """Collect a best-effort GPU snapshot without requiring NVIDIA dependencies."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return GPUSnapshot()

    if result.returncode != 0:
        return GPUSnapshot()

    first_line = result.stdout.strip().splitlines()[0:1]
    if not first_line:
        return GPUSnapshot()

    parts = [part.strip() for part in first_line[0].split(",")]
    if len(parts) < 3:
        return GPUSnapshot()

    gpu_name = parts[0] or None
    total_memory_gb = _mib_to_gib(parts[1])
    used_memory_gb = _mib_to_gib(parts[2])
    return GPUSnapshot(
        gpu_name=gpu_name,
        gpu_memory_gb=total_memory_gb,
        peak_memory_gb=used_memory_gb,
    )


def _mib_to_gib(value: str) -> float | None:
    try:
        return round(float(value) / 1024.0, 3)
    except ValueError:
        return None
