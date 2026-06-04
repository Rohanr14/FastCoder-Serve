"""Plot an open-loop result JSON: the latency-vs-offered-load knee and the SLO-violation curve."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _numeric(value: Any) -> float:
    return float(value) if isinstance(value, int | float) else math.nan


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot an open-loop (arrival-rate) result JSON.")
    parser.add_argument("result", help="path to results/openloop_*.json")
    parser.add_argument("--output", default=None, help="PNG path (default: alongside the result).")
    args = parser.parse_args()

    data = json.loads(Path(args.result).read_text(encoding="utf-8"))
    points = data.get("arrival_rate_points", [])
    if not points:
        print("no arrival_rate_points in result", file=sys.stderr)
        return 1

    rates = [_numeric(p.get("target_rate_rps")) for p in points]
    slo_ttft_ms = _numeric(data.get("slo_ttft_seconds")) * 1000.0

    fig, (ax_lat, ax_slo) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    for key, label in (
        ("p50_latency_seconds", "p50"),
        ("p95_latency_seconds", "p95"),
        ("p99_latency_seconds", "p99"),
    ):
        ax_lat.plot(rates, [_numeric(p.get(key)) for p in points], marker="o", label=label)
    ax_lat.set_ylabel("end-to-end latency (s)")
    ax_lat.set_title(f"{data.get('experiment_name', 'open-loop')}: latency vs offered load")
    ax_lat.grid(True, alpha=0.3)
    ax_lat.legend(title="percentile")

    violations = [_numeric(p.get("slo_violation_rate")) * 100.0 for p in points]
    ax_slo.plot(rates, violations, marker="s", color="crimson")
    ax_slo.set_ylabel("SLO violation (%)")
    ax_slo.set_xlabel("offered arrival rate (requests / s)")
    ax_slo.set_title(f"SLO violation rate (p99 TTFT <= {slo_ttft_ms:.0f} ms)")
    ax_slo.set_ylim(-2, 102)
    ax_slo.grid(True, alpha=0.3)

    fig.tight_layout()
    output = Path(args.output) if args.output else Path(args.result).with_suffix(".png")
    fig.savefig(output, dpi=120)
    plt.close(fig)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
