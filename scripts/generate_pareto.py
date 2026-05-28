"""CLI entrypoint for placeholder Pareto plot generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastcoder.plotting.pareto import generate_pareto_plot  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a FastCoder Pareto placeholder plot.")
    parser.add_argument(
        "results",
        nargs="*",
        default=["results/local_smoke.json"],
        help="One or more benchmark result JSON files.",
    )
    parser.add_argument("--output", default="results/pareto.png", help="Output PNG path.")
    args = parser.parse_args()

    output_path = generate_pareto_plot(args.results, args.output)
    print(f"Wrote Pareto plot to {output_path}")


if __name__ == "__main__":
    main()
