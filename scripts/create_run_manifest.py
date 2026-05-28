"""Create a reproducibility manifest for a benchmark run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastcoder.benchmark.manifest import write_run_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a FastCoder benchmark run manifest.")
    parser.add_argument("--config", required=True, help="Path to benchmark YAML config.")
    parser.add_argument("--output-dir", default="results/manifests")
    parser.add_argument("--notes", default=None)
    args = parser.parse_args()

    try:
        path = write_run_manifest(
            config_path=args.config,
            output_dir=args.output_dir,
            notes=args.notes,
        )
    except (OSError, ValueError) as exc:
        print(f"Failed to create manifest: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote run manifest to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
