"""Unified ``fastcoder`` command-line entry point.

Installs as a console script (see ``[project.scripts]`` in ``pyproject.toml``) so FastCoder-Serve is
usable as a tool: point it at any OpenAI-compatible endpoint and get a schema-validated Pareto + SLO
report. Each subcommand dispatches to the existing, tested script entrypoint.
"""

from __future__ import annotations

import importlib
import sys

_COMMANDS: dict[str, tuple[str, str]] = {
    "endpoint": ("scripts.check_endpoint", "main"),
    "config": ("scripts.validate_config", "main"),
    "bench": ("scripts.run_bench", "main"),
    "validate": ("scripts.validate_results", "main"),
    "slo": ("scripts.slo_analysis", "main"),
    "pareto": ("scripts.generate_pareto", "main"),
}


def _usage() -> str:
    return "usage: fastcoder {" + "|".join(_COMMANDS) + "} [args...]"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(_usage())
        return 0
    command = args[0]
    if command not in _COMMANDS:
        print(f"unknown command {command!r}\n{_usage()}", file=sys.stderr)
        return 2

    module_name, func_name = _COMMANDS[command]
    func = getattr(importlib.import_module(module_name), func_name)
    # Hand the subcommand its own argv so its argparse sees the right tokens.
    sys.argv = [f"fastcoder {command}", *args[1:]]
    result = func()
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
