from __future__ import annotations

import importlib

import pytest

from fastcoder import cli


def test_no_args_prints_usage_and_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 0
    assert "usage: fastcoder" in capsys.readouterr().out


def test_unknown_command_returns_two(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["definitely-not-a-command"]) == 2
    assert "unknown command" in capsys.readouterr().err


def test_every_subcommand_target_is_importable_and_callable() -> None:
    # Guards _COMMANDS against typos: each dispatch target must exist and be callable.
    for module_name, func_name in cli._COMMANDS.values():
        module = importlib.import_module(module_name)
        assert callable(getattr(module, func_name))
