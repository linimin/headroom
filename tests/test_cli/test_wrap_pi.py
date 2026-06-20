"""Tests for `headroom wrap pi`."""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from headroom.providers.pi import PI_SESSION_CONFIG_ENV


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def wrap_modules(monkeypatch: pytest.MonkeyPatch) -> tuple[types.ModuleType, click.Group]:
    headroom_pkg = sys.modules.get("headroom")
    saved_headroom_cli_attr = headroom_pkg.cli if headroom_pkg is not None and hasattr(headroom_pkg, "cli") else None
    saved_modules = {
        name: sys.modules.get(name)
        for name in ("headroom.cli", "headroom.cli.main", "headroom.cli.wrap")
    }

    fake_main_module = types.ModuleType("headroom.cli.main")
    fake_main_module.main = click.Group()
    sys.modules["headroom.cli.main"] = fake_main_module
    sys.modules.pop("headroom.cli", None)
    sys.modules.pop("headroom.cli.wrap", None)

    wrap_cli = importlib.import_module("headroom.cli.wrap")
    monkeypatch.setattr(wrap_cli, "_print_telemetry_notice", lambda: None)

    yield wrap_cli, fake_main_module.main

    sys.modules.pop("headroom.cli.wrap", None)
    sys.modules.pop("headroom.cli", None)
    for name, module in saved_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module
    if headroom_pkg is not None:
        if saved_headroom_cli_attr is None:
            try:
                delattr(headroom_pkg, "cli")
            except AttributeError:
                pass
        else:
            headroom_pkg.cli = saved_headroom_cli_attr


def test_wrap_pi_defaults_to_v1_session_config_and_launches_with_one_extension(
    runner: CliRunner,
    wrap_modules: tuple[types.ModuleType, click.Group],
) -> None:
    wrap_cli, main = wrap_modules
    captured: dict[str, object] = {}

    def fake_run(command: list[str], env: dict[str, str]) -> SimpleNamespace:
        captured["command"] = command
        captured["env"] = dict(env)
        session_path = Path(env[PI_SESSION_CONFIG_ENV])
        extension_path = Path(command[command.index("--extension") + 1])
        captured["session_config"] = json.loads(session_path.read_text(encoding="utf-8"))
        captured["extension_contents"] = extension_path.read_text(encoding="utf-8")
        return SimpleNamespace(returncode=0)

    with (
        patch("headroom.cli.wrap._resolve_pi_binary", return_value="/fake/bin/pi"),
        patch("headroom.cli.wrap.subprocess.run", side_effect=fake_run),
    ):
        result = runner.invoke(main, ["wrap", "pi", "--", "--model", "openai/gpt-5"])

    assert result.exit_code == 0, result.output
    command = captured["command"]
    assert command == [
        "/fake/bin/pi",
        "--model",
        "openai/gpt-5",
        "--extension",
        command[-1],
    ]
    assert command.count("--extension") == 1

    session_config = captured["session_config"]
    assert session_config["managedProviders"] == ["openai", "anthropic", "github-copilot"]
    assert session_config["providers"]["openai"]["routedBaseUrl"] == "http://127.0.0.1:8789/v1"
    assert session_config["providers"]["anthropic"]["routedBaseUrl"] == "http://127.0.0.1:8790"
    assert session_config["providers"]["github-copilot"]["routedBaseUrl"] == "http://127.0.0.1:8788/v1"
    assert "HEADROOM_PI_SESSION_CONFIG" in captured["extension_contents"]


def test_wrap_pi_single_provider_port_override(
    runner: CliRunner,
    wrap_modules: tuple[types.ModuleType, click.Group],
) -> None:
    _wrap_cli, main = wrap_modules
    captured: dict[str, object] = {}

    def fake_run(command: list[str], env: dict[str, str]) -> SimpleNamespace:
        captured["command"] = command
        captured["session_config"] = json.loads(
            Path(env[PI_SESSION_CONFIG_ENV]).read_text(encoding="utf-8")
        )
        return SimpleNamespace(returncode=0)

    with (
        patch("headroom.cli.wrap._resolve_pi_binary", return_value="/fake/bin/pi"),
        patch("headroom.cli.wrap.subprocess.run", side_effect=fake_run),
    ):
        result = runner.invoke(
            main,
            ["wrap", "pi", "--provider", "github-copilot", "--port", "9911", "--", "--continue"],
        )

    assert result.exit_code == 0, result.output
    assert captured["session_config"]["managedProviders"] == ["github-copilot"]
    assert captured["session_config"]["providers"]["github-copilot"]["port"] == 9911
    assert captured["command"][1:] == ["--continue", "--extension", captured["command"][-1]]


def test_wrap_pi_rejects_port_override_for_multiple_providers(
    runner: CliRunner,
    wrap_modules: tuple[types.ModuleType, click.Group],
) -> None:
    _wrap_cli, main = wrap_modules

    with patch("headroom.cli.wrap._resolve_pi_binary", return_value="/fake/bin/pi"):
        result = runner.invoke(
            main,
            ["wrap", "pi", "--provider", "openai", "--provider", "anthropic", "--port", "9911"],
        )

    assert result.exit_code != 0
    assert "'--port' is only valid when exactly one pi provider is managed" in result.output


def test_wrap_pi_rejects_user_supplied_extension_passthrough(
    runner: CliRunner,
    wrap_modules: tuple[types.ModuleType, click.Group],
) -> None:
    _wrap_cli, main = wrap_modules

    with patch("headroom.cli.wrap._resolve_pi_binary", return_value="/fake/bin/pi"):
        result = runner.invoke(main, ["wrap", "pi", "--", "--extension", "/tmp/user.ts"])

    assert result.exit_code != 0
    assert "User-supplied pi '--extension' arguments are rejected during Phase 0" in result.output
