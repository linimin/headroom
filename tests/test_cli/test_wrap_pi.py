"""Tests for ``headroom wrap pi``."""

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


class _FakePiProcess:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.wait_calls: list[int | None] = []
        self.signal_calls: list[int] = []
        self.pid = 4242
        self._done = False

    def wait(self, timeout: int | None = None) -> int:
        self.wait_calls.append(timeout)
        self._done = True
        return self.returncode

    def poll(self) -> int | None:
        return self.returncode if self._done else None

    def send_signal(self, signum: int) -> None:
        self.signal_calls.append(signum)
        self._done = True


class _FakeManagedProcess:
    _next_pid = 4343

    def __init__(self) -> None:
        self.wait_calls: list[int | None] = []
        self.signal_calls: list[int] = []
        self.pid = _FakeManagedProcess._next_pid
        _FakeManagedProcess._next_pid += 1
        self._done = False

    def wait(self, timeout: int | None = None) -> int:
        self.wait_calls.append(timeout)
        self._done = True
        return 0

    def poll(self) -> int | None:
        return 0 if self._done else None

    def send_signal(self, signum: int) -> None:
        self.signal_calls.append(signum)
        self._done = True


def test_wrap_pi_defaults_to_v1_session_config_and_launches_with_one_extension(
    runner: CliRunner,
    wrap_modules: tuple[types.ModuleType, click.Group],
) -> None:
    wrap_cli, main = wrap_modules
    captured: dict[str, object] = {}
    fake_proc = _FakePiProcess()

    def fake_start_pi_managed_proxies(*args, **kwargs):
        return [wrap_cli._PiManagedProxy("openai", 8789, "owned")]

    def fake_popen(command: list[str], env: dict[str, str], start_new_session: bool):
        captured["command"] = command
        captured["env"] = env
        captured["start_new_session"] = start_new_session
        captured["session_config"] = json.loads(
            Path(env[PI_SESSION_CONFIG_ENV]).read_text(encoding="utf-8")
        )
        extension_path = Path(command[command.index("--extension") + 1])
        captured["extension_contents"] = extension_path.read_text(encoding="utf-8")
        return fake_proc

    with (
        patch("headroom.cli.wrap._resolve_pi_binary", return_value="/fake/bin/pi"),
        patch("headroom.cli.wrap._start_pi_managed_proxies", side_effect=fake_start_pi_managed_proxies),
        patch("headroom.cli.wrap.subprocess.Popen", side_effect=fake_popen),
    ):
        result = runner.invoke(main, ["wrap", "pi", "--provider", "openai"])

    assert result.exit_code == 0, result.output
    assert captured["command"][0] == "/fake/bin/pi"
    assert captured["command"].count("--extension") == 1
    assert captured["session_config"]["managedProviders"] == ["openai"]
    assert captured["session_config"]["providers"]["openai"]["ownership"] == "owned"
    assert "HEADROOM_PI_SESSION_CONFIG" in captured["extension_contents"]


def test_wrap_pi_forwards_backend_and_memory_to_proxy_lifecycle(
    runner: CliRunner,
    wrap_modules: tuple[types.ModuleType, click.Group],
) -> None:
    wrap_cli, main = wrap_modules
    captured: dict[str, object] = {}

    def fake_start_pi_managed_proxies(managed_providers, provider_ports, **kwargs):
        captured["managed_providers"] = managed_providers
        captured["provider_ports"] = provider_ports
        captured["kwargs"] = kwargs
        return [wrap_cli._PiManagedProxy("github-copilot", 9911, "attached")]

    with (
        patch("headroom.cli.wrap._resolve_pi_binary", return_value="/fake/bin/pi"),
        patch("headroom.cli.wrap._start_pi_managed_proxies", side_effect=fake_start_pi_managed_proxies),
        patch("headroom.cli.wrap.subprocess.Popen", return_value=_FakePiProcess()),
    ):
        result = runner.invoke(
            main,
            [
                "wrap",
                "pi",
                "--provider",
                "github-copilot",
                "--port",
                "9911",
                "--backend",
                "bedrock",
                "--memory",
                "--",
                "--continue",
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured["managed_providers"] == ["github-copilot"]
    assert captured["provider_ports"] == {"github-copilot": 9911}
    assert captured["kwargs"] == {"backend": "bedrock", "memory": True, "verbose": False}


def test_wrap_pi_rejects_port_override_for_multiple_providers(
    runner: CliRunner,
    wrap_modules: tuple[types.ModuleType, click.Group],
) -> None:
    _wrap_cli, main = wrap_modules

    with patch("headroom.cli.wrap._resolve_pi_binary", return_value="/fake/bin/pi"):
        result = runner.invoke(
            main,
            [
                "wrap",
                "pi",
                "--provider",
                "openai",
                "--provider",
                "anthropic",
                "--port",
                "9911",
            ],
        )

    assert result.exit_code != 0
    assert "exactly one pi provider is managed" in result.output


def test_wrap_pi_rejects_user_supplied_extension_passthrough_before_proxy_start(
    runner: CliRunner,
    wrap_modules: tuple[types.ModuleType, click.Group],
) -> None:
    _wrap_cli, main = wrap_modules

    with (
        patch("headroom.cli.wrap._resolve_pi_binary", return_value="/fake/bin/pi"),
        patch("headroom.cli.wrap._start_pi_managed_proxies") as start_proxies,
    ):
        result = runner.invoke(main, ["wrap", "pi", "--", "--extension", "/tmp/user.ts"])

    assert result.exit_code != 0
    assert "User-supplied pi '--extension' arguments are rejected by `headroom wrap pi` v1" in result.output
    start_proxies.assert_not_called()


def test_wrap_pi_help_describes_supported_v1_contract(
    runner: CliRunner,
    wrap_modules: tuple[types.ModuleType, click.Group],
) -> None:
    _wrap_cli, main = wrap_modules

    result = runner.invoke(main, ["wrap", "pi", "--help"])

    assert result.exit_code == 0, result.output
    assert "temporary Headroom extension" in result.output
    assert "openai" in result.output
    assert "anthropic" in result.output
    assert "github-copilot" in result.output
    assert "--port" in result.output
    assert "exactly one provider is managed" in result.output
    assert "user-supplied pi" in result.output
    assert "Defaults" in result.output
    assert "all three" in result.output


def test_start_or_attach_pi_proxy_accepts_compatible_attach(
    wrap_modules: tuple[types.ModuleType, click.Group],
) -> None:
    wrap_cli, _main = wrap_modules
    probe = SimpleNamespace(status="compatible", reason="ok", metadata=SimpleNamespace(
        headroom_version="1.0.0", backend="anthropic", upstream_family="openai", memory=False
    ))

    with (
        patch("headroom.cli.wrap._probe_pi_attach_compatibility", return_value=probe),
        patch("headroom.cli.wrap._port_bind_error", return_value=OSError("busy")),
    ):
        state = wrap_cli._start_or_attach_pi_proxy(
            "openai",
            8789,
            backend="anthropic",
            memory=False,
            verbose=False,
        )

    assert state.ownership == "attached"
    assert state.process is None


def test_start_or_attach_pi_proxy_rejects_incompatible_attach(
    wrap_modules: tuple[types.ModuleType, click.Group],
) -> None:
    wrap_cli, _main = wrap_modules
    probe = SimpleNamespace(status="incompatible", reason="wrong family", metadata=None)

    with (
        patch("headroom.cli.wrap._probe_pi_attach_compatibility", return_value=probe),
        patch("headroom.cli.wrap._port_bind_error", return_value=OSError("busy")),
    ):
        with pytest.raises(click.ClickException):
            wrap_cli._start_or_attach_pi_proxy(
                "anthropic",
                8790,
                backend="anthropic",
                memory=False,
                verbose=False,
            )


def test_cleanup_pi_wrap_session_stops_pi_then_only_owned_proxies(
    wrap_modules: tuple[types.ModuleType, click.Group],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrap_cli, _main = wrap_modules
    pi_proc = _FakePiProcess()
    owned_proc = _FakeManagedProcess()
    attached_proc = _FakeManagedProcess()
    calls: list[tuple[int, int]] = []

    def fake_killpg(pid: int, signum: int) -> None:
        calls.append((pid, signum))
        if pid == pi_proc.pid:
            pi_proc._done = True
        if pid == owned_proc.pid:
            owned_proc._done = True
        if pid == attached_proc.pid:
            attached_proc._done = True

    monkeypatch.setattr(wrap_cli.os, "killpg", fake_killpg)
    wrap_cli._cleanup_pi_wrap_session(
        pi_proc,
        [
            wrap_cli._PiManagedProxy("openai", 8789, "owned", owned_proc),
            wrap_cli._PiManagedProxy("anthropic", 8790, "attached", attached_proc),
        ],
        forwarded_signal=wrap_cli.signal.SIGINT,
    )

    assert calls[0] == (pi_proc.pid, wrap_cli.signal.SIGINT)
    assert (owned_proc.pid, wrap_cli.signal.SIGTERM) in calls
    assert all(pid != attached_proc.pid for pid, _signum in calls)


def test_start_pi_managed_proxies_cleans_up_owned_proxies_after_partial_failure(
    wrap_modules: tuple[types.ModuleType, click.Group],
) -> None:
    wrap_cli, _main = wrap_modules
    first_proxy = wrap_cli._PiManagedProxy("openai", 8789, "owned", _FakeManagedProcess())
    cleanup_calls: list[list[str]] = []

    def fake_start_or_attach(provider_id: str, *args, **kwargs):
        if provider_id == "openai":
            return first_proxy
        raise click.ClickException("metadata mismatch")

    def fake_cleanup(pi_process, proxies, **kwargs):
        cleanup_calls.append([proxy.provider_id for proxy in proxies])

    with (
        patch("headroom.cli.wrap._start_or_attach_pi_proxy", side_effect=fake_start_or_attach),
        patch("headroom.cli.wrap._cleanup_pi_wrap_session", side_effect=fake_cleanup),
    ):
        with pytest.raises(click.ClickException):
            wrap_cli._start_pi_managed_proxies(
                ["openai", "anthropic"],
                {"openai": 8789, "anthropic": 8790},
                backend="anthropic",
                memory=False,
                verbose=False,
            )

    assert cleanup_calls == [["openai"]]
