"""Tests for `headroom.providers.pi`."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import click
import pytest

from headroom.providers import pi as pi_mod


def test_resolve_managed_providers_defaults_to_v1_set() -> None:
    assert pi_mod.resolve_managed_providers(()) == ["openai", "anthropic", "github-copilot"]


def test_resolve_managed_providers_dedupes_preserving_order() -> None:
    providers = pi_mod.resolve_managed_providers(("github-copilot", "openai", "openai"))
    assert providers == ["github-copilot", "openai"]


@pytest.mark.parametrize("provider", ["all", "unknown-provider"])
def test_resolve_managed_providers_rejects_invalid_values(provider: str) -> None:
    with pytest.raises(click.ClickException):
        pi_mod.resolve_managed_providers((provider,))


def test_resolve_provider_ports_uses_defaults() -> None:
    assert pi_mod.resolve_provider_ports(("openai", "anthropic"), None) == {
        "openai": 8789,
        "anthropic": 8790,
    }


def test_resolve_provider_ports_allows_single_provider_override() -> None:
    assert pi_mod.resolve_provider_ports(("github-copilot",), 9900) == {"github-copilot": 9900}


def test_resolve_provider_ports_rejects_multi_provider_override() -> None:
    with pytest.raises(click.ClickException):
        pi_mod.resolve_provider_ports(("openai", "anthropic"), 9900)


def test_build_pi_wrap_session_config_renders_routed_urls() -> None:
    config = pi_mod.build_pi_wrap_session_config(
        ["openai", "anthropic", "github-copilot"],
        {"openai": 8789, "anthropic": 8790, "github-copilot": 8788},
        phase0={"forceNativeProviders": ["openai"]},
    )

    assert config["version"] == 1
    assert config["managedProviders"] == ["openai", "anthropic", "github-copilot"]
    assert config["providers"]["openai"]["routedBaseUrl"] == "http://127.0.0.1:8789/v1"
    assert config["providers"]["anthropic"]["routedBaseUrl"] == "http://127.0.0.1:8790"
    assert config["providers"]["github-copilot"]["routedBaseUrl"] == "http://127.0.0.1:8788/v1"
    assert config["phase0"]["forceNativeProviders"] == ["openai"]


def test_write_session_config_and_render_extension(tmp_path: Path) -> None:
    config_path = pi_mod.write_pi_session_config(
        tmp_path,
        pi_mod.build_pi_wrap_session_config(["openai"], {"openai": 8789}),
    )
    extension_path = pi_mod.render_pi_extension(tmp_path, config_path)

    assert config_path.name == "session.json"
    assert extension_path.name == "extension.ts"
    assert json.loads(config_path.read_text(encoding="utf-8"))["providers"]["openai"][
        "routedBaseUrl"
    ] == "http://127.0.0.1:8789/v1"
    assert "HEADROOM_PI_SESSION_CONFIG" in extension_path.read_text(encoding="utf-8")


def test_build_pi_launch_env_sets_session_config_and_verbose(tmp_path: Path) -> None:
    config_path = tmp_path / "session.json"
    config_path.write_text("{}\n", encoding="utf-8")

    env = pi_mod.build_pi_launch_env({"EXISTING": "1", pi_mod.PI_VERBOSE_ENV: "0"}, config_path, verbose=True)
    assert env["EXISTING"] == "1"
    assert env[pi_mod.PI_SESSION_CONFIG_ENV] == str(config_path)
    assert env[pi_mod.PI_VERBOSE_ENV] == "1"

    quiet_env = pi_mod.build_pi_launch_env(env, config_path, verbose=False)
    assert quiet_env[pi_mod.PI_SESSION_CONFIG_ENV] == str(config_path)
    assert pi_mod.PI_VERBOSE_ENV not in quiet_env


def test_build_pi_launch_args_appends_exactly_one_extension(tmp_path: Path) -> None:
    extension_path = tmp_path / "extension.ts"
    args = pi_mod.build_pi_launch_args(("--model", "openai/gpt-5"), extension_path)

    assert args == ("--model", "openai/gpt-5", "--extension", str(extension_path))


@pytest.mark.parametrize(
    "pi_args",
    [
        ("--extension", "/tmp/user.ts"),
        ("--extension=/tmp/user.ts",),
        ("-e", "/tmp/user.ts"),
        ("-e/tmp/user.ts",),
    ],
)
def test_build_pi_launch_args_rejects_user_extension_flags(
    tmp_path: Path, pi_args: tuple[str, ...]
) -> None:
    with pytest.raises(click.ClickException):
        pi_mod.build_pi_launch_args(pi_args, tmp_path / "extension.ts")


@pytest.mark.skipif(
    shutil.which("pi") is None or shutil.which("node") is None,
    reason="pi and node are required for the live Phase 0 feasibility probe",
)
def test_run_phase0_feasibility_probe_hits_override_then_native_fallback() -> None:
    result = pi_mod.run_phase0_feasibility_probe()

    assert result["firstResponse"] == "override response"
    assert result["secondResponse"] == "native response"
    assert [request["url"] for request in result["requests"]["override"]] == ["/v1/chat/completions"]
    assert [request["url"] for request in result["requests"]["native"]] == ["/v1/chat/completions"]

    event_types = [event["type"] for event in result["events"]]
    assert "provider_observed" in event_types
    assert "model_select" in event_types
    assert "provider_registered" in event_types
    assert "provider_unregistered" in event_types

    model_selects = [event for event in result["events"] if event["type"] == "model_select"]
    assert [event["currentModel"]["provider"] for event in model_selects] == ["anthropic", "openai"]

    registered = [event for event in result["events"] if event["type"] == "provider_registered"]
    assert registered[0]["providerId"] == "openai"
    assert registered[0]["baseUrl"] == f"http://127.0.0.1:{result['overridePort']}/v1"

    unregistered = [event for event in result["events"] if event["type"] == "provider_unregistered"]
    assert any(event["reason"] == "forced-native" for event in unregistered)
