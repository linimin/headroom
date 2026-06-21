"""Tests for ``headroom.providers.pi``."""

from __future__ import annotations

import importlib.resources as importlib_resources
import json
import os
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import click
import pytest

from headroom._version import __version__
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
        provider_variant_ports={"github-copilot": {"openai": 8788, "anthropic": 8791}},
        provider_variant_backends={
            "github-copilot": {"openai": "openai", "anthropic": "anthropic"}
        },
    )
    assert config["version"] == 1
    assert config["managedProviders"] == ["openai", "anthropic", "github-copilot"]
    assert config["providers"]["openai"]["routedBaseUrl"] == "http://127.0.0.1:8789/v1"
    assert config["providers"]["anthropic"]["routedBaseUrl"] == "http://127.0.0.1:8790"
    assert config["providers"]["github-copilot"]["routedBaseUrl"] == "http://127.0.0.1:8788/v1"
    assert config["providers"]["github-copilot"]["backend"] == "openai"
    assert config["providers"]["github-copilot"]["variants"]["openai"]["routedBaseUrl"] == "http://127.0.0.1:8788/v1"
    assert config["providers"]["github-copilot"]["variants"]["anthropic"]["routedBaseUrl"] == "http://127.0.0.1:8791"
    assert config["providers"]["github-copilot"]["variants"]["anthropic"]["backend"] == "anthropic"
    assert config["phase0"] == {"forceNativeProviders": ["openai"]}


def test_build_pi_provider_payload_switches_copilot_claude_to_root_url() -> None:
    payload = pi_mod.build_pi_provider_payload(
        "github-copilot",
        8788,
        family="anthropic",
    )
    assert payload["rootUrl"] == "http://127.0.0.1:8788"
    assert payload["routedBaseUrl"] == "http://127.0.0.1:8788"
    assert payload["family"] == "anthropic"
    assert payload["backend"] == "anthropic"


def test_build_pi_copilot_provider_payload_contains_both_wire_variants() -> None:
    payload = pi_mod.build_pi_copilot_provider_payload(8788, 8791)
    assert payload["defaultVariant"] == "openai"
    assert payload["variants"]["openai"]["port"] == 8788
    assert payload["variants"]["openai"]["backend"] == "openai"
    assert payload["variants"]["anthropic"]["port"] == 8791
    assert payload["variants"]["anthropic"]["backend"] == "anthropic"
    assert payload["variants"]["anthropic"]["routedBaseUrl"] == "http://127.0.0.1:8791"


def test_build_pi_launch_env_sets_session_config_and_optional_verbose(tmp_path: Path) -> None:
    session_path = tmp_path / "session.json"
    quiet_env = pi_mod.build_pi_launch_env({}, session_path, verbose=False)
    loud_env = pi_mod.build_pi_launch_env({}, session_path, verbose=True)
    assert quiet_env[pi_mod.PI_SESSION_CONFIG_ENV] == str(session_path)
    assert pi_mod.PI_VERBOSE_ENV not in quiet_env
    assert loud_env[pi_mod.PI_VERBOSE_ENV] == "1"


def test_resolve_pi_provider_backend_uses_provider_specific_defaults() -> None:
    assert pi_mod.resolve_pi_provider_backend("openai", None) == "openai"
    assert pi_mod.resolve_pi_provider_backend("anthropic", None) == "anthropic"
    assert pi_mod.resolve_pi_provider_backend("github-copilot", None) == "openai"


def test_resolve_pi_provider_backend_switches_copilot_claude_models_to_anthropic() -> None:
    assert pi_mod.resolve_pi_provider_backend(
        "github-copilot",
        None,
        model_api="anthropic-messages",
        model_id="claude-opus-4.6",
    ) == "anthropic"
    assert pi_mod.resolve_pi_provider_family(
        "github-copilot",
        model_api="anthropic-messages",
        model_id="claude-opus-4.6",
    ) == "anthropic"


def test_resolve_pi_provider_backend_respects_explicit_override() -> None:
    assert pi_mod.resolve_pi_provider_backend("github-copilot", "bedrock") == "bedrock"


def test_load_pi_extension_template_reads_packaged_asset() -> None:
    packaged_template = (
        importlib_resources.files("headroom.providers")
        .joinpath(pi_mod.PI_EXTENSION_TEMPLATE_ASSET)
        .read_text(encoding="utf-8")
    )

    assert pi_mod.load_pi_extension_template() == packaged_template
    assert "HEADROOM_PI_SESSION_CONFIG" in packaged_template
    assert "githubCopilotOAuthProvider" in packaged_template
    assert "setStatus" in packaged_template
    assert "Headroom:" in packaged_template
    assert "/stats?cached=1" in packaged_template
    assert "tokensSaved" in packaged_template
    assert "headroom-status" in packaged_template
    assert "Dashboard:" in packaged_template
    assert "Variant:" in packaged_template
    assert "Headroom took over" in packaged_template
    assert "Headroom reconnecting" in packaged_template
    assert "notifyUiSoon" in packaged_template
    assert "yieldToUi" in packaged_template
    assert "waitForRecoveredHealth" in packaged_template
    assert 'managedConfig.ownership === "owned"' in packaged_template
    assert 'const tookOver = previousOwnership === "attached" && targetConfig.ownership === "owned"' in packaged_template
    assert "const routeChanged =" in packaged_template
    assert 'healthState.status === "unavailable" && !routeChanged' in packaged_template
    assert "anthropic-messages" in packaged_template
    assert "rootUrl" in packaged_template
    assert "variants?.anthropic" in packaged_template


def test_render_pi_extension_copies_packaged_template(tmp_path: Path) -> None:
    extension_path = pi_mod.render_pi_extension(tmp_path, tmp_path / "session.json")

    assert extension_path == tmp_path / "extension.ts"
    assert extension_path.read_text(encoding="utf-8") == pi_mod.load_pi_extension_template()


def _run_installed_pi_packaging_probe(tmp_path: Path) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    dist_dir = tmp_path / "dist"
    site_dir = tmp_path / "site"
    outside_dir = tmp_path / "outside"
    dist_dir.mkdir()
    site_dir.mkdir()
    outside_dir.mkdir()

    build_result = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist_dir)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert build_result.returncode == 0, build_result.stderr or build_result.stdout

    wheel_path = next(dist_dir.glob("*.whl"))
    with zipfile.ZipFile(wheel_path) as wheel_zip:
        assert "headroom/providers/pi_extension_template.ts" in set(wheel_zip.namelist())
        wheel_zip.extractall(site_dir)

    probe_script = textwrap.dedent(
        """
        import importlib.util
        import json
        import os
        import sys
        import tempfile
        import types
        from pathlib import Path

        repo_root = Path(os.environ["HEADROOM_PI_PACKAGING_REPO_ROOT"]).resolve()
        site_dir = Path(os.environ["HEADROOM_PI_PACKAGING_SITE_DIR"]).resolve()
        outside_dir = Path(os.environ["HEADROOM_PI_PACKAGING_OUTSIDE_DIR"]).resolve()

        def _inside_repo(raw_path: str) -> bool:
            try:
                Path(raw_path).resolve().relative_to(repo_root)
            except ValueError:
                return False
            return True

        def _stub_package(name: str, package_dir: Path) -> None:
            spec = importlib.util.spec_from_file_location(
                name,
                package_dir / "__init__.py",
                submodule_search_locations=[str(package_dir)],
            )
            assert spec is not None
            module = types.ModuleType(name)
            module.__file__ = str(package_dir / "__init__.py")
            module.__path__ = [str(package_dir)]
            module.__package__ = name
            module.__spec__ = spec
            sys.modules[name] = module

        def _load_module(name: str, module_path: Path):
            spec = importlib.util.spec_from_file_location(name, module_path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        os.chdir(outside_dir)
        sys.path = [str(site_dir)] + [path for path in sys.path if path and not _inside_repo(path)]
        _stub_package("headroom", site_dir / "headroom")
        _stub_package("headroom.providers", site_dir / "headroom" / "providers")
        _load_module("headroom._version", site_dir / "headroom" / "_version.py")
        pi_module = _load_module("headroom.providers.pi", site_dir / "headroom" / "providers" / "pi.py")

        with tempfile.TemporaryDirectory(dir=outside_dir) as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            session_config_path = temp_dir / "session.json"
            session_config_path.write_text("{}\\n", encoding="utf-8")
            extension_path = pi_module.render_pi_extension(temp_dir, session_config_path)
            print(
                json.dumps(
                    {
                        "cwd": str(Path.cwd()),
                        "module": str(Path(pi_module.__file__).resolve()),
                        "template_len": len(pi_module.load_pi_extension_template()),
                        "rendered": str(extension_path.resolve()),
                        "contains_env": "HEADROOM_PI_SESSION_CONFIG"
                        in extension_path.read_text(encoding="utf-8"),
                    }
                )
            )
        """
    )
    env = os.environ.copy()
    env["HEADROOM_PI_PACKAGING_REPO_ROOT"] = str(repo_root)
    env["HEADROOM_PI_PACKAGING_SITE_DIR"] = str(site_dir)
    env["HEADROOM_PI_PACKAGING_OUTSIDE_DIR"] = str(outside_dir)
    env["PYTHONPATH"] = str(site_dir)
    probe_result = subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--python",
            f"{sys.version_info.major}.{sys.version_info.minor}",
            "--with",
            "click>=8.1.0",
            "python",
            "-c",
            probe_script,
        ],
        cwd=outside_dir,
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=120,
    )
    assert probe_result.returncode == 0, probe_result.stderr or probe_result.stdout
    return json.loads(probe_result.stdout)


def test_installed_pi_packaging_probe_renders_extension_from_built_wheel(tmp_path: Path) -> None:
    result = _run_installed_pi_packaging_probe(tmp_path)

    assert result["cwd"].endswith("/outside")
    assert result["module"].endswith("/site/headroom/providers/pi.py")
    assert result["template_len"] > 0
    assert result["rendered"].endswith("/extension.ts")
    assert result["contains_env"] is True


def test_build_pi_launch_args_appends_headroom_extension(tmp_path: Path) -> None:
    extension_path = tmp_path / "extension.ts"

    assert pi_mod.build_pi_launch_args(("--continue",), extension_path) == (
        "--continue",
        "--extension",
        str(extension_path),
    )


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


def test_build_pi_proxy_metadata_env_uses_provider_family_and_copilot_upstream() -> None:
    env = pi_mod.build_pi_proxy_metadata_env("github-copilot")
    assert env == {
        pi_mod.PI_PROXY_METADATA_FAMILY_ENV: "openai",
        pi_mod.PI_PROXY_METADATA_CAPABILITY_ENV: "1",
        pi_mod.PI_OPENAI_TARGET_API_URL_ENV: "https://api.githubcopilot.com",
        pi_mod.PI_GITHUB_COPILOT_USE_TOKEN_EXCHANGE_ENV: "0",
        pi_mod.PI_LITELLM_SUPPRESS_DEBUG_INFO_ENV: "True",
    }


def test_build_pi_proxy_metadata_env_switches_copilot_claude_to_anthropic_target() -> None:
    env = pi_mod.build_pi_proxy_metadata_env(
        "github-copilot",
        backend="anthropic",
        family="anthropic",
    )
    assert env == {
        pi_mod.PI_PROXY_METADATA_FAMILY_ENV: "anthropic",
        pi_mod.PI_PROXY_METADATA_CAPABILITY_ENV: "1",
        "ANTHROPIC_TARGET_API_URL": "https://api.githubcopilot.com",
        pi_mod.PI_GITHUB_COPILOT_USE_TOKEN_EXCHANGE_ENV: "0",
        pi_mod.PI_LITELLM_SUPPRESS_DEBUG_INFO_ENV: "True",
    }


def test_build_pi_proxy_metadata_env_for_openai_has_only_generic_metadata() -> None:
    env = pi_mod.build_pi_proxy_metadata_env("openai")
    assert env == {
        pi_mod.PI_PROXY_METADATA_FAMILY_ENV: "openai",
        pi_mod.PI_PROXY_METADATA_CAPABILITY_ENV: "1",
    }


class _FakeResponse:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return self._payload.encode("utf-8")


def _urlopen_with_payload(payload: dict[str, object]):
    def _urlopen(url: str, timeout: int = 0) -> _FakeResponse:
        assert url.endswith(pi_mod.PI_ATTACH_METADATA_PATH)
        assert timeout == 2
        return _FakeResponse(json.dumps(payload))

    return _urlopen


def test_probe_attach_compatibility_accepts_matching_metadata() -> None:
    payload = {
        "headroomVersion": __version__,
        "backend": "openai",
        "upstreamFamily": "openai",
        "memory": False,
        "capabilities": {"attachCompatible": True, "wrapPi": True},
    }
    result = pi_mod.probe_attach_compatibility(
        "github-copilot",
        8788,
        backend=None,
        memory=False,
        urlopen=_urlopen_with_payload(payload),
    )
    assert result.status == "compatible"
    assert result.metadata is not None
    assert result.metadata.upstream_family == "openai"


def test_probe_attach_compatibility_rejects_backend_mismatch() -> None:
    payload = {
        "headroomVersion": __version__,
        "backend": "bedrock",
        "upstreamFamily": "openai",
        "memory": False,
        "capabilities": {"attachCompatible": True, "wrapPi": True},
    }
    result = pi_mod.probe_attach_compatibility(
        "openai",
        8789,
        backend=None,
        memory=False,
        urlopen=_urlopen_with_payload(payload),
    )
    assert result.status == "incompatible"
    assert "backend" in result.reason


def test_probe_attach_compatibility_rejects_missing_required_metadata_fields() -> None:
    payload = {
        "headroomVersion": __version__,
        "backend": "openai",
        "memory": False,
        "capabilities": {"attachCompatible": True, "wrapPi": True},
    }
    result = pi_mod.probe_attach_compatibility(
        "openai",
        8789,
        backend=None,
        memory=False,
        urlopen=_urlopen_with_payload(payload),
    )
    assert result.status == "incompatible"
    assert "invalid payload" in result.reason.lower()


def test_probe_attach_compatibility_reports_missing_metadata_endpoint() -> None:
    def failing_urlopen(url: str, timeout: int = 0):
        raise OSError("connection refused")

    result = pi_mod.probe_attach_compatibility(
        "anthropic",
        8790,
        backend="anthropic",
        memory=True,
        urlopen=failing_urlopen,
    )
    assert result.status == "missing"
    assert "/headroom/meta" in result.reason


@pytest.mark.skipif(
    shutil.which("pi") is None or shutil.which("node") is None,
    reason="pi and node are required for the live Phase 0 feasibility probe",
)
def test_run_phase0_feasibility_probe_hits_override_then_native_fallback() -> None:
    result = pi_mod.run_phase0_feasibility_probe()

    assert result["firstResponse"] == "override response"
    assert result["secondResponse"] == "native response"
    assert [request["url"] for request in result["requests"]["override"]] == [
        "/v1/chat/completions"
    ]
    assert [request["url"] for request in result["requests"]["native"]] == [
        "/v1/chat/completions"
    ]

    event_types = [event["type"] for event in result["events"]]
    assert "provider_observed" in event_types
    assert "model_select" in event_types
    assert "provider_registered" in event_types
    assert "provider_unregistered" in event_types

    model_selects = [event for event in result["events"] if event["type"] == "model_select"]
    assert [event["currentModel"]["provider"] for event in model_selects] == [
        "anthropic",
        "openai",
    ]

    registered = [event for event in result["events"] if event["type"] == "provider_registered"]
    assert registered[0]["providerId"] == "openai"
    assert registered[0]["baseUrl"] == f"http://127.0.0.1:{result['overridePort']}/v1"

    unregistered = [event for event in result["events"] if event["type"] == "provider_unregistered"]
    assert any("forced-native" in event["reason"] for event in unregistered)


def test_resolve_runtime_provider_prefers_runtime_provider() -> None:
    assert pi_mod.resolve_runtime_provider("github-copilot", "openai/gpt-5") == (
        "github-copilot",
        "runtime-provider",
    )


def test_resolve_runtime_provider_falls_back_to_supported_model_prefix() -> None:
    assert pi_mod.resolve_runtime_provider(None, "anthropic/claude-sonnet") == (
        "anthropic",
        "model-id",
    )


def test_resolve_runtime_provider_leaves_unknown_models_unmanaged() -> None:
    assert pi_mod.resolve_runtime_provider(None, "google/gemini-2.5-pro") == (
        None,
        "unresolved",
    )


@pytest.mark.skipif(
    shutil.which("pi") is None or shutil.which("node") is None,
    reason="pi and node are required for the live dynamic routing probe",
)
def test_run_dynamic_routing_probe_proves_hysteresis_and_family_routing() -> None:
    result = pi_mod.run_dynamic_routing_probe()

    assert result["firstResponse"] == "overrideOpenai response"
    assert result["suspectResponse"] == "overrideOpenai response"
    assert result["unavailableResponse"] == "nativeOpenai response"
    assert result["recoveryHoldResponse"] == "nativeOpenai response"
    assert result["reattachPrimeResponse"] == "nativeOpenai response"

    assert [request["url"] for request in result["requests"]["overrideOpenai"]] == [
        "/v1/chat/completions",
        "/v1/chat/completions",
    ]
    assert [request["url"] for request in result["requests"]["nativeOpenai"]] == [
        "/v1/chat/completions",
        "/v1/chat/completions",
        "/v1/chat/completions",
        "/v1/chat/completions",
    ]

    registered = [event for event in result["events"] if event["type"] == "provider_registered"]
    openai_registrations = [event for event in registered if event["providerId"] == "openai"]
    registered_pairs = {(event["providerId"], event["baseUrl"]) for event in registered}
    assert ("openai", f"http://127.0.0.1:{result['overrideOpenaiPort']}/v1") in registered_pairs
    assert ("anthropic", f"http://127.0.0.1:{result['overrideAnthropicPort']}") in registered_pairs
    assert len(openai_registrations) >= 2
    assert openai_registrations[-1]["healthStatus"] == "healthy"

    observed = [event for event in result["events"] if event["type"] == "provider_observed"]
    assert any(
        event["resolvedProvider"] == "google"
        and event["resolutionSource"] == "runtime-provider"
        for event in observed
    )

    health_checks = [
        event for event in result["events"]
        if event["type"] == "provider_health_checked" and event["providerId"] == "openai"
    ]
    assert len(health_checks) == 5

    transitions = [
        (event["previousStatus"], event["status"])
        for event in result["events"]
        if event["type"] == "provider_health_transition" and event["providerId"] == "openai"
    ]
    assert ("healthy", "suspect") in transitions
    assert ("suspect", "unavailable") in transitions
    assert ("unavailable", "healthy") in transitions

    unregistered = [event for event in result["events"] if event["type"] == "provider_unregistered"]
    assert any(event["providerId"] == "openai" for event in unregistered)

