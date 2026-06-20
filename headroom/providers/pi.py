"""Helpers for Headroom's Phase 0 `wrap pi` scaffolding."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import click

from headroom._version import __version__

PI_SESSION_CONFIG_ENV = "HEADROOM_PI_SESSION_CONFIG"
PI_VERBOSE_ENV = "HEADROOM_PI_VERBOSE"
PI_ATTACH_METADATA_PATH = "/headroom/meta"
PI_PROXY_METADATA_FAMILY_ENV = "HEADROOM_PROXY_WRAP_PI_UPSTREAM_FAMILY"
PI_PROXY_METADATA_CAPABILITY_ENV = "HEADROOM_PROXY_WRAP_PI_ATTACH_CAPABLE"
PI_SUPPORTED_PROVIDERS = ("openai", "anthropic", "github-copilot")
PI_DEFAULT_PROVIDER_ORDER = PI_SUPPORTED_PROVIDERS
PI_PHASE0_PROBE_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class PiProviderSpec:
    """Static provider routing metadata for `wrap pi`."""

    provider_id: str
    family: str
    default_port: int
    routed_suffix: str

    def root_url(self, port: int) -> str:
        return f"http://127.0.0.1:{port}"

    def routed_base_url(self, port: int) -> str:
        if not self.routed_suffix:
            return self.root_url(port)
        return f"{self.root_url(port)}{self.routed_suffix}"


@dataclass(frozen=True)
class PiAttachMetadata:
    """Structured compatibility metadata advertised by a proxy for wrap-pi attach mode."""

    headroom_version: str
    backend: str
    upstream_family: str
    memory: bool
    attach_compatible: bool
    wrap_pi: bool


@dataclass(frozen=True)
class PiAttachProbeResult:
    """Result of checking whether wrap-pi may attach to a proxy already bound on a port."""

    status: Literal["missing", "compatible", "incompatible"]
    provider_id: str
    port: int
    reason: str
    metadata: PiAttachMetadata | None = None


def managed_provider_specs() -> dict[str, PiProviderSpec]:
    """Return Phase 0 supported pi provider specs."""

    return {
        "openai": PiProviderSpec(
            provider_id="openai",
            family="openai",
            default_port=8789,
            routed_suffix="/v1",
        ),
        "anthropic": PiProviderSpec(
            provider_id="anthropic",
            family="anthropic",
            default_port=8790,
            routed_suffix="",
        ),
        "github-copilot": PiProviderSpec(
            provider_id="github-copilot",
            family="openai",
            default_port=8788,
            routed_suffix="/v1",
        ),
    }


def default_provider_ports() -> dict[str, int]:
    """Return canonical default proxy ports for `wrap pi`."""

    return {
        provider_id: spec.default_port
        for provider_id, spec in managed_provider_specs().items()
    }


def resolve_pi_binary(which: Any = shutil.which) -> str:
    """Resolve the installed `pi` executable path."""

    resolved = which("pi")
    if not resolved:
        raise click.ClickException("Could not find `pi` on PATH.")
    return resolved


def resolve_pi_proxy_backend(backend: str | None) -> str:
    """Resolve the backend a wrap-pi invocation expects for managed proxies."""

    value = (backend or os.environ.get("HEADROOM_BACKEND") or "anthropic").strip()
    return value or "anthropic"


def resolve_managed_providers(providers: Sequence[str]) -> list[str]:
    """Validate and normalize the managed provider list for `wrap pi`."""

    if not providers:
        return list(PI_DEFAULT_PROVIDER_ORDER)

    specs = managed_provider_specs()
    resolved: list[str] = []
    seen: set[str] = set()
    for raw_provider in providers:
        provider = raw_provider.strip().lower()
        if provider == "all":
            raise click.ClickException("'--provider all' is not supported for headroom wrap pi v1.")
        if provider not in specs:
            allowed = ", ".join(PI_SUPPORTED_PROVIDERS)
            raise click.ClickException(
                f"Unsupported pi provider '{raw_provider}'. Allowed values: {allowed}."
            )
        if provider not in seen:
            seen.add(provider)
            resolved.append(provider)
    return resolved


def resolve_provider_ports(
    managed_providers: Sequence[str],
    port_override: int | None,
) -> dict[str, int]:
    """Resolve the per-provider port mapping for the current pi session."""

    ports = default_provider_ports()
    if port_override is None:
        return {provider_id: ports[provider_id] for provider_id in managed_providers}
    if len(managed_providers) != 1:
        raise click.ClickException("'--port' is only valid when exactly one pi provider is managed.")
    provider_id = managed_providers[0]
    return {provider_id: port_override}


def build_pi_wrap_session_config(
    managed_providers: Sequence[str],
    provider_ports: Mapping[str, int],
    *,
    health_ttl_ms: int = 5000,
    detach_failures: int = 2,
    reattach_successes: int = 2,
    enable_status_command: bool = True,
    enable_footer: bool = True,
    phase0: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical versioned `session.json` payload for `wrap pi`."""

    specs = managed_provider_specs()
    providers: dict[str, Any] = {}
    for provider_id in managed_providers:
        spec = specs[provider_id]
        port = provider_ports[provider_id]
        providers[provider_id] = {
            "port": port,
            "rootUrl": spec.root_url(port),
            "routedBaseUrl": spec.routed_base_url(port),
            "family": spec.family,
        }

    payload: dict[str, Any] = {
        "version": 1,
        "proxyHost": "127.0.0.1",
        "managedProviders": list(managed_providers),
        "providers": providers,
        "health": {
            "ttlMs": health_ttl_ms,
            "detachFailures": detach_failures,
            "reattachSuccesses": reattach_successes,
        },
        "ui": {
            "enableStatusCommand": enable_status_command,
            "enableFooter": enable_footer,
        },
    }
    if phase0:
        payload["phase0"] = dict(phase0)
    return payload


def write_pi_session_config(temp_dir: Path, session_config: Mapping[str, Any]) -> Path:
    """Write the session config JSON into a temporary pi workspace."""

    path = temp_dir / "session.json"
    path.write_text(json.dumps(dict(session_config), indent=2) + "\n", encoding="utf-8")
    return path


def render_pi_extension(temp_dir: Path, session_config_path: Path) -> Path:
    """Materialize the checked-in pi extension template into a temp workspace."""

    del session_config_path  # The extension reads the path from HEADROOM_PI_SESSION_CONFIG.
    source = Path(__file__).with_name("pi_extension_template.ts")
    target = temp_dir / "extension.ts"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def build_pi_launch_env(
    base_env: Mapping[str, str],
    session_config_path: Path,
    *,
    verbose: bool,
) -> dict[str, str]:
    """Build the environment for a pi session-scoped extension launch."""

    env = dict(base_env)
    env[PI_SESSION_CONFIG_ENV] = str(session_config_path)
    if verbose:
        env[PI_VERBOSE_ENV] = "1"
    else:
        env.pop(PI_VERBOSE_ENV, None)
    return env


def _has_user_extension_arg(pi_args: Sequence[str]) -> bool:
    for arg in pi_args:
        if arg in {"--extension", "-e"}:
            return True
        if arg.startswith("--extension="):
            return True
        if arg.startswith("-e") and arg != "-e":
            return True
    return False


def build_pi_launch_args(pi_args: Sequence[str], extension_path: Path) -> tuple[str, ...]:
    """Append exactly one Headroom-managed `--extension` flag to pi args."""

    if _has_user_extension_arg(pi_args):
        raise click.ClickException(
            "User-supplied pi '--extension' arguments are rejected during Phase 0 because "
            "extension ordering is not proven deterministic yet."
        )
    return (*pi_args, "--extension", str(extension_path))


def build_pi_proxy_metadata_env(provider_id: str) -> dict[str, str]:
    """Return proxy env vars needed for wrap-pi attach metadata."""

    spec = managed_provider_specs()[provider_id]
    return {
        PI_PROXY_METADATA_FAMILY_ENV: spec.family,
        PI_PROXY_METADATA_CAPABILITY_ENV: "1",
    }


def _coerce_metadata_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _read_attach_metadata_payload(
    port: int,
    *,
    urlopen: Any = urllib.request.urlopen,
) -> tuple[bool, object | None]:
    """Return whether the metadata endpoint was reachable plus the decoded payload."""

    url = f"http://127.0.0.1:{port}{PI_ATTACH_METADATA_PATH}"
    try:
        with urlopen(url, timeout=2) as response:
            raw = response.read().decode("utf-8")
    except (OSError, urllib.error.URLError):
        return False, None
    try:
        return True, json.loads(raw)
    except (ValueError, json.JSONDecodeError):
        return True, None


def _parse_attach_metadata(payload: object) -> PiAttachMetadata | None:
    if not isinstance(payload, dict):
        return None
    headroom_version = payload.get("headroomVersion")
    backend = payload.get("backend")
    upstream_family = payload.get("upstreamFamily")
    memory = payload.get("memory")
    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, dict):
        return None
    attach_compatible = _coerce_metadata_bool(capabilities.get("attachCompatible"))
    wrap_pi = _coerce_metadata_bool(capabilities.get("wrapPi"))
    if not isinstance(headroom_version, str) or not headroom_version:
        return None
    if not isinstance(backend, str) or not backend:
        return None
    if not isinstance(upstream_family, str) or not upstream_family:
        return None
    if not isinstance(memory, bool):
        return None
    if attach_compatible is None or wrap_pi is None:
        return None
    return PiAttachMetadata(
        headroom_version=headroom_version,
        backend=backend,
        upstream_family=upstream_family,
        memory=memory,
        attach_compatible=attach_compatible,
        wrap_pi=wrap_pi,
    )


def probe_attach_compatibility(
    provider_id: str,
    port: int,
    *,
    backend: str | None,
    memory: bool,
    urlopen: Any = urllib.request.urlopen,
) -> PiAttachProbeResult:
    """Validate whether wrap-pi may attach to a proxy already listening on a port."""

    reachable, payload = _read_attach_metadata_payload(port, urlopen=urlopen)
    if not reachable:
        return PiAttachProbeResult(
            status="missing",
            provider_id=provider_id,
            port=port,
            reason=f"No {PI_ATTACH_METADATA_PATH} endpoint responded on port {port}.",
        )

    metadata = _parse_attach_metadata(payload)
    if metadata is None:
        return PiAttachProbeResult(
            status="incompatible",
            provider_id=provider_id,
            port=port,
            reason=f"Metadata endpoint on port {port} returned an invalid payload.",
        )

    expected_backend = resolve_pi_proxy_backend(backend)
    expected_family = managed_provider_specs()[provider_id].family
    if not metadata.attach_compatible or not metadata.wrap_pi:
        reason = f"Metadata on port {port} does not advertise wrap-pi attach compatibility."
        return PiAttachProbeResult(
            status="incompatible",
            provider_id=provider_id,
            port=port,
            reason=reason,
            metadata=metadata,
        )
    if metadata.headroom_version != __version__:
        return PiAttachProbeResult(
            status="incompatible",
            provider_id=provider_id,
            port=port,
            reason=(
                f"Metadata on port {port} reports Headroom {metadata.headroom_version}, "
                f"but wrap pi requires {__version__}."
            ),
            metadata=metadata,
        )
    if metadata.backend != expected_backend:
        return PiAttachProbeResult(
            status="incompatible",
            provider_id=provider_id,
            port=port,
            reason=(
                f"Metadata on port {port} reports backend {metadata.backend!r}, "
                f"expected {expected_backend!r}."
            ),
            metadata=metadata,
        )
    if metadata.upstream_family != expected_family:
        return PiAttachProbeResult(
            status="incompatible",
            provider_id=provider_id,
            port=port,
            reason=(
                f"Metadata on port {port} reports upstream family {metadata.upstream_family!r}, "
                f"expected {expected_family!r} for provider {provider_id!r}."
            ),
            metadata=metadata,
        )
    if metadata.memory != memory:
        return PiAttachProbeResult(
            status="incompatible",
            provider_id=provider_id,
            port=port,
            reason=(
                f"Metadata on port {port} reports memory={metadata.memory!r}, "
                f"expected {memory!r}."
            ),
            metadata=metadata,
        )
    return PiAttachProbeResult(
        status="compatible",
        provider_id=provider_id,
        port=port,
        reason=(
            f"Existing proxy on port {port} matches backend={expected_backend!r}, "
            f"family={expected_family!r}, memory={memory!r}."
        ),
        metadata=metadata,
    )


def _derive_pi_sdk_index_path(pi_binary: str) -> Path:
    resolved = Path(pi_binary).resolve()
    if resolved.name == "cli.js" and resolved.parent.name == "dist":
        return resolved.parent / "index.js"
    candidate = resolved.parent / "index.js"
    if candidate.exists():
        return candidate
    raise RuntimeError(
        f"Unable to derive pi SDK path from executable {pi_binary!r}; expected dist/index.js nearby."
    )



def run_phase0_feasibility_probe(pi_binary: str | None = None) -> dict[str, Any]:
    """Run the local Phase 0 pi runtime proof against fake endpoints."""

    resolved_pi = pi_binary or resolve_pi_binary()
    node_binary = shutil.which("node")
    if not node_binary:
        raise RuntimeError("node is required to run the pi Phase 0 feasibility probe.")

    sdk_index = _derive_pi_sdk_index_path(resolved_pi)

    with tempfile.TemporaryDirectory(prefix="headroom-pi-phase0-") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        session_config_path = write_pi_session_config(
            temp_dir,
            build_pi_wrap_session_config(
                ["openai"],
                {"openai": 32111},
                phase0={"forceNativeProviders": []},
            ),
        )
        extension_path = render_pi_extension(temp_dir, session_config_path)
        results_path = temp_dir / "results.json"
        log_path = temp_dir / "phase0-events.jsonl"
        script_path = temp_dir / "phase0_probe.mjs"

        session_config = json.loads(session_config_path.read_text(encoding="utf-8"))
        session_config["phase0"] = {"logPath": str(log_path), "forceNativeProviders": []}
        session_config_path.write_text(json.dumps(session_config, indent=2) + "\n", encoding="utf-8")

        script_path.write_text(
            _build_phase0_probe_script(
                sdk_index_path=sdk_index,
                session_config_path=session_config_path,
                extension_path=extension_path,
                results_path=results_path,
                work_dir=temp_dir / "workdir",
                agent_dir=temp_dir / "agentdir",
            ),
            encoding="utf-8",
        )

        env = os.environ.copy()
        env[PI_SESSION_CONFIG_ENV] = str(session_config_path)
        result = subprocess.run(
            [node_binary, str(script_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=PI_PHASE0_PROBE_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "pi Phase 0 feasibility probe failed:\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )
        return json.loads(results_path.read_text(encoding="utf-8"))



def _build_phase0_probe_script(
    *,
    sdk_index_path: Path,
    session_config_path: Path,
    extension_path: Path,
    results_path: Path,
    work_dir: Path,
    agent_dir: Path,
) -> str:
    script = textwrap.dedent(
        """
        import http from "node:http";
        import { once } from "node:events";
        import { promises as fs } from "node:fs";
        import process from "node:process";

        const sdk = await import(__SDK_INDEX__);
        const {
          AuthStorage,
          DefaultResourceLoader,
          ModelRegistry,
          SessionManager,
          SettingsManager,
          createAgentSession,
        } = sdk;

        const sessionConfigPath = __SESSION_CONFIG_PATH__;
        const extensionPath = __EXTENSION_PATH__;
        const resultsPath = __RESULTS_PATH__;
        const workDir = __WORK_DIR__;
        const agentDir = __AGENT_DIR__;
        const logPath = JSON.parse(await fs.readFile(sessionConfigPath, "utf8")).phase0.logPath;

        await fs.mkdir(workDir, { recursive: true });
        await fs.mkdir(agentDir, { recursive: true });
        process.env.HEADROOM_PI_SESSION_CONFIG = sessionConfigPath;

        const requests = { native: [], override: [] };

        async function readBody(req) {
          const chunks = [];
          for await (const chunk of req) {
            chunks.push(Buffer.from(chunk));
          }
          return Buffer.concat(chunks).toString("utf8");
        }

        function createOpenAIServer(label) {
          return http.createServer(async (req, res) => {
            const rawBody = await readBody(req);
            const body = JSON.parse(rawBody);
            requests[label].push({ url: req.url, body });
            const model = body.model || "phase0-openai";

            res.writeHead(200, { "content-type": "text/event-stream" });
            res.write(
              "data: " +
                JSON.stringify({
                  id: label + "-1",
                  object: "chat.completion.chunk",
                  created: 0,
                  model,
                  choices: [
                    {
                      index: 0,
                      delta: { role: "assistant", content: label + " response" },
                      finish_reason: null,
                    },
                  ],
                }) +
                "\\n\\n",
            );
            res.write(
              "data: " +
                JSON.stringify({
                  id: label + "-1",
                  object: "chat.completion.chunk",
                  created: 0,
                  model,
                  choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
                  usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
                }) +
                "\\n\\n",
            );
            res.write("data: [DONE]\\n\\n");
            res.end();
          });
        }

        const nativeServer = createOpenAIServer("native");
        const overrideServer = createOpenAIServer("override");
        nativeServer.listen(0, "127.0.0.1");
        overrideServer.listen(0, "127.0.0.1");
        await Promise.all([once(nativeServer, "listening"), once(overrideServer, "listening")]);

        const nativePort = nativeServer.address().port;
        const overridePort = overrideServer.address().port;
        const nativeBaseUrl = "http://127.0.0.1:" + nativePort + "/v1";
        const overrideBaseUrl = "http://127.0.0.1:" + overridePort + "/v1";

        const authStorage = AuthStorage.create();
        authStorage.setRuntimeApiKey("openai", "sk-phase0");
        authStorage.setRuntimeApiKey("anthropic", "sk-phase0");

        const modelRegistry = ModelRegistry.inMemory(authStorage);
        modelRegistry.registerProvider("openai", {
          baseUrl: nativeBaseUrl,
          apiKey: "$OPENAI_API_KEY",
          api: "openai-completions",
          models: [
            {
              id: "phase0-openai",
              name: "Phase 0 OpenAI",
              reasoning: false,
              input: ["text"],
              cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
              contextWindow: 128000,
              maxTokens: 4096,
            },
          ],
        });
        modelRegistry.registerProvider("anthropic", {
          baseUrl: "http://127.0.0.1:9",
          apiKey: "$ANTHROPIC_API_KEY",
          api: "anthropic-messages",
          models: [
            {
              id: "phase0-anthropic",
              name: "Phase 0 Anthropic",
              reasoning: true,
              input: ["text"],
              cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
              contextWindow: 200000,
              maxTokens: 4096,
            },
          ],
        });

        const config = JSON.parse(await fs.readFile(sessionConfigPath, "utf8"));
        config.providers.openai.port = overridePort;
        config.providers.openai.rootUrl = "http://127.0.0.1:" + overridePort;
        config.providers.openai.routedBaseUrl = overrideBaseUrl;
        await fs.writeFile(sessionConfigPath, JSON.stringify(config, null, 2) + "\\n", "utf8");

        const settingsManager = SettingsManager.inMemory({
          compaction: { enabled: false },
          retry: { enabled: false },
        });
        const loader = new DefaultResourceLoader({
          cwd: workDir,
          agentDir,
          additionalExtensionPaths: [extensionPath],
          settingsManager,
        });
        await loader.reload();

        const openaiModel = modelRegistry.find("openai", "phase0-openai");
        const anthropicModel = modelRegistry.find("anthropic", "phase0-anthropic");
        if (!openaiModel || !anthropicModel) {
          throw new Error("Failed to register Phase 0 probe models.");
        }

        const { session } = await createAgentSession({
          cwd: workDir,
          agentDir,
          authStorage,
          modelRegistry,
          settingsManager,
          resourceLoader: loader,
          model: openaiModel,
          sessionManager: SessionManager.inMemory(workDir),
          noTools: "all",
        });

        function lastAssistantText() {
          const assistants = session.messages.filter((message) => message.role === "assistant");
          const latest = assistants[assistants.length - 1];
          if (!latest) {
            return null;
          }
          return latest.content
            .filter((block) => block.type === "text")
            .map((block) => block.text)
            .join("");
        }

        try {
          await session.prompt("Use the current model and respond with one short sentence.");
          const firstResponse = lastAssistantText();

          await session.setModel(anthropicModel);
          await session.setModel(openaiModel);

          const nativeConfig = JSON.parse(await fs.readFile(sessionConfigPath, "utf8"));
          nativeConfig.phase0.forceNativeProviders = ["openai"];
          await fs.writeFile(sessionConfigPath, JSON.stringify(nativeConfig, null, 2) + "\\n", "utf8");

          await session.prompt("Respond again after native fallback is restored.");
          const secondResponse = lastAssistantText();

          const events = (await fs.readFile(logPath, "utf8"))
            .trim()
            .split("\\n")
            .filter(Boolean)
            .map((line) => JSON.parse(line));

          await fs.writeFile(
            resultsPath,
            JSON.stringify(
              {
                firstResponse,
                secondResponse,
                requests,
                events,
                nativePort,
                overridePort,
              },
              null,
              2,
            ) + "\\n",
            "utf8",
          );
        } finally {
          session.dispose();
          await Promise.all([
            new Promise((resolve) => nativeServer.close(resolve)),
            new Promise((resolve) => overrideServer.close(resolve)),
          ]);
        }
        """
    )
    return (
        script.replace("__SDK_INDEX__", json.dumps(sdk_index_path.as_uri()))
        .replace("__SESSION_CONFIG_PATH__", json.dumps(str(session_config_path)))
        .replace("__EXTENSION_PATH__", json.dumps(str(extension_path)))
        .replace("__RESULTS_PATH__", json.dumps(str(results_path)))
        .replace("__WORK_DIR__", json.dumps(str(work_dir)))
        .replace("__AGENT_DIR__", json.dumps(str(agent_dir)))
    )
