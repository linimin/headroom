import type {
  ExtensionAPI,
  ExtensionContext,
} from "@earendil-works/pi-coding-agent";
import { promises as fs } from "node:fs";

interface SessionProviderConfig {
  port: number;
  rootUrl: string;
  routedBaseUrl: string;
  family: string;
}

interface SessionHealthConfig {
  ttlMs: number;
  detachFailures: number;
  reattachSuccesses: number;
}

interface Phase0Config {
  logPath?: string;
  forceNativeProviders?: string[];
}

interface SessionConfig {
  version: number;
  managedProviders: string[];
  providers: Record<string, SessionProviderConfig>;
  health: SessionHealthConfig;
  phase0?: Phase0Config;
}

interface ModelSnapshot {
  provider: string | null;
  id: string;
}

type HealthStatus = "healthy" | "suspect" | "unavailable";
type ResolutionSource = "runtime-provider" | "model-id" | "unresolved";

interface ProviderHealthState {
  status: HealthStatus;
  consecutiveFailures: number;
  consecutiveSuccesses: number;
  lastCheckedAt: number;
  nextProbeAt: number;
  hasEverAttached: boolean;
  lastFailure: string | null;
}

const HEALTH_PATHS = ["/health", "/readyz", "/livez"];
const MANAGED_PROVIDER_PREFIXES = ["openai", "anthropic", "github-copilot"];

export default function (pi: ExtensionAPI) {
  const registeredProviders = new Map<string, string>();
  const providerHealth = new Map<string, ProviderHealthState>();

  const normalizeProvider = (value: string | null | undefined): string | null => {
    if (!value) {
      return null;
    }
    const normalized = value.trim().toLowerCase();
    return normalized || null;
  };

  const getModelSnapshot = (model: { provider?: string; id?: string } | null | undefined): ModelSnapshot | null => {
    if (!model || typeof model.id !== "string" || !model.id.trim()) {
      return null;
    }
    return {
      provider: normalizeProvider(model.provider),
      id: model.id,
    };
  };

  const getCurrentModel = (ctx: ExtensionContext): ModelSnapshot | null => getModelSnapshot(ctx.model);

  const loadConfig = async (): Promise<SessionConfig> => {
    const configPath = process.env.HEADROOM_PI_SESSION_CONFIG;
    if (!configPath) {
      throw new Error("HEADROOM_PI_SESSION_CONFIG is required for the Headroom pi extension.");
    }
    return JSON.parse(await fs.readFile(configPath, "utf8")) as SessionConfig;
  };

  const logEvent = async (
    config: SessionConfig,
    type: string,
    details: Record<string, unknown>,
  ): Promise<void> => {
    const logPath = config.phase0?.logPath;
    if (!logPath) {
      return;
    }
    await fs.appendFile(
      logPath,
      JSON.stringify({ type, ...details, timestamp: Date.now() }) + "\n",
      "utf8",
    );
  };

  const deriveProviderFromModelId = (modelId: string): string | null => {
    const normalized = modelId.trim().toLowerCase();
    for (const providerId of MANAGED_PROVIDER_PREFIXES) {
      if (normalized === providerId || normalized.startsWith(`${providerId}/`)) {
        return providerId;
      }
    }
    return null;
  };

  const getResolution = (
    model: ModelSnapshot | null,
  ): { providerId: string | null; source: ResolutionSource } => {
    if (!model) {
      return { providerId: null, source: "unresolved" };
    }
    if (model.provider) {
      return { providerId: model.provider, source: "runtime-provider" };
    }
    const derivedProvider = deriveProviderFromModelId(model.id);
    if (derivedProvider) {
      return { providerId: derivedProvider, source: "model-id" };
    }
    return { providerId: null, source: "unresolved" };
  };

  const getHealthState = (providerId: string): ProviderHealthState => {
    const existing = providerHealth.get(providerId);
    if (existing) {
      return existing;
    }
    const initialState: ProviderHealthState = {
      status: "healthy",
      consecutiveFailures: 0,
      consecutiveSuccesses: 0,
      lastCheckedAt: 0,
      nextProbeAt: 0,
      hasEverAttached: false,
      lastFailure: null,
    };
    providerHealth.set(providerId, initialState);
    return initialState;
  };

  const probeProviderHealth = async (rootUrl: string): Promise<{ ok: boolean; detail: string }> => {
    let lastDetail = "unreachable";
    for (const path of HEALTH_PATHS) {
      const url = `${rootUrl}${path}`;
      try {
        const response = await fetch(url);
        if (response.ok) {
          return { ok: true, detail: path };
        }
        lastDetail = `${path}:${response.status}`;
      } catch (error) {
        lastDetail = `${path}:${error instanceof Error ? error.message : String(error)}`;
      }
    }
    return { ok: false, detail: lastDetail };
  };

  const refreshProviderHealth = async (
    config: SessionConfig,
    providerId: string,
    reason: string,
  ): Promise<ProviderHealthState> => {
    const managedConfig = config.providers[providerId];
    const state = getHealthState(providerId);
    const now = Date.now();
    if (state.nextProbeAt > now) {
      return state;
    }

    const previousStatus = state.status;
    const probeResult = await probeProviderHealth(managedConfig.rootUrl);
    const detachFailures = Math.max(1, config.health.detachFailures ?? 2);
    const reattachSuccesses = Math.max(1, config.health.reattachSuccesses ?? 2);
    const ttlMs = Math.max(0, config.health.ttlMs ?? 0);

    if (probeResult.ok) {
      state.lastFailure = null;
      state.consecutiveFailures = 0;
      if (previousStatus === "unavailable") {
        state.consecutiveSuccesses += 1;
        if (state.consecutiveSuccesses >= reattachSuccesses) {
          state.status = "healthy";
        }
      } else {
        state.consecutiveSuccesses = 1;
        state.status = "healthy";
      }
    } else {
      state.lastFailure = probeResult.detail;
      state.consecutiveSuccesses = 0;
      state.consecutiveFailures += 1;
      if (state.consecutiveFailures >= detachFailures) {
        state.status = "unavailable";
      } else {
        state.status = "suspect";
      }
    }

    state.lastCheckedAt = now;
    state.nextProbeAt = now + ttlMs;
    providerHealth.set(providerId, state);

    await logEvent(config, "provider_health_checked", {
      providerId,
      reason,
      rootUrl: managedConfig.rootUrl,
      ok: probeResult.ok,
      detail: probeResult.detail,
      status: state.status,
      previousStatus,
      consecutiveFailures: state.consecutiveFailures,
      consecutiveSuccesses: state.consecutiveSuccesses,
      ttlMs,
    });

    if (previousStatus !== state.status) {
      await logEvent(config, "provider_health_transition", {
        providerId,
        reason,
        previousStatus,
        status: state.status,
        detail: probeResult.detail,
        consecutiveFailures: state.consecutiveFailures,
        consecutiveSuccesses: state.consecutiveSuccesses,
      });
    }

    return state;
  };

  const unregisterProvider = async (
    config: SessionConfig,
    providerId: string,
    reason: string,
  ): Promise<void> => {
    if (!registeredProviders.has(providerId)) {
      return;
    }
    pi.unregisterProvider(providerId);
    registeredProviders.delete(providerId);
    await logEvent(config, "provider_unregistered", { providerId, reason });
  };

  const unregisterOtherProviders = async (
    config: SessionConfig,
    activeProviderId: string | null,
    reason: string,
  ): Promise<void> => {
    for (const providerId of Array.from(registeredProviders.keys())) {
      if (providerId === activeProviderId) {
        continue;
      }
      await unregisterProvider(config, providerId, reason);
    }
  };

  const registerProvider = async (
    config: SessionConfig,
    providerId: string,
    managedConfig: SessionProviderConfig,
    reason: string,
    currentModel: ModelSnapshot | null,
    resolutionSource: ResolutionSource,
    healthState: ProviderHealthState,
  ): Promise<void> => {
    const existingBaseUrl = registeredProviders.get(providerId);
    if (existingBaseUrl !== undefined) {
      try {
        pi.unregisterProvider(providerId);
      } catch {
        // pi may treat unregistering an unmanaged provider as a no-op.
      }
    }
    pi.registerProvider(providerId, { baseUrl: managedConfig.routedBaseUrl });
    registeredProviders.set(providerId, managedConfig.routedBaseUrl);
    healthState.hasEverAttached = true;
    await logEvent(config, "provider_registered", {
      providerId,
      baseUrl: managedConfig.routedBaseUrl,
      previousBaseUrl: existingBaseUrl ?? null,
      reapplied: existingBaseUrl === managedConfig.routedBaseUrl,
      reason,
      currentModel,
      resolutionSource,
      healthStatus: healthState.status,
    });
  };

  const syncCurrentProvider = async (
    config: SessionConfig,
    currentModel: ModelSnapshot | null,
    reason: string,
  ): Promise<void> => {
    const resolution = getResolution(currentModel);
    const providerId = resolution.providerId;

    await logEvent(config, "provider_observed", {
      reason,
      currentModel,
      resolvedProvider: providerId,
      resolutionSource: resolution.source,
    });

    if (!providerId) {
      await unregisterOtherProviders(config, null, `${reason}:unresolved-provider`);
      return;
    }

    await unregisterOtherProviders(config, providerId, `${reason}:switched-provider`);

    const managedConfig = config.providers[providerId];
    if (!managedConfig) {
      await unregisterProvider(config, providerId, `${reason}:unmanaged-provider`);
      return;
    }

    if (config.phase0?.forceNativeProviders?.includes(providerId)) {
      await unregisterProvider(config, providerId, `${reason}:forced-native`);
      return;
    }

    const healthState = await refreshProviderHealth(config, providerId, reason);
    const shouldAttach =
      healthState.status === "healthy" ||
      (healthState.status === "suspect" && (healthState.hasEverAttached || registeredProviders.has(providerId)));

    if (!shouldAttach) {
      await unregisterProvider(config, providerId, `${reason}:proxy-${healthState.status}`);
      return;
    }

    await registerProvider(
      config,
      providerId,
      managedConfig,
      reason,
      currentModel,
      resolution.source,
      healthState,
    );
  };

  pi.on("session_start", async (event, ctx) => {
    const config = await loadConfig();
    const currentModel = getCurrentModel(ctx);
    await logEvent(config, "session_start", {
      reason: event.reason,
      currentModel,
      sessionConfigPath: process.env.HEADROOM_PI_SESSION_CONFIG,
    });
    await syncCurrentProvider(config, currentModel, "session_start");
  });

  pi.on("model_select", async (event) => {
    const config = await loadConfig();
    const currentModel = getModelSnapshot(event.model);
    await logEvent(config, "model_select", {
      source: event.source,
      currentModel,
      previousModel: getModelSnapshot(event.previousModel),
    });
    await syncCurrentProvider(config, currentModel, "model_select");
  });

  pi.on("before_agent_start", async (_event, ctx) => {
    const config = await loadConfig();
    await syncCurrentProvider(config, getCurrentModel(ctx), "before_agent_start");
  });

  pi.on("agent_end", async (_event, ctx) => {
    const config = await loadConfig();
    await logEvent(config, "agent_end", {
      currentModel: getCurrentModel(ctx),
    });
  });
}
