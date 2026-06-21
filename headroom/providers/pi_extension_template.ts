import type {
  ExtensionAPI,
  ExtensionContext,
} from "@earendil-works/pi-coding-agent";
import { githubCopilotOAuthProvider } from "@earendil-works/pi-ai/oauth";
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

interface SessionUiConfig {
  enableStatusCommand?: boolean;
  enableFooter?: boolean;
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
  ui?: SessionUiConfig;
  autoManageCurrentProviderOnly?: boolean;
  controlUrl?: string;
  phase0?: Phase0Config;
}

interface ModelSnapshot {
  provider: string | null;
  id: string;
}

interface ProviderModel {
  provider: string;
  id?: string;
  baseUrl?: string;
  [key: string]: unknown;
}

type CopilotOAuthProvider = {
  login?: (...args: unknown[]) => unknown;
  refreshToken?: (...args: unknown[]) => unknown;
  getApiKey?: (...args: unknown[]) => unknown;
  modifyModels?: (models: ProviderModel[], credentials: unknown) => ProviderModel[];
};

type HealthStatus = "healthy" | "suspect" | "unavailable";
type ResolutionSource = "runtime-provider" | "model-id" | "unresolved";

interface PerfSummary {
  requestCount: number;
  tokensSaved: number;
  savingsUsd?: number;
  savingsPercent: number;
  basis: "history" | "runtime";
}

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
const STATUS_SLOT = "headroom-wrap-pi";
const STATUS_COMMAND = "headroom-status";

export default function (pi: ExtensionAPI) {
  const registeredProviders = new Map<string, string>();
  const providerHealth = new Map<string, ProviderHealthState>();
  const providerPerf = new Map<string, PerfSummary>();

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

  const dashboardUrl = (managedConfig: SessionProviderConfig): string =>
    `${managedConfig.rootUrl}/dashboard`;

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

  const numberOrZero = (value: unknown): number =>
    typeof value === "number" && Number.isFinite(value) ? value : 0;

  const numberOrUndefined = (value: unknown): number | undefined =>
    typeof value === "number" && Number.isFinite(value) ? value : undefined;

  const formatCompactMetric = (value: number): string => {
    const abs = Math.abs(value);
    if (abs >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(1)}B`;
    if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
    if (abs >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
    return String(Math.round(value));
  };

  const formatSavingsPercent = (value: number): string => `${Math.round(value)}%`;

  const formatCompactUsd = (value: number): string => {
    const abs = Math.abs(value);
    if (abs >= 1_000_000_000) return `$${(value / 1_000_000_000).toFixed(1)}B`;
    if (abs >= 1_000_000) return `$${(value / 1_000_000).toFixed(1)}M`;
    if (abs >= 1_000) return `$${(value / 1_000).toFixed(1)}k`;
    return `$${value.toFixed(1)}`;
  };

  const deriveSavingsPercent = (tokensSaved: number, totalInputTokens: number): number => {
    const totalBefore = Math.max(0, totalInputTokens) + Math.max(0, tokensSaved);
    return totalBefore > 0 ? (tokensSaved / totalBefore) * 100 : 0;
  };

  const parsePerfSummary = (payload: unknown): PerfSummary | undefined => {
    if (!payload || typeof payload !== "object") return undefined;

    const data = payload as {
      lifetime?: {
        requests?: unknown;
        tokens_saved?: unknown;
        compression_savings_usd?: unknown;
        total_input_tokens?: unknown;
      };
      persistent_savings?: {
        lifetime?: {
          requests?: unknown;
          tokens_saved?: unknown;
          compression_savings_usd?: unknown;
          total_input_tokens?: unknown;
        };
      };
      requests?: { total?: unknown };
      tokens?: { saved?: unknown; savings_percent?: unknown };
      lifetime_stats?: {
        total_requests?: unknown;
        total_tokens_saved?: unknown;
        total_input_tokens?: unknown;
        total_estimated_savings_usd?: unknown;
      };
      total_requests?: unknown;
      total_tokens_saved?: unknown;
      total_input_tokens?: unknown;
      total_estimated_savings_usd?: unknown;
    };

    const lifetime =
      data.lifetime ?? data.persistent_savings?.lifetime ?? data.lifetime_stats;

    if (lifetime && typeof lifetime === "object") {
      const requestCount = numberOrZero(
        "requests" in lifetime ? lifetime.requests : lifetime.total_requests,
      );
      const tokensSaved = numberOrZero(
        "tokens_saved" in lifetime ? lifetime.tokens_saved : lifetime.total_tokens_saved,
      );
      const totalInputTokens = numberOrZero(
        "total_input_tokens" in lifetime ? lifetime.total_input_tokens : undefined,
      );
      const savingsUsd = numberOrUndefined(
        "compression_savings_usd" in lifetime
          ? lifetime.compression_savings_usd
          : lifetime.total_estimated_savings_usd,
      );

      if (requestCount > 0 || tokensSaved > 0 || typeof savingsUsd === "number") {
        return {
          requestCount,
          tokensSaved,
          savingsUsd,
          savingsPercent: deriveSavingsPercent(tokensSaved, totalInputTokens),
          basis: "history",
        };
      }
    }

    const requestCount = numberOrZero(data.requests?.total ?? data.total_requests);
    const tokensSaved = numberOrZero(data.tokens?.saved ?? data.total_tokens_saved);
    const explicitSavingsPercent = numberOrUndefined(data.tokens?.savings_percent);
    const totalInputTokens = numberOrZero(data.total_input_tokens);
    const savingsUsd = numberOrUndefined(data.total_estimated_savings_usd);

    if (
      requestCount === 0 &&
      tokensSaved === 0 &&
      explicitSavingsPercent === undefined &&
      savingsUsd === undefined
    ) {
      return undefined;
    }

    return {
      requestCount,
      tokensSaved,
      savingsUsd,
      savingsPercent:
        explicitSavingsPercent ?? deriveSavingsPercent(tokensSaved, totalInputTokens),
      basis: "runtime",
    };
  };

  const ensureManagedProvider = async (
    config: SessionConfig,
    providerId: string,
  ): Promise<SessionConfig> => {
    if (!config.autoManageCurrentProviderOnly || !config.controlUrl || config.providers[providerId]) {
      return config;
    }
    try {
      const response = await fetch(`${config.controlUrl}/ensure-provider`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ providerId }),
      });
      if (!response.ok) {
        return config;
      }
      return await loadConfig();
    } catch {
      return config;
    }
  };

  const refreshPerfSummary = async (
    providerId: string,
    managedConfig: SessionProviderConfig,
  ): Promise<void> => {
    const urls = [
      `${managedConfig.rootUrl}/stats?cached=1`,
      `${managedConfig.rootUrl}/stats-history`,
      `${managedConfig.rootUrl}/stats`,
    ];

    for (const url of urls) {
      try {
        const response = await fetch(url);
        if (!response.ok) continue;
        const payload = (await response.json()) as unknown;
        const summary = parsePerfSummary(payload);
        if (!summary) continue;
        providerPerf.set(providerId, summary);
        return;
      } catch {
        // Footer metrics are best-effort only.
      }
    }
  };

  const shortProviderLabel = (providerId: string): string => {
    if (providerId === "github-copilot") {
      return "copilot";
    }
    return providerId;
  };

  const activeStatusLine = (
    config: SessionConfig,
    currentModel: ModelSnapshot | null,
  ): string => {
    const resolution = getResolution(currentModel);
    const providerId = resolution.providerId;
    if (!providerId || !config.providers[providerId]) {
      return "Headroom:off";
    }

    const label = shortProviderLabel(providerId);
    const perf = providerPerf.get(providerId);
    const usdSuffix =
      typeof perf?.savingsUsd === "number"
        ? ` | ${formatCompactUsd(perf.savingsUsd)}`
        : "";
    const perfSuffix = perf
      ? ` | ${formatCompactMetric(perf.tokensSaved)} tok${usdSuffix} | ${formatSavingsPercent(perf.savingsPercent)}`
      : "";

    if (registeredProviders.has(providerId)) {
      return `Headroom:${label} ↑${perfSuffix}`;
    }
    const healthState = providerHealth.get(providerId);
    if (healthState?.status === "unavailable") {
      return `Headroom:${label} down${perfSuffix}`;
    }
    return `Headroom:${label} idle${perfSuffix}`;
  };

  const updateUiStatus = (
    config: SessionConfig,
    ctx: ExtensionContext,
    currentModel: ModelSnapshot | null,
  ): void => {
    if (!config.ui?.enableFooter) {
      return;
    }
    try {
      ctx.ui?.setStatus?.(STATUS_SLOT, activeStatusLine(config, currentModel));
    } catch {
      // Footer status is best-effort only.
    }
  };

  const notifyUi = (
    ctx: ExtensionContext,
    message: string,
    level: "info" | "warn" | "error" = "info",
  ): void => {
    try {
      ctx.ui?.notify?.(message, level);
    } catch {
      // Command output is best-effort only.
    }
  };

  const statusLines = (
    config: SessionConfig,
    currentModel: ModelSnapshot | null,
  ): string[] => {
    const resolution = getResolution(currentModel);
    const providerId = resolution.providerId;
    if (!providerId || !config.providers[providerId]) {
      return [
        "Headroom: off",
        "No managed provider is currently selected.",
      ];
    }

    const managedConfig = config.providers[providerId];
    const perf = providerPerf.get(providerId);
    const healthState = providerHealth.get(providerId);
    const attached = registeredProviders.has(providerId);
    return [
      `Provider: ${providerId}`,
      `Status: ${attached ? "running" : healthState?.status ?? "idle"}`,
      `Routed base URL: ${managedConfig.routedBaseUrl}`,
      `Proxy root URL: ${managedConfig.rootUrl}`,
      `Dashboard: ${dashboardUrl(managedConfig)}`,
      `Lifetime tokens saved: ${perf ? formatCompactMetric(perf.tokensSaved) : "(unavailable)"}`,
      `Lifetime compression savings: ${typeof perf?.savingsUsd === "number" ? formatCompactUsd(perf.savingsUsd) : "(unavailable)"}`,
      `Savings percent: ${perf ? formatSavingsPercent(perf.savingsPercent) : "(unavailable)"}`,
      `Footer: ${activeStatusLine(config, currentModel)}`,
    ];
  };

  const registerCommands = (piApi: ExtensionAPI): void => {
    piApi.registerCommand(STATUS_COMMAND, {
      description:
        "Show Headroom routing, proxy status, lifetime savings, and dashboard URL for the current managed provider.",
      handler: async (_args: string, ctx: ExtensionContext) => {
        const config = await loadConfig();
        const currentModel = getCurrentModel(ctx);
        const resolution = getResolution(currentModel);
        const providerId = resolution.providerId;
        if (providerId && config.providers[providerId]) {
          await syncCurrentProvider(config, currentModel, "headroom_status_command");
        }
        updateUiStatus(config, ctx, currentModel);
        notifyUi(ctx, statusLines(config, currentModel).join("\n"), "info");
      },
    });
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

    if (providerId === "github-copilot") {
      const oauthProvider = githubCopilotOAuthProvider as CopilotOAuthProvider;
      pi.registerProvider("github-copilot", {
        baseUrl: managedConfig.routedBaseUrl,
        oauth: {
          name: "GitHub Copilot via Headroom",
          login: oauthProvider.login,
          refreshToken: oauthProvider.refreshToken,
          getApiKey: oauthProvider.getApiKey,
          modifyModels(models: ProviderModel[], credentials: unknown) {
            const oauthAdjusted = oauthProvider.modifyModels
              ? oauthProvider.modifyModels(models, credentials)
              : models;
            return oauthAdjusted.map((model) =>
              model.provider === "github-copilot"
                ? { ...model, baseUrl: managedConfig.routedBaseUrl }
                : model,
            );
          },
        },
      });
    } else {
      pi.registerProvider(providerId, { baseUrl: managedConfig.routedBaseUrl });
    }

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

    let workingConfig = config;
    if (workingConfig.managedProviders.includes(providerId) && !workingConfig.providers[providerId]) {
      workingConfig = await ensureManagedProvider(workingConfig, providerId);
    }

    const managedConfig = workingConfig.providers[providerId];
    if (!managedConfig) {
      await unregisterProvider(config, providerId, `${reason}:unmanaged-provider`);
      return;
    }

    if (workingConfig.phase0?.forceNativeProviders?.includes(providerId)) {
      await unregisterProvider(workingConfig, providerId, `${reason}:forced-native`);
      return;
    }

    const healthState = await refreshProviderHealth(workingConfig, providerId, reason);
    const shouldAttach =
      healthState.status === "healthy" ||
      (healthState.status === "suspect" && (healthState.hasEverAttached || registeredProviders.has(providerId)));

    if (!shouldAttach) {
      await unregisterProvider(workingConfig, providerId, `${reason}:proxy-${healthState.status}`);
      return;
    }

    await registerProvider(
      workingConfig,
      providerId,
      managedConfig,
      reason,
      currentModel,
      resolution.source,
      healthState,
    );
    await refreshPerfSummary(providerId, managedConfig);
  };

  registerCommands(pi);

  pi.on("session_start", async (event, ctx) => {
    const config = await loadConfig();
    const currentModel = getCurrentModel(ctx);
    await logEvent(config, "session_start", {
      reason: event.reason,
      currentModel,
      sessionConfigPath: process.env.HEADROOM_PI_SESSION_CONFIG,
    });
    await syncCurrentProvider(config, currentModel, "session_start");
    updateUiStatus(config, ctx, currentModel);
  });

  pi.on("model_select", async (event, ctx) => {
    const config = await loadConfig();
    const currentModel = getModelSnapshot(event.model);
    await logEvent(config, "model_select", {
      source: event.source,
      currentModel,
      previousModel: getModelSnapshot(event.previousModel),
    });
    await syncCurrentProvider(config, currentModel, "model_select");
    updateUiStatus(config, ctx, currentModel);
  });

  pi.on("before_agent_start", async (_event, ctx) => {
    const config = await loadConfig();
    const currentModel = getCurrentModel(ctx);
    await syncCurrentProvider(config, currentModel, "before_agent_start");
    updateUiStatus(config, ctx, currentModel);
  });

  pi.on("agent_end", async (_event, ctx) => {
    const config = await loadConfig();
    const currentModel = getCurrentModel(ctx);
    await logEvent(config, "agent_end", {
      currentModel,
    });
    updateUiStatus(config, ctx, currentModel);
  });
}
