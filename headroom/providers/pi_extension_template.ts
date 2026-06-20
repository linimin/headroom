import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { promises as fs } from "node:fs";

interface SessionProviderConfig {
  port: number;
  rootUrl: string;
  routedBaseUrl: string;
  family: string;
}

interface Phase0Config {
  logPath?: string;
  forceNativeProviders?: string[];
}

interface SessionConfig {
  version: number;
  managedProviders: string[];
  providers: Record<string, SessionProviderConfig>;
  phase0?: Phase0Config;
}

interface ModelSnapshot {
  provider: string;
  id: string;
}

export default function (pi: ExtensionAPI) {
  const registeredProviders = new Set<string>();

  const getCurrentModel = (ctx: ExtensionContext): ModelSnapshot | null => {
    if (!ctx.model) {
      return null;
    }
    return { provider: ctx.model.provider, id: ctx.model.id };
  };

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

  const syncCurrentProvider = async (ctx: ExtensionContext, reason: string): Promise<void> => {
    const config = await loadConfig();
    const currentModel = getCurrentModel(ctx);

    await logEvent(config, "provider_observed", {
      reason,
      currentModel,
    });

    if (!currentModel) {
      return;
    }

    for (const providerId of [...registeredProviders]) {
      if (providerId !== currentModel.provider) {
        await unregisterProvider(config, providerId, `switched:${currentModel.provider}`);
      }
    }

    const managedConfig = config.providers[currentModel.provider];
    if (!managedConfig) {
      await unregisterProvider(config, currentModel.provider, "unmanaged-provider");
      return;
    }

    if (config.phase0?.forceNativeProviders?.includes(currentModel.provider)) {
      await unregisterProvider(config, currentModel.provider, "forced-native");
      return;
    }

    pi.registerProvider(currentModel.provider, { baseUrl: managedConfig.routedBaseUrl });
    registeredProviders.add(currentModel.provider);
    await logEvent(config, "provider_registered", {
      providerId: currentModel.provider,
      baseUrl: managedConfig.routedBaseUrl,
      reason,
      currentModel,
    });
  };

  pi.on("session_start", async (event, ctx) => {
    const config = await loadConfig();
    await logEvent(config, "session_start", {
      reason: event.reason,
      currentModel: getCurrentModel(ctx),
      sessionConfigPath: process.env.HEADROOM_PI_SESSION_CONFIG,
    });
  });

  pi.on("model_select", async (event) => {
    const config = await loadConfig();
    await logEvent(config, "model_select", {
      source: event.source,
      currentModel: { provider: event.model.provider, id: event.model.id },
      previousModel: event.previousModel
        ? { provider: event.previousModel.provider, id: event.previousModel.id }
        : null,
    });
  });

  pi.on("before_agent_start", async (_event, ctx) => {
    await syncCurrentProvider(ctx, "before_agent_start");
  });

  pi.on("agent_end", async (_event, ctx) => {
    const config = await loadConfig();
    await logEvent(config, "agent_end", {
      currentModel: getCurrentModel(ctx),
    });
  });
}
