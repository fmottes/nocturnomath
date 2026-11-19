import { elements, state } from "./state.js?v=20261117-2";
import { initWebSocket, sendWs } from "./transport.js?v=20261016-3";
import { openWorkspace, applyWorkspace, browseFolders, closeFolderPicker, loadDocument, loadPlots, openFolderPicker, rawModelId, resolveCatalogueModel, setActiveModelAndEffort, setAuthStatus, setContextUsage, setEffortForModel, setExplorationPrompt, setModelCatalogue, setPromptMode, setSessionDefaults, selectViewerTab, refreshDocuments, setPlotsFilter, togglePlotsFilterMenu, setSessionId } from "./workspace.js?v=20261118-2";
import { appendAssistantChunk, appendAssistantDelta, appendDocumentNotification, appendErrorMessage, appendNoteNotification, appendProbeFinish, appendProbeStart, appendProbeVerdict, appendSystemMessage, appendUserMessage, finalizeAssistantTurn } from "./chat.js?v=20261118-1";
import { clearChat, loadSessions, replaySession, restoreChat } from "./history.js?v=20261118-1";
import { refreshSendButton, updateAgentStatus, updateKernelStatus } from "./status.js?v=20261118-1";
import { closeLightbox, cycleLightbox, isLightboxOpen } from "./ui.js?v=20261016-3";
import { cancelCreateFolder, cancelEdit, openDocument, saveEdit, saveFolder, setDocumentsDefault, showDocumentsView, startCreate, startCreateFolder, startModify } from "./documents.js?v=20261118-2";

const THEME_KEY = "nocturnomath.theme";

function applyTheme(theme) {
  const selected = theme === "dark" ? "dark" : "light";
  document.documentElement.dataset.theme = selected;
  document.querySelectorAll("[data-theme-toggle]").forEach((button) => {
    const next = selected === "dark" ? "Light" : "Dark";
    button.textContent = next;
    button.title = `Use ${next.toLowerCase()} mode`;
    button.setAttribute("aria-label", `Use ${next.toLowerCase()} mode`);
  });
  try {
    window.localStorage.setItem(THEME_KEY, selected);
  } catch (error) {
    // The theme still applies for this page when storage is unavailable.
  }
  window.dispatchEvent(new CustomEvent("nocturnomath:themechange"));
}

function explorerLabel(agent) {
  return agent.session_id || "New";
}

function updateCompactButton() {
  const agent = state.agents.get(state.activeAgentId);
  elements.btnCompact.disabled = !agent?.can_compact || agent.is_busy;
}

function renderExplorerTabs() {
  if (!elements.explorerTabs) return;
  elements.explorerTabs.querySelectorAll(".explorer-tab").forEach((node) => node.remove());
  [...state.agents.values()].forEach((agent) => {
    const tab = document.createElement("div");
    tab.className = `explorer-tab${agent.agent_id === state.activeAgentId ? " active" : ""}${agent.is_busy ? " busy" : ""}`;
    tab.setAttribute("role", "tab");
    tab.tabIndex = 0;
    tab.setAttribute("aria-selected", String(agent.agent_id === state.activeAgentId));
    tab.title = `Explorer ${explorerLabel(agent)}`;
    const title = document.createElement("span");
    title.className = "pane-title";
    const label = document.createElement("span");
    label.textContent = "Explorer";
    const badge = document.createElement("span");
    badge.className = "badge badge-sm";
    badge.textContent = explorerLabel(agent);
    title.append(label, badge);
    const status = document.createElement("span");
    status.className = "explorer-tab-status";
    status.title = agent.is_busy ? "Running" : "Idle";
    const close = document.createElement("button");
    close.type = "button";
    close.className = "btn btn-xs btn-ghost explorer-close";
    close.textContent = "×";
    close.title = agent.is_busy ? "Explorer is running" : "Close explorer";
    close.disabled = agent.is_busy || state.agents.size === 1;
    close.addEventListener("click", async (event) => {
      event.stopPropagation();
      await closeExplorer(agent.agent_id);
    });
    tab.append(title, status, close);
    tab.addEventListener("click", () => selectExplorer(agent.agent_id));
    tab.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectExplorer(agent.agent_id);
      }
    });
    elements.explorerTabs.insertBefore(tab, elements.btnNewExplorer);
  });
  updateCompactButton();
}

function saveExplorerView() {
  const agent = state.agents.get(state.activeAgentId);
  if (!agent) return;
  agent.draftText = elements.promptInput.value;
  agent.scrollTop = elements.chatMessages.scrollTop;
  agent.followChat = state.followChat;
}

function restoreExplorerView(agent) {
  elements.promptInput.value = agent.draftText || "";
  elements.promptInput.style.height = "44px";
  elements.promptInput.style.height = Math.min(elements.promptInput.scrollHeight, 160) + "px";
  state.followChat = agent.followChat ?? true;
  elements.chatMessages.scrollTop = state.followChat
    ? elements.chatMessages.scrollHeight
    : (agent.scrollTop || 0);
}

function renderPendingPrompts(prompts = []) {
  if (!elements.pendingPrompts) return;
  elements.pendingPrompts.classList.toggle("hidden", !prompts.length);
  elements.pendingPrompts.replaceChildren(...prompts.map((item) => {
    const row = document.createElement("div");
    row.className = "pending-prompt-item";

    const mode = document.createElement("span");
    mode.className = "pending-prompt-mode";
    mode.textContent = item.mode === "queue" ? "Queue" : "Steer";

    const text = document.createElement("span");
    text.className = "pending-prompt-text";
    text.textContent = item.text;
    text.title = item.text;

    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.className = "btn btn-xs btn-ghost pending-prompt-cancel";
    cancel.textContent = "×";
    cancel.title = "Cancel pending prompt";
    cancel.setAttribute("aria-label", "Cancel pending prompt");
    cancel.addEventListener("click", () => {
      sendWs("cancel_pending_prompt", { agent_id: state.activeAgentId, prompt_id: item.id });
    });

    row.append(mode, text, cancel);
    return row;
  }));
}

function applyExplorer(agent, repaint = false) {
  if (!agent?.agent_id) return;
  const prior = state.agents.get(agent.agent_id) || {};
  const merged = {
    ...prior,
    ...agent,
    modelProvider: agent.model !== undefined
      ? (state.modelProviders[agent.model] || agent.modelProvider || prior.modelProvider || null)
      : (prior.modelProvider || null),
  };
  let resolvedDraft = null;
  if (merged.draftModel || merged.draftRawModel) {
    resolvedDraft = resolveCatalogueModel(
      merged.draftRawModel || merged.draftModel,
      merged.draftProvider,
    );
    if (resolvedDraft) {
      merged.draftModel = resolvedDraft;
      merged.draftProvider = state.modelProviders[resolvedDraft] || merged.draftProvider || null;
      merged.draftRawModel = rawModelId(resolvedDraft, merged.draftProvider);
    } else {
      merged.draftModel = null;
      merged.draftProvider = null;
      merged.draftRawModel = null;
      merged.draftEffort = null;
    }
  }
  state.agents.set(agent.agent_id, merged);
  if (agent.agent_id === state.activeAgentId) {
    setSessionId(merged.session_id);
    setActiveModelAndEffort(
      resolvedDraft || merged.model,
      resolvedDraft ? (merged.draftEffort || merged.effort) : merged.effort,
    );
    setContextUsage(merged.context_usage);
    updateKernelStatus(merged.kernel_alive, merged.kernel_busy);
    updateAgentStatus(merged.is_busy && (!merged.status || merged.status === "idle")
      ? "thinking" : (merged.status || "idle"));
    renderPendingPrompts(merged.pending_prompts || []);
    if (agent.documents) refreshDocuments();
    if (repaint) {
      restoreChat(merged.records || [], merged.kernel_busy);
      restoreExplorerView(merged);
    }
  }
  renderExplorerTabs();
}

let explorerSelectSeq = 0;

async function selectExplorer(agentId) {
  if (!state.agents.has(agentId) || agentId === state.activeAgentId) return;
  const seq = ++explorerSelectSeq;
  saveExplorerView();
  state.activeAgentId = agentId;
  const cached = state.agents.get(agentId);
  applyExplorer(cached, true);
  try {
    const response = await fetch(`/api/agents/${encodeURIComponent(agentId)}`);
    if (!response.ok) throw new Error("Explorer no longer exists");
    const snapshot = await response.json();
    if (state.activeAgentId === agentId && seq === explorerSelectSeq) {
      saveExplorerView();
      applyExplorer(snapshot, true);
    }
    loadPlots();
  } catch (error) {
    appendErrorMessage(error.message);
  }
}

function installExplorers(agents = [], preserveView = true) {
  if (preserveView) saveExplorerView();
  const previous = preserveView ? state.agents : new Map();
  state.agents = new Map(agents.map((agent) => {
    const prior = previous.get(agent.agent_id) || {};
    return [
      agent.agent_id,
      {
        ...prior,
        ...agent,
        modelProvider: agent.model
          ? (state.modelProviders[agent.model] || prior.modelProvider || null)
          : null,
      },
    ];
  }));
  if (!state.agents.has(state.activeAgentId)) state.activeAgentId = agents[0]?.agent_id || null;
  if (state.activeAgentId) applyExplorer(state.agents.get(state.activeAgentId), true);
  renderExplorerTabs();
}

async function refreshContextUsage(agentId) {
  const seqBefore = state.agents.get(agentId)?.stateSeq || 0;
  try {
    const response = await fetch(`/api/agents/${encodeURIComponent(agentId)}`);
    if (!response.ok) return;
    const agent = await response.json();
    const current = state.agents.get(agentId);
    if (current && (current.stateSeq || 0) !== seqBefore) {
      delete agent.is_busy;
      delete agent.status;
      delete agent.kernel_busy;
    }
    applyExplorer(agent);
  } catch (error) {
    // The next websocket snapshot will reconcile an unavailable local service.
  }
}

async function createExplorer() {
  elements.btnNewExplorer.disabled = true;
  try {
    const response = await fetch("/api/agents", { method: "POST" });
    const agent = await response.json();
    if (!response.ok) throw new Error(agent.detail || "Unable to start explorer");
    applyExplorer(agent);
    await selectExplorer(agent.agent_id);
  } catch (error) {
    appendErrorMessage(error.message);
  } finally {
    elements.btnNewExplorer.disabled = false;
  }
}

async function closeExplorer(agentId) {
  try {
    const response = await fetch(`/api/agents/${encodeURIComponent(agentId)}`, { method: "DELETE" });
    if (response.status === 404 && state.agents.size > 1) {
      handleServerEvent({ type: "agent_removed", agent_id: agentId });
      return;
    }
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Unable to close explorer");
  } catch (error) {
    appendErrorMessage(error.message);
  }
}

async function compactExplorer() {
  const agentId = state.activeAgentId;
  if (!agentId || elements.btnCompact.disabled) return;
  elements.btnCompact.disabled = true;
  try {
    const response = await fetch(`/api/agents/${encodeURIComponent(agentId)}/compact`, { method: "POST" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Unable to compact explorer");
    await refreshContextUsage(agentId);
  } catch (error) {
    await refreshContextUsage(agentId);
    if (state.activeAgentId === agentId) appendErrorMessage(error.message);
  } finally {
    updateCompactButton();
  }
}

// ============================================================================
// Server Event Dispatcher
// ============================================================================
function handleServerEvent(event) {
  if (event.type === "agent_added") {
    applyExplorer(event);
    if (event.requested_by === state.activeAgentId) selectExplorer(event.agent_id);
    return;
  }
  if (event.type === "agent_focused") {
    selectExplorer(event.agent_id);
    return;
  }
  if (event.agent_id && event.type !== "agent_removed") {
    const current = state.agents.get(event.agent_id);
    if (!current) {
      if (event.type === "error") appendErrorMessage(event.message);
      else if (event.type === "system_message") appendSystemMessage(event.text);
      if (!["session_resumed", "session_reset"].includes(event.type)) return;
    } else {
      if (event.type === "status_change") {
        current.is_busy = !["idle", "cancelled"].includes(event.status);
        current.status = event.status;
        current.stateSeq = (current.stateSeq || 0) + 1;
      }
      if (event.type === "probe_start" && event.kernel_alive !== undefined && event.backend !== "slurm") {
        current.kernel_alive = event.kernel_alive;
        current.kernel_busy = true;
        if (current.is_busy) current.status = "probing";
        current.stateSeq = (current.stateSeq || 0) + 1;
      }
      if (event.type === "probe_finish" && event.backend !== "slurm" && event.status !== "slurm-error") {
        current.kernel_busy = false;
        if (current.is_busy && current.status === "probing") current.status = "thinking";
        current.stateSeq = (current.stateSeq || 0) + 1;
      }
      if (event.type === "kernel_restarted" || event.type === "kernel_interrupted") {
        current.kernel_alive = true;
        current.kernel_busy = false;
        current.stateSeq = (current.stateSeq || 0) + 1;
      }
      if (event.type === "context_usage_changed") current.context_usage = event.context_usage;
      if (event.type === "pending_prompts_changed") current.pending_prompts = event.pending_prompts || [];
      if (event.type === "user_message") {
        if (event.session_id) current.session_id = event.session_id;
        if (event.model) {
          current.model = event.model;
          current.modelProvider = state.modelProviders[event.model] || current.modelProvider || null;
        }
        if (event.effort !== undefined) current.effort = event.effort;
      }
      state.agents.set(event.agent_id, current);
      renderExplorerTabs();
      if (
        event.agent_id !== state.activeAgentId
        && !["agent_focused", "session_reset", "session_resumed"].includes(event.type)
      ) {
        if (event.type === "record_changed" && state.activeTab !== "plots") {
          loadDocument(state.activeDoc, "", true);
        }
        if (event.type === "status_change" && event.status === "idle") {
          refreshContextUsage(event.agent_id);
        }
        return;
      }
    }
  }
  switch (event.type) {
    case "service_stopping":
      state.exiting = true;
      elements.btnSend.disabled = true;
      appendSystemMessage("Nocturnomath is stopping. You can close this tab.");
      break;
    case "init":
      applyWorkspace(event.workspace);
      if (event.workspace?.is_open) installExplorers(event.workspace.agents || []);
      break;
    case "workspace_updated":
      clearChat();
      state.activeAgentId = null;
      applyWorkspace(event.workspace);
      installExplorers(event.workspace.agents || [], false);
      break;

    case "agent_removed": {
      const removedActive = event.agent_id === state.activeAgentId;
      state.agents.delete(event.agent_id);
      if (removedActive) {
        state.activeAgentId = null;
        const nextId = state.agents.keys().next().value || null;
        if (nextId) selectExplorer(nextId);
      }
      renderExplorerTabs();
      break;
    }

    case "auth_changed": {
      applyAuthChange(event);
      const changed = event.provider ? event.auth?.providers?.[event.provider] : null;
      const summary = changed
        ? `${changed.title} authentication changed to ${changed.label}.`
        : `Authentication changed to ${event.auth?.label || "the selected method"}.`;
      appendSystemMessage(summary);
      break;
    }

    case "user_message": {
      const agent = state.agents.get(event.agent_id);
      if (agent && (event.model !== undefined || event.effort !== undefined) && !agent.draftModel && !agent.draftRawModel) {
        setActiveModelAndEffort(agent.model, agent.effort);
      }
      if (event.session_id && agent) {
        agent.session_id = event.session_id;
        setSessionId(event.session_id);
        renderExplorerTabs();
      }
      appendUserMessage(event.text);
      break;
    }

    case "probe_start":
      appendProbeStart(event.expected, event.code, event.probe_id, event.backend);
      if (event.kernel_alive !== undefined && event.backend !== "slurm") {
        updateKernelStatus(event.kernel_alive, true);
        if (state.agents.get(event.agent_id)?.is_busy) updateAgentStatus("probing");
      }
      break;

    case "probe_update":
    case "probe_finish":
      // Preserve the web app's existing probe card contract: `output` includes
      // source paths and any guidance emitted alongside the raw kernel text.
      appendProbeFinish(event.expected, event.code, event.output, event.plot_urls, event.plot_images, event.probe_id, event.status, event.backend);
      if (event.type === "probe_finish") {
        if (event.backend !== "slurm" && event.status !== "slurm-error") {
          const current = state.agents.get(event.agent_id);
          updateKernelStatus(current?.kernel_alive ?? true, false);
          if (current?.is_busy) updateAgentStatus(current.status || "thinking");
        }
        loadPlots();
      }
      break;

    case "record_changed":
      appendNoteNotification(event.kind, `${event.entry_id}: ${event.text}`);
      if (state.activeTab !== "plots" && [state.workspace?.evidence_path, state.workspace?.thoughts_path].includes(state.activeDoc)) {
        loadDocument(state.activeDoc, "", true);
      }
      break;

    case "document_saved":
      appendDocumentNotification(event.name, event.action);
      refreshDocuments();
      break;

    case "probe_verdict":
      appendProbeVerdict(event.text, event.probe_id);
      break;

    case "slurm_review":
      appendSystemMessage(`Reviewing finished Slurm probe ${event.probe_id}.`);
      break;

    case "assistant_delta":
      appendAssistantDelta(event.text);
      break;

    case "assistant_text":
      appendAssistantChunk(event.text);
      break;

    case "turn_complete":
      finalizeAssistantTurn();
      refreshDocuments();
      break;

    case "status_change": {
      const agent = state.agents.get(event.agent_id);
      if (agent) {
        agent.is_busy = !["idle", "cancelled"].includes(event.status);
        if (
          ["cancelled", "idle"].includes(event.status) &&
          !agent.draftModel &&
          !agent.draftRawModel
        ) {
          setActiveModelAndEffort(agent.model, agent.effort);
        }
      }
      updateAgentStatus(event.status);
      renderExplorerTabs();
      if (event.status === "idle") {
        refreshContextUsage(event.agent_id);
      }
      break;
    }

    case "kernel_restarted":
      appendSystemMessage("Kernel restarted. In-memory variables and state cleared.");
      updateKernelStatus(true, false);
      break;

    case "kernel_interrupted":
      appendSystemMessage("Kernel interrupted by user.");
      updateKernelStatus(true, false);
      break;

    case "context_usage_changed":
      setContextUsage(event.context_usage);
      break;

    case "context_compacted":
      appendSystemMessage(
        event.record_reloaded === false
          ? "This Explorer's conversation was compacted. The latest scientific record will be attached on the next message."
          : "This Explorer's conversation was compacted. The latest scientific record was reloaded before continuing."
      );
      break;

    case "session_reset": {
      const targetId = event.agent_id || state.activeAgentId;
      if (state.agents.has(targetId)) {
        Object.assign(state.agents.get(targetId), {
          context_usage: null,
          session_id: event.session_id ?? null,
          records: [],
          can_compact: false,
          pending_prompts: [],
          model: event.model,
          modelProvider: state.modelProviders[event.model] || null,
          effort: event.effort,
          draftModel: null,
          draftProvider: null,
          draftRawModel: null,
          draftEffort: null,
        });
      }
      renderExplorerTabs();
      if (!event.agent_id || event.agent_id === state.activeAgentId) {
        renderPendingPrompts([]);
        setContextUsage(null);
        setSessionId(event.session_id ?? null);
        setActiveModelAndEffort(event.model, event.effort);
        clearChat();
        updateKernelStatus(true, false);
        updateCompactButton();
      }
      break;
    }

    case "session_defaults_changed":
      setSessionDefaults(event.session_defaults);
      break;

    case "exploration_prompt_changed":
      setExplorationPrompt(event.exploration_prompt, event.exploration_prompt_is_custom);
      break;

    case "prompt_mode_changed":
      setPromptMode(event.prompt_mode);
      break;

    case "pending_prompts_changed":
      renderPendingPrompts(event.pending_prompts || []);
      break;

    case "session_resumed": {
      ++explorerSelectSeq;
      const resumedId = event.session_id || event.id || null;
      if (state.agents.has(event.agent_id)) {
        Object.assign(state.agents.get(event.agent_id), {
          session_id: resumedId,
          can_compact: event.can_compact ?? Boolean(event.context_restored),
          draftText: "",
          scrollTop: 0,
          followChat: true,
          draftModel: null,
          draftProvider: null,
          draftRawModel: null,
          draftEffort: null,
        });
      }
      applyExplorer({
        is_busy: false,
        status: "idle",
        kernel_busy: false,
        ...event,
        session_id: resumedId,
        can_compact: event.can_compact ?? Boolean(event.context_restored),
      });
      if (!event.agent_id || event.agent_id === state.activeAgentId) {
        restoreExplorerView(state.agents.get(event.agent_id));
        setSessionId(resumedId);
        replaySession(
          event.records || [],
          event.context_restored,
          event.kernel_reset,
          event.kernel_restored,
          event.probes_replayed
        );
      }
      break;
    }

    case "system_message":
      appendSystemMessage(event.text);
      break;

    case "error":
      appendErrorMessage(event.message);
      break;

    default:
      console.log("Unhandled event:", event);
  }
}

// ============================================================================
// Event Listeners & Setup
// ============================================================================
function setupEventListeners() {
  window.addEventListener("nocturnomath:focus-explorer", (event) => selectExplorer(event.detail));
  window.addEventListener("nocturnomath:workspace-opened", (event) => installExplorers(event.detail?.agents || [], false));
  elements.chatMessages.addEventListener("scroll", () => {
    const log = elements.chatMessages;
    state.followChat = log.scrollHeight - log.clientHeight - log.scrollTop < 60;
    const agent = state.agents.get(state.activeAgentId);
    if (agent) {
      agent.scrollTop = log.scrollTop;
      agent.followChat = state.followChat;
    }
  }, { passive: true });
  elements.btnSettings.addEventListener("click", () => elements.settingsModal.classList.remove("hidden"));
  elements.btnCompact.addEventListener("click", compactExplorer);
  [elements.btnAuth, elements.btnAuthLanding, elements.btnAuthSettings].forEach((button) => {
    button.addEventListener("click", openAuthModal);
  });
  elements.authProvider.addEventListener("change", () => {
    elements.authCredential.value = "";
    elements.authError.textContent = "";
    elements.authError.classList.add("hidden");
    populateAuthMethods(elements.authProvider.value);
    updateAuthForm();
  });
  elements.authMethod.addEventListener("change", updateAuthForm);
  elements.authModalClose.addEventListener("click", closeAuthModal);
  elements.authModalCancel.addEventListener("click", closeAuthModal);
  elements.authModal.addEventListener("click", (event) => {
    if (event.target === elements.authModal) closeAuthModal();
  });
  elements.authForm.addEventListener("submit", submitAuth);
  elements.settingsClose.addEventListener("click", () => elements.settingsModal.classList.add("hidden"));
  elements.settingsModal.addEventListener("click", (event) => {
    if (event.target === elements.settingsModal) elements.settingsModal.classList.add("hidden");
  });
  const exitService = async (button, reportError) => {
    button.disabled = true;
    try {
      const response = await fetch("/api/exit", { method: "POST" });
      if (!response.ok) throw new Error((await response.json()).detail || "Unable to stop service");
      state.exiting = true;
      state.ws?.close();
      window.close();
      document.body.replaceChildren();
      const message = document.createElement("p");
      message.className = "empty-state";
      message.textContent = "Nocturnomath is stopping. You can close this tab.";
      document.body.appendChild(message);
    } catch (error) {
      reportError(error.message);
      button.disabled = false;
    }
  };
  elements.btnExit.addEventListener("click", () => exitService(elements.btnExit, appendErrorMessage));
  elements.btnExitLanding.addEventListener("click", () =>
    exitService(elements.btnExitLanding, (message) => {
      document.getElementById("recent-workspace-error").textContent = message;
    }),
  );
  document.querySelectorAll("[data-theme-toggle]").forEach((button) => {
    button.addEventListener("click", () => {
      applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
    });
  });
  applyTheme(document.documentElement.dataset.theme);

  // Chat form submit
  elements.chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = elements.promptInput.value.trim();
    if (!text) return;
    const selectedModel = elements.modelSelect.value;
    if (!selectedModel) return;
    const selectedProvider = state.modelProviders[selectedModel] || null;

    const sent = sendWs("query", {
      agent_id: state.activeAgentId,
      text,
      model: selectedModel,
      provider: selectedProvider,
      raw_model: rawModelId(selectedModel, selectedProvider),
      effort: elements.effortSelect.value || null,
    });
    if (!sent) {
      appendErrorMessage("WebSocket is reconnecting; your message was kept in the input box.");
      return;
    }
    elements.promptInput.value = "";
    elements.promptInput.style.height = "44px";
    const agent = state.agents.get(state.activeAgentId);
    if (agent) {
      agent.draftText = "";
      agent.draftModel = null;
      agent.draftProvider = null;
      agent.draftRawModel = null;
      agent.draftEffort = null;
    }
  });

  // Prompt Enter / Shift+Enter handling
  elements.promptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      elements.chatForm.dispatchEvent(new Event("submit"));
    }
  });

  // Auto-expand textarea
  elements.promptInput.addEventListener("input", () => {
    elements.promptInput.style.height = "44px";
    elements.promptInput.style.height = Math.min(elements.promptInput.scrollHeight, 160) + "px";
    const agent = state.agents.get(state.activeAgentId);
    if (agent) agent.draftText = elements.promptInput.value;
  });
  const recordModelDraft = () => {
    const agent = state.agents.get(state.activeAgentId);
    if (!agent) return;
    const selected = elements.modelSelect.value;
    const provider = state.modelProviders[selected] || null;
    agent.draftModel = selected;
    agent.draftProvider = provider;
    agent.draftRawModel = rawModelId(selected, provider);
    agent.draftEffort = elements.effortSelect.value || null;
  };
  elements.modelSelect.addEventListener("change", () => {
    setEffortForModel(elements.effortSelect, elements.modelSelect.value);
    recordModelDraft();
  });
  elements.effortSelect.addEventListener("change", recordModelDraft);
  elements.defaultModelSelect.addEventListener("change", () => {
    setEffortForModel(elements.defaultEffortSelect, elements.defaultModelSelect.value);
  });
  elements.sessionDefaultsForm.addEventListener("submit", saveSessionDefaults);
  elements.explorationPromptForm.addEventListener("submit", (event) => {
    event.preventDefault();
    saveExplorationPrompt(elements.explorationPromptInput.value);
  });
  elements.explorationPromptReset.addEventListener("click", () => saveExplorationPrompt(null));
  if (elements.promptModeSelect) {
    elements.promptModeSelect.addEventListener("change", () => savePromptMode(elements.promptModeSelect.value));
  }

  // Figures session filter
  elements.plotsFilterBtn.addEventListener("click", () => togglePlotsFilterMenu());
  elements.plotsFilterMenu.addEventListener("change", (e) => {
    const input = e.target;
    if (input.name === "plots-filter-mode") {
      setPlotsFilter(input.value);
    } else if (input.type === "checkbox") {
      const chosen = [...elements.plotsFilterSessions.querySelectorAll("input:checked")].map((box) => box.value);
      setPlotsFilter("selected", chosen);
    }
  });
  document.addEventListener("click", (e) => {
    if (!elements.plotsFilter.contains(e.target)) togglePlotsFilterMenu(false);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") togglePlotsFilterMenu(false);
  });

  // Control Buttons
  elements.btnNewExplorer.addEventListener("click", createExplorer);
  elements.btnHistory.addEventListener("click", () => {
    elements.historyModal.classList.remove("hidden");
    loadSessions();
  });
  elements.btnDownloadNotebook.addEventListener("click", async () => {
    elements.btnDownloadNotebook.disabled = true;
    try {
      const query = state.activeAgentId ? `?agent_id=${encodeURIComponent(state.activeAgentId)}` : "";
      const response = await fetch(`/api/session/notebook${query}`);
      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `Download failed (${response.status})`);
      }
      const blob = await response.blob();
      const disposition = response.headers.get("Content-Disposition") || "";
      const filename = disposition.match(/filename="([^"]+)"/)?.[1] || "nocturnomath-session.ipynb";
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) {
      appendErrorMessage(error.message);
    } finally {
      elements.btnDownloadNotebook.disabled = false;
    }
  });
  elements.historyModalClose.addEventListener("click", () => {
    if (!state.historyLoading) elements.historyModal.classList.add("hidden");
  });
  elements.historyModalCancel.addEventListener("click", () => {
    if (!state.historyLoading) elements.historyModal.classList.add("hidden");
  });
  elements.historyModal.addEventListener("click", (e) => {
    if (e.target === elements.historyModal && !state.historyLoading) elements.historyModal.classList.add("hidden");
  });
  elements.btnInterrupt.addEventListener("click", () => sendWs("interrupt", { agent_id: state.activeAgentId }));

  // Workspace Switcher Modal
  elements.btnOpenWorkspace.addEventListener("click", () => openFolderPicker("open"));
  const setAboutOpen = (open) => {
    elements.aboutPanel.hidden = !open;
    elements.landingScreen.classList.toggle("about-open", open);
    elements.btnAboutToggle.setAttribute("aria-expanded", String(open));
  };
  elements.btnAboutToggle.addEventListener("click", () => setAboutOpen(elements.aboutPanel.hidden));
  elements.btnAboutClose.addEventListener("click", () => setAboutOpen(false));
  elements.workspacePill.addEventListener("click", () => openFolderPicker("change"));
  elements.modalClose.addEventListener("click", closeFolderPicker);
  elements.modalCancel.addEventListener("click", closeFolderPicker);
  elements.folderModal.addEventListener("click", (e) => {
    if (e.target === elements.folderModal) closeFolderPicker();
  });
  elements.btnFolderGo.addEventListener("click", () => {
    browseFolders(elements.newFolderInput.value.trim());
  });
  elements.btnFolderUp.addEventListener("click", () => browseFolders(state.folderParent));
  elements.newFolderInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      browseFolders(elements.newFolderInput.value.trim());
    }
  });

  // Folder form submission
  elements.folderForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const newPath = elements.newFolderInput.value.trim();
    if (!newPath) return;

    elements.modalSubmit.disabled = true;
    elements.modalSubmit.textContent = "Preparing environment…";
    try {
      await openWorkspace(newPath);
      closeFolderPicker();
    } catch (err) {
      elements.folderPickerError.textContent = err.message;
      elements.folderPickerError.classList.remove("hidden");
    } finally {
      elements.modalSubmit.disabled = false;
      elements.modalSubmit.textContent = "Open folder";
    }
  });

  // Documents tab: list, reader, and editor
  elements.documentsBack.addEventListener("click", () => showDocumentsView("list"));
  elements.documentsNew.addEventListener("click", startCreate);
  elements.documentsNewFolder.addEventListener("click", startCreateFolder);
  elements.documentsFolderCancel.addEventListener("click", cancelCreateFolder);
  elements.documentsFolderForm.addEventListener("submit", (e) => {
    e.preventDefault();
    saveFolder();
  });
  elements.documentsModify.addEventListener("click", startModify);
  elements.documentsCancel.addEventListener("click", cancelEdit);
  elements.documentsEditor.addEventListener("submit", (e) => {
    e.preventDefault();
    saveEdit();
  });
  elements.documentsEditorText.addEventListener("keydown", (e) => {
    if (e.key === "Escape") cancelEdit();
  });
  elements.documentsDefaultToggle.addEventListener("change", (e) => {
    setDocumentsDefault(e.target.checked);
  });
  elements.chatMessages.addEventListener("click", async (event) => {
    const link = event.target.closest("[data-document-path]");
    if (!link) return;
    await selectViewerTab("documents");
    await openDocument(link.dataset.documentPath);
  });

  document.querySelectorAll("[data-tab]").forEach((button) => {
    button.addEventListener("click", () => selectViewerTab(button.dataset.tab));
  });

  // Lightbox
  elements.lightboxClose.addEventListener("click", closeLightbox);
  elements.lightboxModal.addEventListener("click", (e) => {
    if (e.target === elements.lightboxModal) closeLightbox();
  });
  document.addEventListener("keydown", (e) => {
    if (!isLightboxOpen()) return;
    if (e.key === "Escape") {
      e.preventDefault();
      closeLightbox();
    } else if (e.key === "ArrowLeft") {
      if (cycleLightbox(-1)) e.preventDefault();
    } else if (e.key === "ArrowRight") {
      if (cycleLightbox(1)) e.preventDefault();
    }
  });

  // Draggable Gutter / Pane Resizing
  setupGutterResize();
}

function applyAuthChange(data) {
  setAuthStatus(data.auth);
  setModelCatalogue(
    data.models || [],
    data.model_labels || {},
    data.model,
    data.model_efforts || {},
    data.effort,
    data.model_providers || {},
    data.agent_models || null,
    data.model_raw_ids || {},
  );
  if (data.session_defaults) setSessionDefaults(data.session_defaults);
  refreshSendButton();
}

async function saveSessionDefaults(event) {
  event.preventDefault();
  elements.sessionDefaultsSave.disabled = true;
  elements.sessionDefaultsStatus.textContent = "Saving…";
  try {
    const selectedModel = elements.defaultModelSelect.value;
    const selectedProvider = state.modelProviders[selectedModel] || null;
    const response = await fetch("/api/session/defaults", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: selectedModel,
        provider: selectedProvider,
        raw_model: rawModelId(selectedModel, selectedProvider),
        effort: elements.defaultEffortSelect.value || null,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Unable to save session defaults");
    setSessionDefaults(data.session_defaults);
    elements.sessionDefaultsStatus.textContent = "Saved for this workspace.";
  } catch (error) {
    elements.sessionDefaultsStatus.textContent = error.message;
  } finally {
    elements.sessionDefaultsSave.disabled = !state.models.length;
  }
}

async function saveExplorationPrompt(prompt) {
  elements.explorationPromptSave.disabled = true;
  elements.explorationPromptReset.disabled = true;
  elements.explorationPromptStatus.textContent = "Saving…";
  try {
    const response = await fetch("/api/settings/exploration-prompt", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Unable to save prompt");
    setExplorationPrompt(data.exploration_prompt, data.exploration_prompt_is_custom);
    elements.explorationPromptStatus.textContent = "Saved for this project.";
  } catch (error) {
    elements.explorationPromptStatus.textContent = error.message;
  } finally {
    elements.explorationPromptSave.disabled = false;
    elements.explorationPromptReset.disabled = !state.explorationPromptIsCustom;
  }
}

async function savePromptMode(mode) {
  const previous = state.promptMode;
  elements.promptModeSelect.disabled = true;
  elements.promptModeStatus.textContent = "Saving…";
  try {
    const response = await fetch("/api/settings/prompt-mode", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Unable to save prompt mode");
    setPromptMode(data.prompt_mode);
    elements.promptModeStatus.textContent = "Saved for this workspace.";
  } catch (error) {
    setPromptMode(previous);
    elements.promptModeStatus.textContent = error.message;
  } finally {
    elements.promptModeSelect.disabled = false;
  }
}

const AUTH_PROVIDER_METHODS = {
  anthropic: [
    {
      value: "cli",
      label: "Claude CLI (automatic)",
      help: "Use whatever Claude Code is already signed in with on this machine.",
    },
    {
      value: "subscription",
      label: "Claude subscription",
      credentialLabel: "Subscription token",
      placeholder: "Token from claude setup-token",
      help: "Run `claude setup-token` in a terminal, then paste the generated token here. It replaces any credential set in the environment.",
    },
    {
      value: "api_key",
      label: "Anthropic API key",
      credentialLabel: "API key",
      placeholder: "sk-ant-api…",
      help: "Create a key in the Anthropic Console. API usage is billed separately from a subscription.",
    },
  ],
  openai: [
    {
      value: "cli",
      label: "Codex CLI (automatic)",
      help: "Use whatever Codex CLI is already signed in with on this machine.",
    },
    {
      value: "api_key",
      label: "OpenAI API key",
      credentialLabel: "API key",
      placeholder: "sk-…",
      help: "Create an OpenAI API key. It overrides environment keys only for the Codex CLI launched by Nocturnomath.",
    },
  ],
  google: [
    {
      value: "cli",
      label: "Antigravity CLI (automatic)",
      help: "Use whatever Antigravity (`agy`) is already signed in with on this machine.",
    },
    {
      value: "api_key",
      label: "Google Gemini API key",
      credentialLabel: "Gemini API key",
      placeholder: "AIza…",
      help: "Create a Gemini API key in Google AI Studio. It runs in an isolated Antigravity state directory while keeping your local login untouched.",
    },
  ],
};

function populateAuthMethods(provider) {
  const methods = AUTH_PROVIDER_METHODS[provider] || AUTH_PROVIDER_METHODS.anthropic;
  elements.authMethod.replaceChildren(...methods.map((item) => new Option(item.label, item.value)));
  const currentMethod = state.auth?.providers?.[provider]?.method || "cli";
  elements.authMethod.value = methods.some((item) => item.value === currentMethod)
    ? currentMethod
    : methods[0].value;
}

function openAuthModal() {
  const provider = state.modelProviders[elements.modelSelect.value] || "anthropic";
  elements.authProvider.value = AUTH_PROVIDER_METHODS[provider] ? provider : "anthropic";
  populateAuthMethods(elements.authProvider.value);
  elements.authCredential.value = "";
  elements.authError.textContent = "";
  elements.authError.classList.add("hidden");
  updateAuthForm();
  elements.authModal.classList.remove("hidden");
}

function closeAuthModal() {
  if (elements.authSubmit.disabled) return;
  elements.authCredential.value = "";
  elements.authModal.classList.add("hidden");
}

function updateAuthForm() {
  const provider = elements.authProvider.value || "anthropic";
  const method = elements.authMethod.value || "cli";
  const methods = AUTH_PROVIDER_METHODS[provider] || AUTH_PROVIDER_METHODS.anthropic;
  const spec = methods.find((item) => item.value === method) || methods[0];
  const needsSecret = method !== "cli";
  elements.authCredentialGroup.classList.toggle("hidden", !needsSecret);
  elements.authCredential.required = needsSecret;
  if (needsSecret) {
    elements.authCredentialLabel.textContent = spec.credentialLabel || "Credential";
    elements.authCredential.placeholder = spec.placeholder || "";
  }
  elements.authHelp.textContent = spec.help || "";
  const providerInfo = state.auth?.providers?.[provider];
  elements.authCurrent.textContent = providerInfo
    ? `Current (${providerInfo.title}): ${providerInfo.label}`
    : "";
  const noteText = providerInfo?.note || "";
  elements.authNote.textContent = noteText;
  elements.authNote.classList.toggle("hidden", !noteText);
}

async function submitAuth(event) {
  event.preventDefault();
  elements.authSubmit.disabled = true;
  elements.authSubmit.textContent = "Applying…";
  elements.authError.classList.add("hidden");
  const provider = elements.authProvider.value || "anthropic";
  const method = elements.authMethod.value || "cli";
  const body = { provider, method };
  if (method !== "cli") body.credential = elements.authCredential.value;
  try {
    const response = await fetch("/api/auth", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Unable to update authentication");
    applyAuthChange(data);
    elements.authCredential.value = "";
    elements.authModal.classList.add("hidden");
  } catch (error) {
    elements.authCredential.value = "";
    elements.authError.textContent = error.message;
    elements.authError.classList.remove("hidden");
  } finally {
    elements.authSubmit.disabled = false;
    elements.authSubmit.textContent = "Use selected method";
  }
}

function setupGutterResize() {
  let isDragging = false;

  elements.gutter.addEventListener("mousedown", (e) => {
    isDragging = true;
    elements.gutter.classList.add("dragging");
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  });

  window.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    const containerWidth = elements.paneChat.parentElement.clientWidth;
    const newLeftWidth = (e.clientX / containerWidth) * 100;

    // Limit resize between 25% and 75%
    if (newLeftWidth >= 25 && newLeftWidth <= 75) {
      elements.paneChat.style.flex = `0 0 ${newLeftWidth}%`;
      elements.paneViewer.style.flex = `0 0 ${100 - newLeftWidth}%`;
    }
  });

  window.addEventListener("mouseup", () => {
    if (isDragging) {
      isDragging = false;
      elements.gutter.classList.remove("dragging");
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    }
  });
}

// Initial Boot
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  setInterval(() => {
    if (!state.exiting && !document.hidden && window.getSelection()?.isCollapsed !== false) refreshDocuments();
  }, 3000);
  initWebSocket(
    handleServerEvent,
    () => {
      if (state.workspace?.is_open) updateKernelStatus(true, false);
    },
    () => {
      updateKernelStatus(false, false);
      document.getElementById("app-loading").textContent = "Connecting…";
    }
  );
});

async function changeEnvironment(python) {
  const form = document.getElementById("environment-form");
  const message = document.getElementById("environment-error");
  form.querySelectorAll("button").forEach(button => button.disabled = true);
  message.textContent = "Preparing environment…";
  try {
    await openWorkspace(state.workspace.path, python);
    message.textContent = "Environment selected; a fresh session is ready.";
  } catch (error) {
    message.textContent = error.message;
  } finally {
    form.querySelectorAll("button").forEach(button => button.disabled = false);
  }
}

document.getElementById("environment-form").addEventListener("submit", event => {
  event.preventDefault();
  const python = document.getElementById("environment-python").value.trim();
  if (python) changeEnvironment(python);
});
document.getElementById("environment-managed").addEventListener("click", () => changeEnvironment("managed"));
