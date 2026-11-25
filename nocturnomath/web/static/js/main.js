import { elements, state } from "./state.js";
import { agentQuery, apiJson, initWebSocket, sendWs } from "./transport.js";
import { openWorkspace, applyWorkspace, browseFolders, clearDraftSelection, closeFolderPicker, loadDocument, loadPlots, openFolderPicker, rawModelId, setActiveModelAndEffort, setAuthStatus, setContextUsage, setEffortForModel, setExplorationPrompt, setModelCatalogue, setPromptMode, setSessionDefaults, selectViewerTab, refreshDocuments, setPlotsFilter, togglePlotsFilterMenu, setSessionId, syncDraftSelection } from "./workspace.js";
import { appendAssistantChunk, appendAssistantDelta, appendDocumentNotification, appendErrorMessage, appendNoteNotification, appendProbeFinish, appendProbeStart, appendProbeVerdict, appendSystemMessage, appendUserMessage, finalizeAssistantTurn, syncJumpButton } from "./chat.js";
import { clearChat, loadSessions, replaySession, restoreChat } from "./history.js";
import { refreshSendButton, setConnectionStatus, updateAgentStatus, updateKernelStatus } from "./status.js";
import { closeLightbox, cycleLightbox, isLightboxOpen } from "./ui.js";
import { cancelCreateFolder, cancelEdit, fetchDocuments, finishEdit, flushAutosave, openDocument, saveEdit, saveFolder, scheduleAutosave, setDocuments, setDocumentsDefault, showDocumentsView, startCreate, startCreateFolder, startModify } from "./documents.js";

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

let updateDismissed = false;
function setUpdateNotice(info) {
  const show = Boolean(info?.latest) && !updateDismissed;
  document.querySelectorAll("[data-update-notice]").forEach((el) => {
    el.classList.toggle("hidden", !show);
    if (!show) return;
    const link = el.querySelector("a");
    link.textContent = `v${info.latest} available`;
    link.href = info.url || "https://github.com/fmottes/nocturnomath";
    link.title = `Update available (${info.current} → ${info.latest}). Run: ${info.command}`;
  });
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
    const labelText = explorerLabel(agent);
    tab.title = `Explorer ${labelText}`;
    tab.setAttribute("aria-label", `Explorer ${labelText}`);
    const title = document.createElement("span");
    title.className = "pane-title";
    title.textContent = labelText;
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
  if (!agent) return;
  elements.promptInput.value = agent.draftText || "";
  elements.promptInput.style.height = "44px";
  elements.promptInput.style.height = Math.min(elements.promptInput.scrollHeight, 160) + "px";
  state.followChat = agent.followChat ?? true;
  elements.chatMessages.scrollTop = state.followChat
    ? elements.chatMessages.scrollHeight
    : (agent.scrollTop || 0);
  syncJumpButton(false);
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
  const resolvedDraft = syncDraftSelection(merged);
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
    if (agent.documents) setDocuments(agent.documents, agent.documents_default, agent.folders);
    if (repaint) {
      restoreChat(merged.records || [], Boolean(merged.is_busy || merged.kernel_busy));
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
  const cachedRecordsKey = JSON.stringify(cached?.records || []);
  const cachedBusy = Boolean(cached?.is_busy || cached?.kernel_busy);
  applyExplorer(cached, true);
  try {
    const snapshot = await apiJson(`/api/agents/${encodeURIComponent(agentId)}`, {
      fallbackError: "Explorer no longer exists",
    });
    if (state.activeAgentId === agentId && seq === explorerSelectSeq) {
      saveExplorerView();
      const snapshotBusy = Boolean(snapshot.is_busy || snapshot.kernel_busy);
      const recordsChanged = JSON.stringify(snapshot.records || []) !== cachedRecordsKey
        || snapshotBusy !== cachedBusy;
      applyExplorer(snapshot, recordsChanged);
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
    const agent = await apiJson("/api/agents", {
      method: "POST",
      fallbackError: "Unable to start explorer",
    });
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
    await apiJson(`/api/agents/${encodeURIComponent(agentId)}`, {
      method: "DELETE",
      fallbackError: "Unable to close explorer",
    });
  } catch (error) {
    if (error.status === 404 && state.agents.size > 1) {
      handleServerEvent({ type: "agent_removed", agent_id: agentId });
      return;
    }
    appendErrorMessage(error.message);
  }
}

async function compactExplorer() {
  const agentId = state.activeAgentId;
  if (!agentId || elements.btnCompact.disabled) return;
  elements.btnCompact.disabled = true;
  try {
    await apiJson(`/api/agents/${encodeURIComponent(agentId)}/compact`, {
      method: "POST",
      fallbackError: "Unable to compact explorer",
    });
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
      let tabsChanged = false;
      if (event.type === "status_change") {
        const nextBusy = !["idle", "cancelled"].includes(event.status);
        if (current.is_busy !== nextBusy) tabsChanged = true;
        current.is_busy = nextBusy;
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
        if (event.session_id && current.session_id !== event.session_id) {
          current.session_id = event.session_id;
          tabsChanged = true;
        }
        if (event.model) {
          current.model = event.model;
          current.modelProvider = state.modelProviders[event.model] || current.modelProvider || null;
        }
        if (event.effort !== undefined) current.effort = event.effort;
      }
      state.agents.set(event.agent_id, current);
      if (tabsChanged) renderExplorerTabs();
      if (
        event.agent_id !== state.activeAgentId
        && !["agent_focused", "session_reset", "session_resumed"].includes(event.type)
      ) {
        if (
          event.type === "record_changed"
          && !["plots", "documents"].includes(state.activeTab)
          && [state.workspace?.evidence_path, state.workspace?.thoughts_path].includes(state.activeDoc)
        ) {
          loadDocument(state.activeDoc, "", true);
        }
        if (event.type === "probe_finish") loadPlots();
        if (event.type === "document_saved") {
          if (state.activeTab === "documents") refreshDocuments();
          else fetchDocuments();
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
      setUpdateNotice(event.workspace?.update_available);
      if (event.workspace?.is_open) installExplorers(event.workspace.agents || []);
      break;
    case "workspace_updated":
      clearChat();
      state.activeAgentId = null;
      applyWorkspace(event.workspace);
      setUpdateNotice(event.workspace?.update_available);
      installExplorers(event.workspace.agents || [], false);
      break;

    case "update_available":
      setUpdateNotice(event);
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
      if (changed?.error) {
        appendErrorMessage(`${changed.title}: ${changed.error}`);
      } else {
        const noteSuffix = changed?.note ? ` (${changed.note})` : "";
        const summary = changed
          ? `${changed.title} authentication changed to ${changed.label}.${noteSuffix}`
          : `Authentication changed to ${event.auth?.label || "the selected method"}.`;
        appendSystemMessage(summary);
      }
      break;
    }

    case "user_message": {
      const agent = state.agents.get(event.agent_id);
      if (agent && (event.model !== undefined || event.effort !== undefined) && !agent.draftModel && !agent.draftRawModel) {
        setActiveModelAndEffort(agent.model, agent.effort);
      }
      if (event.session_id && agent) {
        setSessionId(event.session_id);
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
      if (
        !["plots", "documents"].includes(state.activeTab)
        && [state.workspace?.evidence_path, state.workspace?.thoughts_path].includes(state.activeDoc)
      ) {
        loadDocument(state.activeDoc, "", true);
      }
      break;

    case "document_saved":
      appendDocumentNotification(event.name, event.action);
      if (state.activeTab === "documents") refreshDocuments();
      else fetchDocuments();
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
      if (
        agent &&
        ["cancelled", "idle"].includes(event.status) &&
        !agent.draftModel &&
        !agent.draftRawModel
      ) {
        setActiveModelAndEffort(agent.model, agent.effort);
      }
      updateAgentStatus(event.status);
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
      const target = state.agents.get(targetId);
      if (target) {
        Object.assign(target, {
          context_usage: null,
          session_id: event.session_id ?? null,
          records: [],
          can_compact: false,
          pending_prompts: [],
          model: event.model,
          modelProvider: state.modelProviders[event.model] || null,
          effort: event.effort,
        });
        clearDraftSelection(target);
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
      const targetAgentId = event.agent_id || state.activeAgentId;
      const target = targetAgentId ? state.agents.get(targetAgentId) : null;
      if (target) {
        Object.assign(target, {
          session_id: resumedId,
          can_compact: event.can_compact ?? Boolean(event.context_restored),
          draftText: "",
          scrollTop: 0,
          followChat: true,
        });
        clearDraftSelection(target);
      }
      applyExplorer({
        is_busy: false,
        status: "idle",
        kernel_busy: false,
        ...event,
        agent_id: targetAgentId,
        session_id: resumedId,
        can_compact: event.can_compact ?? Boolean(event.context_restored),
      });
      if (!event.agent_id || event.agent_id === state.activeAgentId) {
        restoreExplorerView(targetAgentId ? state.agents.get(targetAgentId) : null);
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

function bindModalClose(modal, onClose, ...buttons) {
  buttons.forEach((button) => button?.addEventListener("click", onClose));
  modal.addEventListener("click", (event) => {
    if (event.target === modal) onClose();
  });
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
    syncJumpButton(false);
  }, { passive: true });
  elements.btnJumpBottom?.addEventListener("click", () => {
    state.followChat = true;
    const agent = state.agents.get(state.activeAgentId);
    if (agent) agent.followChat = true;
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    syncJumpButton(false);
  });
  elements.btnSettings.addEventListener("click", () => {
    elements.settingsModal.classList.remove("hidden");
    elements.settingsClose.focus();
  });
  elements.btnCompact.addEventListener("click", compactExplorer);
  [elements.btnAuth, elements.btnAuthLanding, elements.btnAuthSettings].forEach((button) => {
    button.addEventListener("click", openAuthModal);
  });
  elements.authProvider.addEventListener("change", () => {
    elements.authCredential.value = "";
    elements.authError.textContent = "";
    elements.authError.classList.add("hidden");
    populateAuthMethods(elements.authProvider.value);
    updateAuthForm(true);
  });
  elements.authMethod.addEventListener("change", () => {
    elements.authCredential.value = "";
    elements.authError.textContent = "";
    elements.authError.classList.add("hidden");
    updateAuthForm(false);
  });
  bindModalClose(elements.authModal, closeAuthModal, elements.authModalClose, elements.authModalCancel);
  elements.authForm.addEventListener("submit", submitAuth);
  bindModalClose(elements.settingsModal, () => elements.settingsModal.classList.add("hidden"), elements.settingsClose);
  const exitService = async (button, reportError) => {
    button.disabled = true;
    try {
      await apiJson("/api/exit", { method: "POST", fallbackError: "Unable to stop service" });
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
  document.querySelectorAll("[data-dismiss-update]").forEach((button) => {
    button.addEventListener("click", () => {
      updateDismissed = true;
      setUpdateNotice(null);
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
      clearDraftSelection(agent);
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

  // Control Buttons
  elements.btnNewExplorer.addEventListener("click", createExplorer);
  elements.btnHistory.addEventListener("click", () => {
    elements.historyModal.classList.remove("hidden");
    elements.historyModalClose.focus();
    loadSessions();
  });
  elements.btnDownloadNotebook.addEventListener("click", async () => {
    elements.btnDownloadNotebook.disabled = true;
    try {
      const response = await fetch(`/api/session/notebook${agentQuery()}`);
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
  bindModalClose(
    elements.historyModal,
    () => { if (!state.historyLoading) elements.historyModal.classList.add("hidden"); },
    elements.historyModalClose,
    elements.historyModalCancel,
  );
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
  bindModalClose(elements.folderModal, closeFolderPicker, elements.modalClose, elements.modalCancel);
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
      elements.modalSubmit.textContent = state.folderPickerMode === "open" ? "Open folder" : "Switch workspace";
    }
  });

  // Research environment form
  document.getElementById("environment-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const python = document.getElementById("environment-python").value.trim();
    if (python) changeEnvironment(python);
  });
  document.getElementById("environment-managed").addEventListener("click", () => changeEnvironment("managed"));

  // Documents tab: list, reader, and editor
  elements.documentsBack.addEventListener("click", () => {
    if (state.documentsView === "edit") finishEdit("list");
    else showDocumentsView("list");
  });
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
  elements.documentsEditorText.addEventListener("input", scheduleAutosave);
  elements.documentsEditorText.addEventListener("blur", () => {
    if (state.documentsView === "edit") flushAutosave();
  });
  elements.documentsEditorText.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      e.stopPropagation();
      cancelEdit();
    }
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

  // Lightbox & modal Escape handling
  bindModalClose(elements.lightboxModal, closeLightbox, elements.lightboxClose);
  elements.lightboxPrev?.addEventListener("click", (e) => {
    e.stopPropagation();
    cycleLightbox(-1);
  });
  elements.lightboxNext?.addEventListener("click", (e) => {
    e.stopPropagation();
    cycleLightbox(1);
  });
  document.addEventListener("keydown", (e) => {
    if (isLightboxOpen()) {
      if (e.key === "Escape") {
        e.preventDefault();
        closeLightbox();
      } else if (e.key === "ArrowLeft") {
        if (cycleLightbox(-1)) e.preventDefault();
      } else if (e.key === "ArrowRight") {
        if (cycleLightbox(1)) e.preventDefault();
      }
      return;
    }
    if (e.key !== "Escape") return;
    if (!elements.folderModal.classList.contains("hidden")) {
      e.preventDefault();
      closeFolderPicker();
    } else if (!elements.authModal.classList.contains("hidden")) {
      e.preventDefault();
      closeAuthModal();
    } else if (!elements.historyModal.classList.contains("hidden")) {
      if (!state.historyLoading) {
        e.preventDefault();
        elements.historyModal.classList.add("hidden");
      }
    } else if (!elements.settingsModal.classList.contains("hidden")) {
      e.preventDefault();
      elements.settingsModal.classList.add("hidden");
    } else if (!elements.plotsFilterMenu.classList.contains("hidden")) {
      togglePlotsFilterMenu(false);
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
    const data = await apiJson("/api/session/defaults", {
      method: "POST",
      body: {
        model: selectedModel,
        provider: selectedProvider,
        raw_model: rawModelId(selectedModel, selectedProvider),
        effort: elements.defaultEffortSelect.value || null,
      },
      fallbackError: "Unable to save session defaults",
    });
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
    const data = await apiJson("/api/settings/exploration-prompt", {
      method: "PUT",
      body: { prompt },
      fallbackError: "Unable to save prompt",
    });
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
    const data = await apiJson("/api/settings/prompt-mode", {
      method: "PUT",
      body: { mode },
      fallbackError: "Unable to save prompt mode",
    });
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
      help: "Use whatever Claude Code is already signed in with on this machine, or ANTHROPIC_API_KEY from the environment over direct HTTP.",
    },
    {
      value: "subscription",
      label: "Claude subscription",
      credentialLabel: "Subscription token",
      placeholder: "Token from claude setup-token",
      help: "Run `claude setup-token` in a terminal, then paste the generated token here. Requires the `claude` CLI and replaces any credential in the environment.",
    },
    {
      value: "api_key",
      label: "Anthropic API key",
      credentialLabel: "API key",
      placeholder: "sk-ant-api…",
      help: "Create a key in the Anthropic Console. Connects directly to Anthropic's HTTP API with no CLI required, and stays in memory only.",
    },
  ],
  openai: [
    {
      value: "cli",
      label: "Codex CLI (automatic)",
      help: "Use whatever Codex CLI is already signed in with on this machine, or OPENAI_API_KEY / CODEX_API_KEY from the environment over direct HTTP.",
    },
    {
      value: "api_key",
      label: "OpenAI API key",
      credentialLabel: "API key",
      placeholder: "sk-…",
      help: "Create an OpenAI API key. Connects directly to OpenAI's HTTP API with no CLI required, and stays in memory only.",
    },
  ],
  google: [
    {
      value: "cli",
      label: "Antigravity CLI (automatic)",
      help: "Use whatever Antigravity (`agy`) is already signed in with on this machine, or GEMINI_API_KEY / GOOGLE_API_KEY from the environment over direct HTTP.",
    },
    {
      value: "api_key",
      label: "Google Gemini API key",
      credentialLabel: "Gemini API key",
      placeholder: "AIza…",
      help: "Create a Gemini API key in Google AI Studio. Connects directly to Gemini's HTTP API with no CLI required, and stays in memory only.",
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
  updateAuthForm(true);
  elements.authModal.classList.remove("hidden");
  elements.authProvider.focus();
}

function closeAuthModal() {
  if (elements.authSubmit.disabled) return;
  elements.authCredential.value = "";
  elements.authModal.classList.add("hidden");
}

function updateAuthForm(showProviderError = false) {
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
  if (showProviderError && providerInfo?.error) {
    elements.authError.textContent = providerInfo.error;
    elements.authError.classList.remove("hidden");
  }
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
    const data = await apiJson("/api/auth", {
      method: "POST",
      body,
      fallbackError: "Unable to update authentication",
    });
    applyAuthChange(data);
    elements.authCredential.value = "";
    const providerError = data.auth?.providers?.[provider]?.error;
    if (providerError) {
      elements.authError.textContent = providerError;
      elements.authError.classList.remove("hidden");
    } else {
      elements.authModal.classList.add("hidden");
    }
  } catch (error) {
    elements.authCredential.value = "";
    elements.authError.textContent = error.message;
    elements.authError.classList.remove("hidden");
  } finally {
    elements.authSubmit.disabled = false;
    elements.authSubmit.textContent = "Use selected method";
  }
}

const SPLIT_RATIO_KEY = "nocturnomath.split-ratio";

function applySplitRatio(percent) {
  elements.paneChat.style.flex = `0 0 ${percent}%`;
  elements.paneViewer.style.flex = `0 0 ${100 - percent}%`;
}

function setupGutterResize() {
  let isDragging = false;
  let currentRatio = null;

  try {
    const saved = Number.parseFloat(window.localStorage.getItem(SPLIT_RATIO_KEY) || "");
    if (Number.isFinite(saved) && saved >= 25 && saved <= 75) {
      currentRatio = saved;
      applySplitRatio(saved);
    }
  } catch {
    // Ignore storage errors and keep the default 50/50 split.
  }

  elements.gutter.addEventListener("mousedown", () => {
    isDragging = true;
    elements.gutter.classList.add("dragging");
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  });

  elements.gutter.addEventListener("dblclick", () => {
    currentRatio = null;
    elements.paneChat.style.flex = "";
    elements.paneViewer.style.flex = "";
    try {
      window.localStorage.removeItem(SPLIT_RATIO_KEY);
    } catch {
      // Ignore storage errors.
    }
  });

  window.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    const containerWidth = elements.paneChat.parentElement.clientWidth;
    if (!containerWidth) return;
    const newLeftWidth = (e.clientX / containerWidth) * 100;

    // Limit resize between 25% and 75%
    if (newLeftWidth >= 25 && newLeftWidth <= 75) {
      currentRatio = Math.round(newLeftWidth * 10) / 10;
      applySplitRatio(currentRatio);
    }
  });

  window.addEventListener("mouseup", () => {
    if (isDragging) {
      isDragging = false;
      elements.gutter.classList.remove("dragging");
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      if (currentRatio !== null) {
        try {
          window.localStorage.setItem(SPLIT_RATIO_KEY, String(currentRatio));
        } catch {
          // Ignore storage errors.
        }
      }
    }
  });
}

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

// Initial Boot
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  setInterval(() => {
    if (!state.exiting && !document.hidden && window.getSelection()?.isCollapsed !== false) refreshDocuments();
  }, 3000);
  initWebSocket(
    handleServerEvent,
    () => {
      setConnectionStatus(true);
    },
    () => {
      setConnectionStatus(false);
      document.getElementById("app-loading").textContent = "Connecting…";
    }
  );
});
