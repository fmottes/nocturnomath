import { elements, state } from "./state.js?v=20260913-1";
import { initWebSocket, sendWs } from "./transport.js?v=20260913-1";
import { openWorkspace, applyWorkspace, browseFolders, closeFolderPicker, loadDocument, loadPlots, openFolderPicker, setActiveModelAndEffort, setAuthStatus, setCarryContext, setContextUsage, setEffortForModel, setModelCatalogue, setSessionDefaults, selectViewerTab, refreshDocuments, setPlotsFilter, togglePlotsFilterMenu, setSessionId } from "./workspace.js?v=20260923-2";
import { appendAssistantChunk, appendAssistantDelta, appendErrorMessage, appendNoteNotification, appendProbeFinish, appendProbeStart, appendProbeVerdict, appendSystemMessage, appendUserMessage, finalizeAssistantTurn } from "./chat.js?v=20260922-2";
import { clearChat, loadSessions, replaySession, restoreChat } from "./history.js?v=20260922-2";
import { refreshSendButton, updateAgentStatus, updateKernelStatus } from "./status.js?v=20260913-1";
import { closeLightbox, cycleLightbox, isLightboxOpen } from "./ui.js?v=20260923-1";
import { cancelEdit, saveEdit, setDocumentsDefault, showDocumentsView, startCreate, startModify } from "./documents.js?v=20260923-1";

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

function applyExplorer(agent, repaint = false) {
  if (!agent?.agent_id) return;
  const prior = state.agents.get(agent.agent_id) || {};
  state.agents.set(agent.agent_id, { ...prior, ...agent });
  if (agent.agent_id === state.activeAgentId) {
    setSessionId(agent.session_id);
    setActiveModelAndEffort(agent.draftModel || agent.model, agent.draftEffort || agent.effort);
    setCarryContext(agent.carry_chat_context);
    setContextUsage(agent.context_usage);
    updateKernelStatus(agent.kernel_alive, agent.kernel_busy);
    updateAgentStatus(agent.is_busy ? "thinking" : "idle");
    if (agent.documents) refreshDocuments();
    if (repaint) {
      restoreChat(agent.records || [], agent.kernel_busy);
      restoreExplorerView(state.agents.get(agent.agent_id));
    }
  }
  renderExplorerTabs();
}

async function selectExplorer(agentId) {
  if (!state.agents.has(agentId) || agentId === state.activeAgentId) return;
  saveExplorerView();
  state.activeAgentId = agentId;
  const cached = state.agents.get(agentId);
  applyExplorer(cached, true);
  try {
    const response = await fetch(`/api/agents/${encodeURIComponent(agentId)}`);
    if (!response.ok) throw new Error("Explorer no longer exists");
    if (state.activeAgentId === agentId) {
      saveExplorerView();
      applyExplorer(await response.json(), true);
    }
    loadPlots();
  } catch (error) {
    appendErrorMessage(error.message);
  }
}

function installExplorers(agents = [], preserveView = true) {
  if (preserveView) saveExplorerView();
  const previous = preserveView ? state.agents : new Map();
  state.agents = new Map(agents.map((agent) => [
    agent.agent_id,
    { ...(previous.get(agent.agent_id) || {}), ...agent },
  ]));
  if (!state.agents.has(state.activeAgentId)) state.activeAgentId = agents[0]?.agent_id || null;
  if (state.activeAgentId) applyExplorer(state.agents.get(state.activeAgentId), true);
  renderExplorerTabs();
}

async function refreshContextUsage(agentId) {
  try {
    const response = await fetch(`/api/agents/${encodeURIComponent(agentId)}`);
    if (!response.ok) return;
    const agent = await response.json();
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
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Unable to close explorer");
  } catch (error) {
    appendErrorMessage(error.message);
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
    const current = state.agents.get(event.agent_id) || { agent_id: event.agent_id };
    if (event.type === "status_change") current.is_busy = !["idle", "cancelled"].includes(event.status);
    if (event.type === "kernel_restarted" || event.type === "kernel_interrupted") {
      current.kernel_alive = true;
      current.kernel_busy = false;
    }
    if (event.type === "carry_context_changed") current.carry_chat_context = event.carry_chat_context;
    if (event.type === "context_usage_changed") current.context_usage = event.context_usage;
    state.agents.set(event.agent_id, current);
    renderExplorerTabs();
    if (event.agent_id !== state.activeAgentId && event.type !== "agent_focused") {
      if (event.type === "record_changed" && state.activeTab !== "plots") {
        loadDocument(state.activeDoc, "", true);
      }
      return;
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
      if (removedActive) state.activeAgentId = state.agents.keys().next().value || null;
      if (state.activeAgentId) applyExplorer(state.agents.get(state.activeAgentId), true);
      renderExplorerTabs();
      break;
    }

    case "auth_changed":
      applyAuthChange(event);
      appendSystemMessage(`Claude authentication changed to ${event.auth?.label || "the selected method"}.`);
      break;

    case "user_message":
      if (event.session_id && state.agents.has(event.agent_id)) {
        state.agents.get(event.agent_id).session_id = event.session_id;
        setSessionId(event.session_id);
        renderExplorerTabs();
      }
      appendUserMessage(event.text);
      break;

    case "probe_start":
      appendProbeStart(event.expected, event.code, event.probe_id);
      break;

    case "probe_finish":
      // Preserve the web app's existing probe card contract: `output` includes
      // source paths and any guidance emitted alongside the raw kernel text.
      appendProbeFinish(event.expected, event.code, event.output, event.plot_urls, event.plot_images, event.probe_id);
      loadPlots();
      break;

    case "record_changed":
      appendNoteNotification(event.kind, `${event.entry_id}: ${event.text}`);
      if (state.activeTab !== "plots" && [state.workspace?.evidence_path, state.workspace?.thoughts_path].includes(state.activeDoc)) {
        loadDocument(state.activeDoc, "", true);
      }
      break;

    case "probe_verdict":
      appendProbeVerdict(event.text);
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
      refreshContextUsage(event.agent_id);
      break;

    case "status_change":
      state.agents.get(event.agent_id).is_busy = !["idle", "cancelled"].includes(event.status);
      updateAgentStatus(event.status);
      renderExplorerTabs();
      break;

    case "kernel_restarted":
      appendSystemMessage("Kernel restarted. In-memory variables and state cleared.");
      updateKernelStatus(true, false);
      break;

    case "kernel_interrupted":
      appendSystemMessage("Kernel interrupted by user.");
      updateKernelStatus(true, false);
      break;

    case "carry_context_changed":
      setCarryContext(event.carry_chat_context);
      appendSystemMessage(
        state.carryChatContext
          ? "Keeping context from here on. The agent sees this conversation as it grows; earlier messages are not recovered."
          : "Discarding context. Every message starts fresh from the system prompt, `evidence.md`, `thoughts.md`, and the checked documents."
      );
      break;

    case "context_usage_changed":
      setContextUsage(event.context_usage);
      break;

    case "session_reset":
      if (state.agents.has(event.agent_id)) state.agents.get(event.agent_id).context_usage = null;
      setContextUsage(null);
      setSessionId(event.session_id);
      setActiveModelAndEffort(event.model, event.effort);
      clearChat();
      updateKernelStatus(true, false);
      break;

    case "session_defaults_changed":
      setSessionDefaults(event.session_defaults);
      break;

    case "session_resumed":
      if (state.agents.has(event.agent_id)) {
        Object.assign(state.agents.get(event.agent_id), {
          draftText: "", scrollTop: 0, followChat: true,
        });
      }
      applyExplorer(event);
      restoreExplorerView(state.agents.get(event.agent_id));
      setSessionId(event.id);
      state.carryChatContext = Boolean(event.carry_chat_context);
      replaySession(
        event.records || [],
        event.context_restored,
        event.carry_chat_context,
        event.kernel_reset,
        event.kernel_restored,
        event.probes_replayed
      );
      break;

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
  [elements.btnAuth, elements.btnAuthLanding, elements.btnAuthSettings].forEach((button) => {
    button.addEventListener("click", openAuthModal);
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
    if (!elements.modelSelect.value) return;

    sendWs("query", {
      agent_id: state.activeAgentId,
      text,
      model: elements.modelSelect.value,
      effort: elements.effortSelect.value || null,
    });
    elements.promptInput.value = "";
    elements.promptInput.style.height = "44px";
    const agent = state.agents.get(state.activeAgentId);
    if (agent) agent.draftText = "";
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
  elements.modelSelect.addEventListener("change", () => {
    setEffortForModel(elements.effortSelect, elements.modelSelect.value);
    const agent = state.agents.get(state.activeAgentId);
    if (agent) {
      agent.draftModel = elements.modelSelect.value;
      agent.draftEffort = elements.effortSelect.value || null;
    }
  });
  elements.effortSelect.addEventListener("change", () => {
    const agent = state.agents.get(state.activeAgentId);
    if (agent) agent.draftEffort = elements.effortSelect.value || null;
  });
  elements.defaultModelSelect.addEventListener("change", () => {
    setEffortForModel(elements.defaultEffortSelect, elements.defaultModelSelect.value);
  });
  elements.sessionDefaultsForm.addEventListener("submit", saveSessionDefaults);

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
  elements.contextToggle.addEventListener("change", (e) => {
    sendWs("set_carry_context", { agent_id: state.activeAgentId, enabled: e.target.checked });
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
  );
  if (data.session_defaults) setSessionDefaults(data.session_defaults);
  refreshSendButton();
}

async function saveSessionDefaults(event) {
  event.preventDefault();
  elements.sessionDefaultsSave.disabled = true;
  elements.sessionDefaultsStatus.textContent = "Saving…";
  try {
    const response = await fetch("/api/session/defaults", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: elements.defaultModelSelect.value,
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

function openAuthModal() {
  elements.authMethod.value = state.auth?.method || "claude_code";
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
  const method = elements.authMethod.value;
  elements.authCredentialGroup.classList.toggle("hidden", method === "claude_code");
  elements.authCredential.required = method !== "claude_code";
  if (method === "subscription") {
    elements.authCredentialLabel.textContent = "Subscription token";
    elements.authCredential.placeholder = "Token from claude setup-token";
    elements.authHelp.textContent = "Run `claude setup-token` in a terminal, then paste the generated token here. It replaces any credential set in the environment.";
  } else if (method === "api_key") {
    elements.authCredentialLabel.textContent = "API key";
    elements.authCredential.placeholder = "sk-ant-api…";
    elements.authHelp.textContent = "Create a key in the Claude Console. API usage is billed separately from a subscription.";
  } else {
    elements.authHelp.textContent = "Use whatever Claude Code is already signed in with, exactly as the Agent SDK would on its own.";
  }
}

async function submitAuth(event) {
  event.preventDefault();
  elements.authSubmit.disabled = true;
  elements.authSubmit.textContent = "Applying…";
  elements.authError.classList.add("hidden");
  const method = elements.authMethod.value;
  const body = { method };
  if (method !== "claude_code") body.credential = elements.authCredential.value;
  try {
    const response = await fetch("/api/auth", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Unable to authenticate with Claude");
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
