import { elements, state } from "./state.js?v=20260910-1";
import { initWebSocket, sendWs } from "./transport.js?v=20260910-1";
import { openWorkspace, applyWorkspace, browseFolders, closeFolderPicker, loadDocument, loadPlots, openFolderPicker, setAuthStatus, setCarryContext, setModelCatalogue, selectViewerTab, refreshDocuments, setPlotsFilter, togglePlotsFilterMenu, setSessionId } from "./workspace.js?v=20260910-1";
import { appendAssistantChunk, appendAssistantDelta, appendErrorMessage, appendNoteNotification, appendProbeFinish, appendProbeStart, appendProbeVerdict, appendSystemMessage, appendUserMessage, finalizeAssistantTurn } from "./chat.js?v=20260910-1";
import { clearChat, loadSessions, replaySession, restoreChat } from "./history.js?v=20260910-1";
import { refreshSendButton, updateAgentStatus, updateKernelStatus } from "./status.js?v=20260910-1";
import { closeLightbox, cycleLightbox, isLightboxOpen } from "./ui.js?v=20260910-1";
import { cancelEdit, saveEdit, setDocumentsDefault, showDocumentsView, startCreate, startModify } from "./documents.js?v=20260910-1";

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

// ============================================================================
// Server Event Dispatcher
// ============================================================================
function handleServerEvent(event) {
  switch (event.type) {
    case "service_stopping":
      state.exiting = true;
      elements.btnSend.disabled = true;
      appendSystemMessage("Nocturnomath is stopping. You can close this tab.");
      break;
    case "init":
      applyWorkspace(event.workspace);
      if (event.workspace?.is_open) restoreChat(event.workspace.records || []);
      break;
    case "workspace_updated":
      clearChat();
      applyWorkspace(event.workspace);
      break;

    case "auth_changed":
      applyAuthChange(event);
      appendSystemMessage(`Claude authentication changed to ${event.auth?.label || "the selected method"}.`);
      break;

    case "user_message":
      appendUserMessage(event.text);
      break;

    case "probe_start":
      appendProbeStart(event.expected, event.code);
      break;

    case "probe_finish":
      // Preserve the web app's existing probe card contract: `output` includes
      // source paths and any guidance emitted alongside the raw kernel text.
      appendProbeFinish(event.expected, event.code, event.output, event.plot_urls, event.plot_images);
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
      break;

    case "status_change":
      updateAgentStatus(event.status);
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

    case "session_reset":
      setSessionId(event.session_id);
      clearChat();
      updateKernelStatus(true, false);
      break;

    case "session_resumed":
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
  elements.chatMessages.addEventListener("scroll", () => {
    const log = elements.chatMessages;
    state.followChat = log.scrollHeight - log.clientHeight - log.scrollTop < 60;
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
  elements.btnExit.addEventListener("click", async () => {
    elements.btnExit.disabled = true;
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
      appendErrorMessage(error.message);
      elements.btnExit.disabled = false;
    }
  });
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

    sendWs("query", { text, model: elements.modelSelect.value });
    elements.promptInput.value = "";
    elements.promptInput.style.height = "44px";
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
  });

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
  elements.btnNewSession.addEventListener("click", () => sendWs("new_session"));
  elements.btnHistory.addEventListener("click", () => {
    elements.historyModal.classList.remove("hidden");
    loadSessions();
  });
  elements.btnDownloadNotebook.addEventListener("click", async () => {
    elements.btnDownloadNotebook.disabled = true;
    try {
      const response = await fetch("/api/session/notebook");
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
    sendWs("set_carry_context", { enabled: e.target.checked });
  });

  elements.btnInterrupt.addEventListener("click", () => sendWs("interrupt"));

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
  setModelCatalogue(data.models || [], data.model_labels || {}, data.model);
  refreshSendButton();
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
