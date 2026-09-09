import { elements, state } from "./state.js?v=20260909-2";
import { initWebSocket, sendWs } from "./transport.js?v=20260909-2";
import { openWorkspace, applyWorkspace, browseFolders, closeFolderPicker, loadDocument, loadPlots, openFolderPicker, setCarryContext, selectViewerTab, refreshDocuments } from "./workspace.js?v=20260909-2";
import { appendAssistantChunk, appendErrorMessage, appendNoteNotification, appendProbeFinish, appendProbeStart, appendProbeVerdict, appendSystemMessage, appendUserMessage, finalizeAssistantTurn } from "./chat.js?v=20260909-2";
import { clearChat, loadSessions, replaySession } from "./history.js?v=20260909-2";
import { updateAgentStatus, updateKernelStatus } from "./status.js?v=20260909-2";
import { closeLightbox, cycleLightbox, isLightboxOpen } from "./ui.js?v=20260909-2";

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
      break;
    case "workspace_updated":
      clearChat();
      applyWorkspace(event.workspace);
      break;

    case "user_message":
      appendUserMessage(event.text);
      break;

    case "probe_start":
      appendProbeStart(event.expected, event.code);
      break;

    case "probe_finish":
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
          : "Discarding context. Every message starts fresh from the system prompt plus `evidence.md` and `thoughts.md`."
      );
      break;

    case "session_reset":
      clearChat();
      updateKernelStatus(true, false);
      break;

    case "session_resumed":
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

  // Control Buttons
  elements.btnNewSession.addEventListener("click", () => sendWs("new_session"));
  elements.btnHistory.addEventListener("click", () => {
    elements.historyModal.classList.remove("hidden");
    loadSessions();
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

  // Document controls
  elements.fileSelect.addEventListener("change", (e) => {
    loadDocument(e.target.value, "", false, "doc");
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
