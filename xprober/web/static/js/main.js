import { elements, state } from "./state.js";
import { initWebSocket, sendWs } from "./transport.js";
import { applyWorkspace, browseFolders, closeFolderPicker, loadDocument, loadPlots, openFolderPicker, setCarryContext } from "./workspace.js";
import { appendAssistantChunk, appendErrorMessage, appendNoteNotification, appendProbeFinish, appendProbeStart, appendProbeVerdict, appendSystemMessage, appendUserMessage, finalizeAssistantTurn } from "./chat.js";
import { loadSessions, replaySession } from "./history.js";
import { updateAgentStatus, updateKernelStatus } from "./status.js";
import { closeLightbox } from "./ui.js";

// ============================================================================
// Server Event Dispatcher
// ============================================================================
function handleServerEvent(event) {
  switch (event.type) {
    case "init":
    case "workspace_updated":
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

    case "note_added":
      appendNoteNotification(event.kind, event.text);
      if (state.autoSync && state.activeDoc === "notes.md") {
        loadDocument("notes.md");
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
          : "Discarding context. Every message starts fresh from the system prompt plus `notes.md`."
      );
      break;

    case "session_reset":
      appendSystemMessage("Started new exploration session with fresh context.");
      break;

    case "session_resumed":
      state.carryChatContext = Boolean(event.carry_chat_context);
      replaySession(
        event.records || [],
        event.context_restored,
        event.carry_chat_context
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
  // Chat form submit
  elements.chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = elements.promptInput.value.trim();
    if (!text) return;

    sendWs("query", { text });
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

  // Slash command chips
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const cmd = chip.dataset.cmd;
      sendWs("query", { text: cmd });
    });
  });

  // Control Buttons
  elements.btnNewSession.addEventListener("click", () => sendWs("new_session"));
  elements.btnHistory.addEventListener("click", () => {
    elements.historyModal.classList.remove("hidden");
    loadSessions();
  });
  elements.historyModalClose.addEventListener("click", () =>
    elements.historyModal.classList.add("hidden")
  );
  elements.historyModalCancel.addEventListener("click", () =>
    elements.historyModal.classList.add("hidden")
  );
  elements.btnRefreshSessions.addEventListener("click", () => loadSessions());
  elements.historyModal.addEventListener("click", (e) => {
    if (e.target === elements.historyModal) elements.historyModal.classList.add("hidden");
  });
  elements.contextToggle.addEventListener("change", (e) => {
    sendWs("set_carry_context", { enabled: e.target.checked });
  });

  elements.btnRestartKernel.addEventListener("click", () => sendWs("restart_kernel"));
  elements.btnInterrupt.addEventListener("click", () => sendWs("interrupt"));

  // Workspace Switcher Modal
  elements.btnOpenWorkspace.addEventListener("click", () => openFolderPicker("open"));
  elements.workspacePill.addEventListener("click", () => openFolderPicker("change"));
  elements.btnChangeFolder.addEventListener("click", (e) => {
    e.stopPropagation();
    openFolderPicker("change");
  });
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

    try {
      const res = await fetch("/api/workspace", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: newPath }),
      });
      if (res.ok) {
        const data = await res.json();
        applyWorkspace(data.workspace);
        closeFolderPicker();
        appendSystemMessage(`Opened workspace folder: <code>${data.workspace.path}</code>`);
      } else {
        const err = await res.json();
        alert(`Failed to switch folder: ${err.detail || "Unknown error"}`);
      }
    } catch (err) {
      alert(`Error switching folder: ${err.message}`);
    }
  });

  // Document controls
  elements.fileSelect.addEventListener("change", (e) => {
    loadDocument(e.target.value);
  });

  elements.btnRefreshDoc.addEventListener("click", () => {
    loadDocument(elements.fileSelect.value);
  });

  // No control for this in the UI at the moment; auto-sync is simply on. The
  // listener stays wired for whenever the toggle comes back.
  if (elements.autoRefreshToggle) {
    elements.autoRefreshToggle.addEventListener("change", (e) => {
      state.autoSync = e.target.checked;
    });
  }

  elements.btnToggleRaw.addEventListener("click", () => {
    state.rawView = !state.rawView;
    if (state.rawView) {
      elements.markdownContainer.classList.add("hidden");
      elements.rawMarkdownContainer.classList.remove("hidden");
      elements.btnToggleRaw.textContent = "Rendered";
    } else {
      elements.markdownContainer.classList.remove("hidden");
      elements.rawMarkdownContainer.classList.add("hidden");
      elements.btnToggleRaw.textContent = "Raw";
    }
  });

  elements.btnCopyDoc.addEventListener("click", async () => {
    const raw = elements.rawMarkdownContainer.querySelector("code").textContent;
    try {
      await navigator.clipboard.writeText(raw);
      const original = elements.btnCopyDoc.textContent;
      elements.btnCopyDoc.textContent = "Copied";
      setTimeout(() => (elements.btnCopyDoc.textContent = original), 1500);
    } catch (e) {
      console.error("Clipboard copy failed:", e);
    }
  });

  // Viewer Tabs
  elements.tabDoc.addEventListener("click", () => {
    elements.tabDoc.classList.add("active");
    elements.tabPlots.classList.remove("active");
    elements.viewDoc.classList.remove("hidden");
    elements.viewPlots.classList.add("hidden");
    elements.fileSelect.parentElement.classList.remove("hidden");
  });

  elements.tabPlots.addEventListener("click", () => {
    elements.tabPlots.classList.add("active");
    elements.tabDoc.classList.remove("active");
    elements.viewPlots.classList.remove("hidden");
    elements.viewDoc.classList.add("hidden");
    elements.fileSelect.parentElement.classList.add("hidden");
    loadPlots();
  });

  elements.btnRefreshPlots.addEventListener("click", () => loadPlots());

  // Lightbox
  elements.lightboxClose.addEventListener("click", closeLightbox);
  elements.lightboxModal.addEventListener("click", (e) => {
    if (e.target === elements.lightboxModal) closeLightbox();
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
  initWebSocket(
    handleServerEvent,
    () => {
      if (state.workspace?.is_open) updateKernelStatus(true, false);
    },
    () => updateKernelStatus(false, false)
  );
});
