import { elements, state } from "./state.js?v=20260913-1";
import { appendAssistantChunk, appendNoteNotification, appendProbeFinish, appendProbeStart, appendProbeVerdict, appendSystemMessage, appendUserMessage, finalizeAssistantTurn, scrollChatToBottom } from "./chat.js?v=20260922-2";

// ============================================================================
// Chat History (past sessions)
// ============================================================================
export function formatSessionDate(epochSeconds) {
  const d = new Date(epochSeconds * 1000);
  return d.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export async function loadSessions() {
  elements.sessionList.innerHTML = `<div class="empty-state">Loading sessions…</div>`;
  try {
    const res = await fetch("/api/sessions");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    state.carryChatContext = Boolean(data.carry_chat_context);
    renderSessions(data.sessions || []);
  } catch (e) {
    const message = document.createElement("div");
    message.className = "empty-state";
    message.textContent = `Failed to load history: ${e.message}`;
    elements.sessionList.replaceChildren(message);
  }
}

export function renderSessions(sessions) {
  elements.sessionList.innerHTML = "";

  if (!sessions.length) {
    elements.sessionList.innerHTML = `<div class="empty-state">No past chats recorded in this workspace yet.</div>`;
    return;
  }

  sessions.forEach((s) => {
    const item = document.createElement("div");
    item.className = "session-item" + (s.open_agent_id ? " current" : "");

    const info = document.createElement("div");
    info.className = "session-info";

    const title = document.createElement("span");
    title.className = "session-title";
    title.textContent = s.title;
    title.title = s.title;

    const meta = document.createElement("div");
    meta.className = "session-meta";

    const parts = [
      formatSessionDate(s.started),
      `${s.turns} ${s.turns === 1 ? "turn" : "turns"}`,
      `${s.probes} ${s.probes === 1 ? "probe" : "probes"}`,
      `${s.notes} ${s.notes === 1 ? "note" : "notes"}`,
    ];
    parts.forEach((text) => {
      const span = document.createElement("span");
      span.textContent = text;
      meta.appendChild(span);
    });

    if (state.carryChatContext && !s.context_restorable) {
      const warn = document.createElement("span");
      warn.className = "warn";
      warn.textContent = "context rebuilt from transcript";
      warn.title =
        "This chat has no stored agent context, so it will be continued from a recap of its transcript.";
      meta.appendChild(warn);
    }

    if (s.open_agent_id) {
      const badge = document.createElement("span");
      badge.textContent = "open";
      meta.appendChild(badge);
    }

    info.appendChild(title);
    info.appendChild(meta);

    const controls = document.createElement("div");
    controls.className = "session-controls";

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn btn-sm btn-success-outline";
    btn.textContent = s.open_agent_id ? "Focus" : "Resume";
    const mode = document.createElement("select");
    mode.className = "model-select-native session-resume-mode";
    mode.setAttribute("aria-label", `Resume mode for ${s.title}`);
    mode.innerHTML = `<option value="chat">Chat only</option><option value="kernel">With kernel</option>`;
    btn.addEventListener("click", () => resumeSession(s.id, mode.value === "kernel", s.open_agent_id));

    item.appendChild(info);
    controls.appendChild(btn);
    controls.appendChild(mode);
    item.appendChild(controls);
    elements.sessionList.appendChild(item);
  });
}

export async function resumeSession(sessionId, restoreKernel = false, openAgentId = null) {
  state.historyLoading = true;
  elements.historyLoading.classList.remove("hidden");
  elements.historyModal.querySelectorAll("button, select").forEach((control) => {
    control.disabled = true;
  });
  try {
    if (!state.activeAgentId && !openAgentId) throw new Error("No explorer is selected");
    if (openAgentId) {
      window.dispatchEvent(new CustomEvent("nocturnomath:focus-explorer", { detail: openAgentId }));
      elements.historyModal.classList.add("hidden");
      return;
    }
    const res = await fetch(`/api/agents/${encodeURIComponent(state.activeAgentId)}/resume`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: sessionId, restore_kernel: restoreKernel }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      alert(`Failed to resume session: ${err.detail || res.status}`);
      return;
    }
    // The chat is repainted by the "session_resumed" broadcast.
    elements.historyModal.classList.add("hidden");
  } catch (e) {
    alert(`Error resuming session: ${e.message}`);
  } finally {
    state.historyLoading = false;
    elements.historyLoading.classList.add("hidden");
    elements.historyModal.querySelectorAll("button, select").forEach((control) => {
      control.disabled = false;
    });
  }
}

const welcomeCard = document.getElementById("system-welcome")?.cloneNode(true);

export function clearChat(showWelcome = true) {
  state.followChat = true;
  elements.chatMessages.replaceChildren();
  if (showWelcome && welcomeCard) {
    elements.chatMessages.appendChild(welcomeCard.cloneNode(true));
  }
  state.currentAssistantBubble = null;
  state.currentProbeCard = null;
  state.lastProbeCard = null;
}

// Rebuild the chat from the transcript the browser reconnected to, so a page
// reload or a dropped socket shows the running session rather than a blank log.
export function restoreChat(records = [], isBusy = false) {
  clearChat(!records.length);
  if (!records.length) return;
  renderRecords(records, isBusy);
  scrollChatToBottom();
}

export function replaySession(records, contextRestored, carryChatContext, kernelReset = false, kernelRestored = false, probesReplayed = 0) {
  clearChat(false);
  renderRecords(records);

  let resumeNote;
  if (!carryChatContext) {
    resumeNote =
      "Reopened this chat. New replies are appended to it, but the agent still starts every query fresh from its prompt and `evidence.md` and `thoughts.md` — it does not read the conversation above.";
  } else if (contextRestored) {
    resumeNote =
      "Resumed this chat. The agent still has its original conversation context.";
  } else {
    resumeNote =
      "Resumed this chat. The agent's original context was not available, so it will pick up from a recap of the transcript above.";
  }
  if (kernelRestored) {
    resumeNote += ` The kernel was reconstructed by replaying ${probesReplayed} stored ${probesReplayed === 1 ? "probe" : "probes"} in order.`;
  } else {
    resumeNote += kernelReset ? " A fresh kernel is ready; previous variables are gone." : " The current kernel is unchanged.";
  }
  appendSystemMessage(resumeNote);
  scrollChatToBottom();
}

function renderRecords(records, isBusy = false) {
  const outcomes = new Set(records.filter((rec) =>
    rec.kind === "run" || rec.kind === "probe_failed").map((rec) => rec.probe_id));
  const starts = new Map(records.filter((rec) => rec.kind === "probe_started")
    .map((rec) => [rec.probe_id, rec]));
  const activeProbe = isBusy
    ? [...starts.keys()].reverse().find((probeId) => !outcomes.has(probeId))
    : null;
  records.forEach((rec) => {
    switch (rec.kind) {
      case "user":
        appendUserMessage(rec.text || "", rec.t);
        break;

      case "agent":
        appendAssistantChunk(rec.text || "", rec.t);
        finalizeAssistantTurn();
        break;

      case "probe_started":
        if (!outcomes.has(rec.probe_id)) {
          if (rec.probe_id === activeProbe) {
            appendProbeStart(rec.expected, rec.code, rec.probe_id);
          } else {
            appendProbeFinish(rec.expected, rec.code,
              `${rec.probe_id}: no recorded outcome. Inspect the saved probe before using it.`, [], [], rec.probe_id);
          }
        }
        break;

      case "probe_failed": {
        const start = starts.get(rec.probe_id) || {};
        appendProbeFinish(start.expected || "", start.code || "",
          `${rec.probe_id}: failed. ${rec.text || ""}`, [], [], rec.probe_id);
        break;
      }

      case "run":
        appendProbeFinish(
          rec.expected || "",
          rec.code || "",
          [rec.probe_id, rec.output, rec.execution_note, rec.status].filter(Boolean).join("\n"),
          (rec.images || []).map((name) => `/api/asset?path=${encodeURIComponent(name)}`),
          [],
          rec.probe_id
        );
        break;

      case "verdict":
        appendProbeVerdict(rec.text || "");
        break;

      case "evidence":
      case "thought":
      case "evidence_struck":
        appendNoteNotification(rec.kind, `${rec.entry_id}: ${rec.text || ""}`);
        break;

      default:
        break;
    }
  });
}
