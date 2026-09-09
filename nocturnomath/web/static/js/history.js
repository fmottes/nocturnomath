import { elements, state } from "./state.js?v=20260906-1";
import { appendAssistantChunk, appendNoteNotification, appendProbeFinish, appendProbeVerdict, appendSystemMessage, appendUserMessage, finalizeAssistantTurn, scrollChatToBottom } from "./chat.js?v=20260906-1";

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
    elements.sessionList.innerHTML = `<div class="empty-state">Failed to load history: ${e.message}</div>`;
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
    item.className = "session-item" + (s.is_current ? " current" : "");

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

    if (s.is_current) {
      const badge = document.createElement("span");
      badge.textContent = "current";
      meta.appendChild(badge);
    }

    info.appendChild(title);
    info.appendChild(meta);

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn btn-xs btn-outline";
    btn.textContent = s.is_current ? "Reload" : "Resume";
    btn.addEventListener("click", () => resumeSession(s.id));

    item.appendChild(info);
    item.appendChild(btn);
    elements.sessionList.appendChild(item);
  });
}

export async function resumeSession(sessionId) {
  try {
    const res = await fetch("/api/session/resume", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: sessionId }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      alert(`Failed to resume chat: ${err.detail || res.status}`);
      return;
    }
    // The chat is repainted by the "session_resumed" broadcast.
    elements.historyModal.classList.add("hidden");
  } catch (e) {
    alert(`Error resuming chat: ${e.message}`);
  }
}

export function clearChat() {
  state.followChat = true;
  elements.chatMessages.innerHTML = "";
  state.currentAssistantBubble = null;
  state.currentProbeCard = null;
  state.lastProbeCard = null;
}

export function replaySession(records, contextRestored, carryChatContext) {
  clearChat();

  const outcomes = new Set(records.filter((rec) =>
    rec.kind === "run" || rec.kind === "probe_failed").map((rec) => rec.probe_id));
  const starts = new Map(records.filter((rec) => rec.kind === "probe_started")
    .map((rec) => [rec.probe_id, rec]));
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
          appendProbeFinish(rec.expected, rec.code,
            `${rec.probe_id}: no recorded outcome. Inspect the saved probe before using it.`, [], []);
        }
        break;

      case "probe_failed": {
        const start = starts.get(rec.probe_id) || {};
        appendProbeFinish(start.expected || "", start.code || "",
          `${rec.probe_id}: failed. ${rec.text || ""}`, [], []);
        break;
      }

      case "run":
        appendProbeFinish(
          rec.expected || "",
          rec.code || "",
          [rec.probe_id, rec.output, rec.execution_note, rec.status].filter(Boolean).join("\n"),
          (rec.images || []).map((name) => `/api/asset?path=${encodeURIComponent(name)}`),
          []
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

  let resumeNote;
  if (!carryChatContext) {
    resumeNote =
      "Reopened this chat. New replies are appended to it, but the agent still starts every query fresh from its prompt and `evidence.md` and `thoughts.md` — it does not read the conversation above.";
  } else if (contextRestored) {
    resumeNote =
      "Resumed this chat. The agent still has its original context; the kernel keeps whatever state it holds now.";
  } else {
    resumeNote =
      "Resumed this chat. The agent's original context was not available, so it will pick up from a recap of the transcript above.";
  }
  appendSystemMessage(resumeNote);
  scrollChatToBottom();
}
