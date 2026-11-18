import { elements, state } from "./state.js?v=20261117-2";

// ============================================================================
// Status Updaters
// ============================================================================
export function updateKernelStatus(alive, busy) {
  if (!alive) {
    elements.kernelDot.className = "status-dot red";
    elements.kernelText.textContent = "Kernel dead";
    elements.btnInterrupt.classList.add("hidden");
  } else if (busy) {
    elements.kernelDot.className = "status-dot busy";
    elements.kernelText.textContent = "Running code";
    elements.btnInterrupt.classList.remove("hidden");
  } else {
    elements.kernelDot.className = "status-dot green";
    elements.kernelText.textContent = "Kernel ready";
    elements.btnInterrupt.classList.add("hidden");
  }
}

let agentStatus = "idle";

// Re-derive the Send button from the current status after the model
// catalogue changes underneath it.
export function refreshSendButton() {
  updateAgentStatus(agentStatus);
}

export function updateAgentStatus(status) {
  agentStatus = status;
  elements.chatStatusBadge.textContent = status.charAt(0).toUpperCase() + status.slice(1);
  const busy = status === "thinking" || status === "probing" || status === "compacting";
  elements.btnSend.disabled = state.exiting || !elements.modelSelect.value;
  if (elements.sendBtnText) {
    elements.sendBtnText.textContent = busy
      ? (state.promptMode === "queue" ? "Queue" : "Steer")
      : "Send";
  }
  if (busy) {
    elements.chatStatusBadge.style.color = "var(--blue)";
    elements.btnInterrupt.classList.toggle("hidden", status === "compacting");
  } else {
    elements.chatStatusBadge.style.color = "var(--ink-soft)";
    elements.btnInterrupt.classList.add("hidden");
  }
}
