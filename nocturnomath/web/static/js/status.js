import { elements, state } from "./state.js?v=20261117-2";

let kernelAlive = true;
let kernelBusy = false;
let agentStatus = "idle";

function syncInterruptButton() {
  const agentInterruptible = agentStatus === "thinking" || agentStatus === "probing";
  elements.btnInterrupt.classList.toggle(
    "hidden",
    !(kernelAlive && kernelBusy) && !agentInterruptible,
  );
}

// ============================================================================
// Status Updaters
// ============================================================================
export function updateKernelStatus(alive, busy) {
  kernelAlive = Boolean(alive);
  kernelBusy = Boolean(busy);
  if (!alive) {
    elements.kernelDot.className = "status-dot red";
    elements.kernelText.textContent = "Kernel dead";
  } else if (busy) {
    elements.kernelDot.className = "status-dot busy";
    elements.kernelText.textContent = "Running code";
  } else {
    elements.kernelDot.className = "status-dot green";
    elements.kernelText.textContent = "Kernel ready";
  }
  syncInterruptButton();
}

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
      ? (status === "compacting" || state.promptMode === "queue" ? "Queue" : "Steer")
      : "Send";
  }
  if (busy) {
    elements.chatStatusBadge.style.color = "var(--blue)";
  } else {
    elements.chatStatusBadge.style.color = "var(--ink-soft)";
  }
  syncInterruptButton();
}
