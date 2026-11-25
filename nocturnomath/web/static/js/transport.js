import { state } from "./state.js";

export function agentQuery(agentId = state.activeAgentId) {
  return agentId ? `?agent_id=${encodeURIComponent(agentId)}` : "";
}

export async function apiJson(url, options = {}) {
  const { fallbackError, body, headers, ...rest } = options;
  const fetchOptions = { ...rest };
  if (body !== undefined && typeof body !== "string") {
    fetchOptions.headers = { "Content-Type": "application/json", ...(headers || {}) };
    fetchOptions.body = JSON.stringify(body);
  } else {
    if (headers) fetchOptions.headers = headers;
    if (body !== undefined) fetchOptions.body = body;
  }
  const response = await fetch(url, fetchOptions);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map((item) => item?.msg || JSON.stringify(item)).join("; ")
      : data.detail;
    const error = new Error(detail || fallbackError || `HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return data;
}

export function initWebSocket(onMessage, onOpen, onClose) {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  state.ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
  state.ws.onopen = () => {
    console.log("WebSocket connected");
    onOpen();
  };
  state.ws.onmessage = (event) => {
    try {
      onMessage(JSON.parse(event.data));
    } catch (error) {
      console.error("Failed to parse websocket message:", error);
    }
  };
  state.ws.onclose = () => {
    if (state.exiting) return;
    console.log("WebSocket closed, attempting reconnect in 2s...");
    onClose();
    setTimeout(() => initWebSocket(onMessage, onOpen, onClose), 2000);
  };
  state.ws.onerror = (error) => console.warn("WebSocket error:", error);
}

export function sendWs(action, payload = {}) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ action, ...payload }));
    return true;
  }
  console.warn("WebSocket is not connected");
  return false;
}
