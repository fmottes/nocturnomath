import { state } from "./state.js?v=20260909-2";

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
  } else {
    console.warn("WebSocket is not connected");
  }
}
