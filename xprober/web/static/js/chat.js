import { elements, state } from "./state.js?v=20260906-1";
import { highlightBlocks, renderMarkdown } from "./markdown.js?v=20260906-1";
import { openLightbox } from "./ui.js?v=20260906-1";

// ============================================================================
// Chat UI Rendering
// ============================================================================
export function scrollChatToBottom() {
  elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

export function nowLabel() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function appendUserMessage(text, timeLabel) {
  closeAssistantBubble();
  const bubble = document.createElement("div");
  bubble.className = "chat-bubble user";

  const meta = document.createElement("div");
  meta.className = "bubble-meta";
  meta.textContent = `You • ${timeLabel || nowLabel()}`;

  const body = document.createElement("div");
  body.className = "bubble-body";
  body.textContent = text;

  bubble.appendChild(meta);
  bubble.appendChild(body);
  elements.chatMessages.appendChild(bubble);
  scrollChatToBottom();
}

export function appendAssistantChunk(text, timeLabel) {
  if (!state.currentAssistantBubble) {
    const bubble = document.createElement("div");
    bubble.className = "chat-bubble agent";

    const meta = document.createElement("div");
    meta.className = "bubble-meta";
    meta.textContent = `Agent • ${timeLabel || nowLabel()}`;

    const body = document.createElement("div");
    body.className = "bubble-body markdown-body";

    bubble.appendChild(meta);
    bubble.appendChild(body);
    elements.chatMessages.appendChild(bubble);

    state.currentAssistantBubble = {
      container: bubble,
      body: body,
      rawText: "",
    };
  }

  state.currentAssistantBubble.rawText += (state.currentAssistantBubble.rawText ? "\n\n" : "") + text;
  state.currentAssistantBubble.body.innerHTML = renderMarkdown(state.currentAssistantBubble.rawText);
  highlightBlocks(state.currentAssistantBubble.body);
  scrollChatToBottom();
}

export function finalizeAssistantTurn() {
  state.currentAssistantBubble = null;
}

// Consecutive text blocks share a bubble, but anything else in the log ends it.
// Without this the whole turn's prose accumulates in the bubble opened by its
// first block, so a verdict written after five probes renders above all five.
export function closeAssistantBubble() {
  state.currentAssistantBubble = null;
}

export function appendProbeStart(expected, code) {
  closeAssistantBubble();
  const card = document.createElement("div");
  card.className = "probe-card";

  const header = document.createElement("div");
  header.className = "probe-header";

  const pill = document.createElement("span");
  pill.className = "probe-pill";
  pill.textContent = "Code probe";

  const status = document.createElement("span");
  status.className = "badge";
  status.textContent = "Executing";

  header.appendChild(pill);
  header.appendChild(status);

  const expBox = document.createElement("div");
  expBox.className = "probe-expected";
  expBox.innerHTML = `<strong>Expected:</strong> ${expected}`;

  const details = document.createElement("details");
  details.className = "probe-code-details";
  details.open = false;

  const lineCount = (code || "").split("\n").length;
  const summary = document.createElement("summary");
  summary.className = "probe-code-summary";
  summary.textContent = `Python Code Probe (${lineCount} ${lineCount === 1 ? "line" : "lines"})`;

  const pre = document.createElement("pre");
  pre.className = "probe-code-block";
  const codeEl = document.createElement("code");
  codeEl.className = "language-python";
  codeEl.textContent = code;
  pre.appendChild(codeEl);

  details.appendChild(summary);
  details.appendChild(pre);

  card.appendChild(header);
  card.appendChild(expBox);
  card.appendChild(details);

  elements.chatMessages.appendChild(card);
  scrollChatToBottom();
  highlightBlocks(details);

  state.currentProbeCard = {
    card,
    statusBadge: status,
  };
}

export function appendProbeFinish(expected, code, output, plotUrls, plotImages) {
  let card = state.currentProbeCard ? state.currentProbeCard.card : null;

  if (!card) {
    appendProbeStart(expected, code);
    card = state.currentProbeCard.card;
  }

  if (state.currentProbeCard && state.currentProbeCard.statusBadge) {
    state.currentProbeCard.statusBadge.textContent = "Completed";
    state.currentProbeCard.statusBadge.style.color = "var(--ok)";
  }

  // Add output box
  const outBox = document.createElement("pre");
  outBox.className = "probe-output-box";
  outBox.textContent = output || "(No text output)";
  card.appendChild(outBox);

  // Add inline plots if any
  if (plotUrls && plotUrls.length > 0) {
    const imgContainer = document.createElement("div");
    imgContainer.className = "probe-images";

    plotUrls.forEach((url, i) => {
      const img = document.createElement("img");
      img.className = "probe-img-thumb";
      img.src = url;
      img.alt = `Plot ${i + 1}`;
      img.onclick = () => openLightbox(
        url,
        `Plot ${i + 1}`,
        plotUrls.map((src, index) => ({ src, caption: `Plot ${index + 1}` }))
      );
      imgContainer.appendChild(img);
    });

    card.appendChild(imgContainer);
  }

  state.currentProbeCard = null;
  state.lastProbeCard = card;
  scrollChatToBottom();
}

export function appendProbeVerdict(text) {
  const card = state.lastProbeCard;
  if (!card) {
    // No probe to attach to (a stray verdict, or a replayed transcript whose
    // probe was trimmed): show it rather than dropping it.
    appendAssistantChunk(text);
    return;
  }
  const box = document.createElement("div");
  box.className = "probe-verdict";
  const tag = document.createElement("strong");
  tag.textContent = "Verdict:";
  box.appendChild(tag);
  // textContent, not innerHTML: the verdict is model prose and may contain
  // anything, and it is shown verbatim rather than rendered.
  box.appendChild(document.createTextNode(" " + text));
  card.appendChild(box);
  // One verdict per probe, so the next one cannot stack onto this card.
  state.lastProbeCard = null;
  scrollChatToBottom();
}

export function appendNoteNotification(kind, text) {
  closeAssistantBubble();
  const notif = document.createElement("div");
  notif.className = `note-notification ${kind}`;

  const badge = document.createElement("span");
  badge.className = "note-badge";
  badge.textContent = kind === "fact" ? "Fact" : "Dead End";

  const content = document.createElement("span");
  content.textContent = text;

  notif.appendChild(badge);
  notif.appendChild(content);

  elements.chatMessages.appendChild(notif);
  scrollChatToBottom();
}

export function appendSystemMessage(text) {
  closeAssistantBubble();
  const bubble = document.createElement("div");
  bubble.className = "chat-bubble system";

  const body = document.createElement("div");
  body.className = "bubble-body";
  body.innerHTML = renderMarkdown(text);

  bubble.appendChild(body);
  elements.chatMessages.appendChild(bubble);
  scrollChatToBottom();
}

export function appendErrorMessage(msg) {
  closeAssistantBubble();
  const bubble = document.createElement("div");
  bubble.className = "chat-bubble system";

  const body = document.createElement("div");
  body.className = "bubble-body";
  body.style.borderColor = "var(--danger)";
  body.style.color = "var(--danger)";
  body.textContent = `Error: ${msg}`;

  bubble.appendChild(body);
  elements.chatMessages.appendChild(bubble);
  scrollChatToBottom();
}
