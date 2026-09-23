import { elements, state } from "./state.js?v=20260913-1";
import { highlightBlocks, renderMarkdown, renderMath } from "./markdown.js?v=20260922-1";
import { openLightbox } from "./ui.js?v=20260913-1";

// ============================================================================
// Chat UI Rendering
// ============================================================================
export function scrollChatToBottom() {
  if (state.followChat) elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

export function nowLabel() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function appendUserMessage(text, timeLabel) {
  closeAssistantBubble();
  document.getElementById("system-welcome")?.remove();
  const bubble = document.createElement("div");
  bubble.className = "chat-bubble user";

  const meta = document.createElement("div");
  meta.className = "bubble-meta";
  meta.textContent = `You • ${timeLabel || nowLabel()}`;

  const body = document.createElement("div");
  body.className = "bubble-body";
  body.textContent = text;
  renderMath(body);

  bubble.appendChild(meta);
  bubble.appendChild(body);
  elements.chatMessages.appendChild(bubble);
  scrollChatToBottom();
}

function openAssistantBubble(timeLabel) {
  if (state.currentAssistantBubble) return state.currentAssistantBubble;

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
    streamText: "",
  };
  return state.currentAssistantBubble;
}

function renderAssistantBubble(bubble) {
  const separator = bubble.rawText && bubble.streamText ? "\n\n" : "";
  bubble.body.innerHTML = renderMarkdown(bubble.rawText + separator + bubble.streamText);
  highlightBlocks(bubble.body);
  renderMath(bubble.body);
  scrollChatToBottom();
}

// Live streaming: the chunk extends the block currently being written, which
// assistant_text later replaces with its final form.
export function appendAssistantDelta(text) {
  const bubble = openAssistantBubble();
  bubble.streamText += text;
  renderAssistantBubble(bubble);
}

export function appendAssistantChunk(text, timeLabel) {
  const bubble = openAssistantBubble(timeLabel);
  // The completed block is authoritative: drop whatever was streamed for it
  // rather than appending, so streamed text is never duplicated.
  bubble.streamText = "";
  bubble.rawText += (bubble.rawText ? "\n\n" : "") + text;
  renderAssistantBubble(bubble);
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

export function appendProbeStart(expected, code, probeId) {
  closeAssistantBubble();
  const card = document.createElement("div");
  card.className = "probe-card";

  const header = document.createElement("div");
  header.className = "probe-header";

  const pill = document.createElement("span");
  pill.className = "probe-pill";
  pill.textContent = "Code probe";
  const number = probeId?.split("/").pop();
  if (/^P\d+$/.test(number || "")) {
    const badge = document.createElement("span");
    badge.className = "badge badge-sm";
    badge.textContent = number;
    pill.appendChild(badge);
  }

  const status = document.createElement("span");
  status.className = "badge";
  status.textContent = "Executing";

  header.appendChild(pill);
  header.appendChild(status);

  const expBox = document.createElement("div");
  expBox.className = "probe-expected";
  const expectedLabel = document.createElement("strong");
  expectedLabel.textContent = "Expected:";
  expBox.append(expectedLabel, document.createTextNode(` ${expected}`));

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

export function appendProbeFinish(expected, code, output, plotUrls, plotImages, probeId) {
  let card = state.currentProbeCard ? state.currentProbeCard.card : null;

  if (!card) {
    appendProbeStart(expected, code, probeId);
    card = state.currentProbeCard.card;
  }

  if (state.currentProbeCard && state.currentProbeCard.statusBadge) {
    state.currentProbeCard.statusBadge.textContent = "Completed";
    state.currentProbeCard.statusBadge.style.color = "var(--ok)";
  }

  // Keep text output available without letting long probe logs dominate the chat.
  const outputDetails = document.createElement("details");
  outputDetails.className = "probe-output-details";
  outputDetails.open = false;

  const outputText = output || "(No text output)";
  const outputSummary = document.createElement("summary");
  outputSummary.className = "probe-output-summary";
  outputSummary.textContent = "Text Output";

  const outBox = document.createElement("pre");
  outBox.className = "probe-output-box";
  outBox.textContent = outputText;

  outputDetails.appendChild(outputSummary);
  outputDetails.appendChild(outBox);
  card.appendChild(outputDetails);

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
  badge.textContent = ({ evidence: "Evidence", thought: "Thought", evidence_struck: "Struck" })[kind] || kind;

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
  renderMath(body);

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
