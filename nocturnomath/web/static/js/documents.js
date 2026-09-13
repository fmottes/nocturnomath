import { elements, state } from "./state.js?v=20260913-1";
import { highlightBlocks, renderMarkdown, renderMath } from "./markdown.js?v=20260913-2";
import { rewriteDocumentLinks, rewriteEmbeddedImageUrls } from "./workspace.js?v=20260913-1";

// ============================================================================
// Documents tab: a list of .nocturnomath/documents/*.md with a checkbox that
// decides whether each one travels with the next message, a reader, and a
// minimal editor. Views: "list", "read", "edit" (existing), "create" (new).
// ============================================================================

let listSignature = null;

export function setDocuments(documents = [], documentsDefault = state.documentsDefault) {
  state.documents = documents;
  state.documentsDefault = Boolean(documentsDefault);
  elements.documentsDefaultToggle.checked = state.documentsDefault;
  const signature = JSON.stringify(documents.map((d) => [d.name, d.included, d.modified, d.size]));
  if (signature !== listSignature) {
    listSignature = signature;
    renderDocumentsList();
  }
  if (state.openDocument && !documents.some((d) => d.name === state.openDocument)) {
    showDocumentsView("list");
  }
}

export async function fetchDocuments() {
  const res = await fetch("/api/documents");
  if (!res.ok) return;
  const data = await res.json();
  setDocuments(data.documents || [], data.documents_default);
}

export function showDocumentsView(view) {
  state.documentsView = view;
  if (view === "list") state.openDocument = null;
  const reading = view === "read";
  const editing = view === "edit" || view === "create";
  elements.documentsList.classList.toggle("hidden", view !== "list");
  elements.documentsHint.classList.toggle("hidden", view !== "list");
  elements.documentsMarkdown.classList.toggle("hidden", !reading);
  elements.documentsEditor.classList.toggle("hidden", !editing);
  elements.documentsEditorTitle.classList.toggle("hidden", view !== "create");
  elements.documentsBack.classList.toggle("hidden", view === "list");
  elements.documentsNew.classList.toggle("hidden", view !== "list");
  elements.documentsModify.classList.toggle("hidden", !reading);
  elements.documentsError.classList.add("hidden");
  elements.documentsTitle.textContent = view === "list" ? "Documents"
    : view === "create" ? "New document" : state.openDocument || "";
  if (!reading) elements.documentsMeta.textContent = "";
}

function formatMeta(document) {
  const time = new Date(document.modified * 1000).toLocaleString([], {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
  return `${time} · ${document.size} B`;
}

function renderDocumentsList() {
  const list = elements.documentsList;
  list.replaceChildren();
  if (!state.documents.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "No documents yet. Create one, or drop Markdown files in .nocturnomath/documents/.";
    list.appendChild(empty);
    return;
  }
  state.documents.forEach((doc) => {
    const row = document.createElement("div");
    row.className = "document-row";

    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = Boolean(doc.included);
    box.title = "Send with the next message";
    box.setAttribute("aria-label", `Include ${doc.name} in context`);
    box.addEventListener("change", () => toggleIncluded(doc.name, box.checked, box));

    const title = document.createElement("button");
    title.type = "button";
    title.className = "document-row-title";
    title.textContent = doc.name.replace(/\.md$/, "");
    title.title = doc.path;
    title.addEventListener("click", () => openDocument(doc.name));

    const meta = document.createElement("span");
    meta.className = "document-row-meta";
    meta.textContent = formatMeta(doc);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "document-row-delete";
    remove.title = `Delete ${doc.name}`;
    remove.setAttribute("aria-label", `Delete ${doc.name}`);
    remove.innerHTML = TRASH_ICON;
    remove.addEventListener("click", () => deleteDocument(doc.name));

    row.append(box, title, meta, remove);
    list.appendChild(row);
  });
}

// A small drafted bin: lid, body, two slats. Inherits the row's ink colour.
const TRASH_ICON = '<svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" aria-hidden="true">'
  + '<path d="M2.5 4.5h11M6 4.5V3h4v1.5M4 4.5l.7 9h6.6l.7-9M6.7 7v4.5M9.3 7v4.5"/></svg>';

async function deleteDocument(name) {
  if (!window.confirm(`Delete ${name}? This removes the file from .nocturnomath/documents/.`)) return;
  try {
    const res = await fetch(`/api/documents/${encodeURIComponent(name)}`, { method: "DELETE" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    setDocuments(data.documents || [], data.documents_default);
  } catch (error) {
    console.error("Unable to delete document:", error);
    showDocumentError(error.message);
  }
}

async function toggleIncluded(name, included, box) {
  try {
    const res = await fetch(`/api/documents/${encodeURIComponent(name)}/include`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ included }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || `HTTP ${res.status}`);
    const doc = state.documents.find((d) => d.name === name);
    if (doc) doc.included = included;
    listSignature = null;
  } catch (error) {
    console.error("Unable to change document inclusion:", error);
    box.checked = !included;
  }
}

export async function setDocumentsDefault(enabled) {
  try {
    const res = await fetch("/api/documents/default", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || `HTTP ${res.status}`);
    const data = await res.json();
    setDocuments(data.documents || [], data.documents_default);
  } catch (error) {
    console.error("Unable to change the documents default:", error);
    elements.documentsDefaultToggle.checked = state.documentsDefault;
  }
}

export async function openDocument(name, quiet = false) {
  const res = await fetch(`/api/documents/${encodeURIComponent(name)}`);
  if (!res.ok) {
    if (!quiet) {
      const message = document.createElement("p");
      message.className = "empty-state";
      message.textContent = `Document not found: ${name}`;
      elements.documentsMarkdown.replaceChildren(message);
      state.openDocument = name;
      showDocumentsView("read");
    }
    return;
  }
  const data = await res.json();
  if (quiet && (state.openDocument !== name || window.getSelection()?.isCollapsed === false)) return;
  if (quiet && state.openDocumentContent === data.content) return;
  state.openDocument = name;
  state.openDocumentContent = data.content;
  elements.documentsMarkdown.innerHTML = renderMarkdown(data.content || "", { breaks: false });
  rewriteEmbeddedImageUrls(elements.documentsMarkdown, data.path);
  rewriteDocumentLinks(elements.documentsMarkdown, data.path);
  highlightBlocks(elements.documentsMarkdown);
  renderMath(elements.documentsMarkdown);
  if (!quiet) showDocumentsView("read");
  elements.documentsMeta.textContent = `Updated: ${new Date(data.modified * 1000).toLocaleTimeString()} (${data.size} B)`;
}

export function refreshOpenDocument() {
  if (state.documentsView === "read" && state.openDocument) return openDocument(state.openDocument, true);
  return Promise.resolve();
}

export function startCreate() {
  elements.documentsEditorTitle.value = "";
  elements.documentsEditorText.value = "";
  showDocumentsView("create");
  elements.documentsEditorTitle.focus();
}

export function startModify() {
  if (!state.openDocument) return;
  elements.documentsEditorText.value = state.openDocumentContent || "";
  showDocumentsView("edit");
  elements.documentsEditorText.focus();
}

export function cancelEdit() {
  showDocumentsView(state.openDocument ? "read" : "list");
}

function showDocumentError(message) {
  elements.documentsError.textContent = message;
  elements.documentsError.classList.remove("hidden");
}

export async function saveEdit() {
  const text = elements.documentsEditorText.value;
  const creating = state.documentsView === "create";
  const url = creating ? "/api/documents" : `/api/documents/${encodeURIComponent(state.openDocument)}`;
  const body = creating ? { title: elements.documentsEditorTitle.value, text } : { text };
  elements.documentsSave.disabled = true;
  try {
    const res = await fetch(url, {
      method: creating ? "POST" : "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    if (creating) setDocuments(data.documents || [], data.documents_default);
    else await fetchDocuments();
    await openDocument(creating ? data.name : state.openDocument);
  } catch (error) {
    showDocumentError(error.message);
  } finally {
    elements.documentsSave.disabled = false;
  }
}
