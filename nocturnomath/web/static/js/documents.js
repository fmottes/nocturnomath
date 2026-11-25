import { elements, state } from "./state.js";
import { highlightBlocks, renderMarkdown, renderMath } from "./markdown.js";
import { rewriteDocumentLinks, rewriteEmbeddedImageUrls } from "./workspace.js";
import { agentQuery, apiJson } from "./transport.js";

// ============================================================================
// Documents tab: a folder browser for .nocturnomath/documents/ with a checkbox that
// decides whether each one travels with the next message, a reader, and a
// minimal editor. Views: "list", "read", "edit" (existing), "create" (new).
// ============================================================================

let listSignature = null;
const AUTOSAVE_DELAY_MS = 600;
let autosaveTimer = null;
let autosavePromise = null;
let autosaveNeedsAnother = false;
let lastSavedEditorText = null;

export function setDocuments(documents = [], documentsDefault = state.documentsDefault, folders = state.documentFolders) {
  state.documents = documents;
  state.documentFolders = folders;
  state.documentsDefault = Boolean(documentsDefault);
  elements.documentsDefaultToggle.checked = state.documentsDefault;
  while (state.documentsFolder && !folders.includes(state.documentsFolder)) {
    state.documentsFolder = parentFolder(state.documentsFolder);
  }
  const signature = JSON.stringify({
    documents: documents.map((d) => [d.name, d.included, d.modified, d.size]),
    folders,
    current: state.documentsFolder,
  });
  if (signature !== listSignature) {
    listSignature = signature;
    renderDocumentsList();
    renderBreadcrumbs();
  }
  if (state.openDocument && !documents.some((d) => d.name === state.openDocument)) {
    showDocumentsView("list");
  }
}

export async function fetchDocuments() {
  const res = await fetch(`/api/documents${agentQuery()}`);
  if (!res.ok) return;
  const data = await res.json();
  setDocuments(data.documents || [], data.documents_default, data.folders || []);
}

function parentFolder(path) {
  return path.includes("/") ? path.slice(0, path.lastIndexOf("/")) : "";
}

function renderBreadcrumbs() {
  const nav = elements.documentsBreadcrumbs;
  nav.replaceChildren();
  const crumbs = [{ path: "", label: "Documents" }];
  let path = "";
  for (const part of state.documentsFolder.split("/").filter(Boolean)) {
    path = path ? `${path}/${part}` : part;
    crumbs.push({ path, label: part });
  }
  crumbs.forEach((crumb, index) => {
    if (index) {
      const separator = document.createElement("span");
      separator.textContent = "/";
      separator.setAttribute("aria-hidden", "true");
      nav.append(separator);
    }
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = crumb.label;
    button.disabled = crumb.path === state.documentsFolder && state.documentsView === "list";
    button.addEventListener("click", () => navigateFolder(crumb.path));
    nav.append(button);
  });
}

export async function navigateFolder(path) {
  if (path && !state.documentFolders.includes(path)) return;
  if (state.documentsView === "edit") await flushAutosave();
  state.documentsFolder = path;
  listSignature = null;
  showDocumentsView("list");
}

export function showDocumentsView(view) {
  state.documentsView = view;
  if (view === "list") {
    state.openDocument = null;
    if (listSignature === null) renderDocumentsList();
  }
  const reading = view === "read";
  const editing = view === "edit" || view === "create";
  elements.documentsList.classList.toggle("hidden", view !== "list");
  elements.documentsHint.classList.toggle("hidden", view !== "list");
  elements.documentsMarkdown.classList.toggle("hidden", !reading);
  elements.documentsEditor.classList.toggle("hidden", !editing);
  elements.documentsEditorTitle.classList.toggle("hidden", view !== "create");
  elements.documentsCancel.classList.toggle("hidden", view === "edit");
  elements.documentsSave.textContent = view === "edit" ? "Done" : "Create";
  elements.documentsBack.classList.toggle("hidden", view === "list");
  elements.documentsNew.classList.toggle("hidden", view !== "list");
  elements.documentsNewFolder.classList.toggle("hidden", view !== "list");
  if (view !== "list") elements.documentsFolderForm.classList.add("hidden");
  elements.documentsModify.classList.toggle("hidden", !reading);
  elements.documentsError.classList.add("hidden");
  elements.documentsTitle.textContent = view === "list" ? "Folder"
    : view === "create" ? "New document" : state.openDocument?.split("/").at(-1) || "";
  if (!reading && view !== "edit") elements.documentsMeta.textContent = "";
  renderBreadcrumbs();
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
  const folders = state.documentFolders.filter((path) => parentFolder(path) === state.documentsFolder);
  const documents = state.documents.filter((doc) => parentFolder(doc.name) === state.documentsFolder);
  if (!folders.length && !documents.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "This folder is empty. Create a document or folder here.";
    list.appendChild(empty);
    return;
  }
  folders.forEach((path) => {
    const row = document.createElement("div");
    row.className = "document-row document-folder-row";
    const button = document.createElement("button");
    button.type = "button";
    button.className = "document-row-title";
    const icon = document.createElement("span");
    icon.className = "document-folder-icon";
    icon.innerHTML = FOLDER_ICON;
    const label = document.createElement("span");
    label.className = "document-folder-name";
    label.textContent = path.split("/").at(-1);
    button.append(icon, label);
    button.addEventListener("click", () => navigateFolder(path));
    row.append(button);
    list.append(row);
  });
  documents.forEach((doc) => {
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
    title.textContent = doc.name.split("/").at(-1).replace(/\.md$/, "");
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
    remove.addEventListener("click", () => confirmDeleteDocument(row, doc.name));

    row.append(box, title, meta, remove);
    list.appendChild(row);
  });
}

// Drafted 1px line icons that inherit the row's ink colour.
const FOLDER_ICON = '<svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
  + '<path d="M2 4.2v8.3h12V5.8H7.6L6.3 4.2H2z"/></svg>';

const TRASH_ICON = '<svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" aria-hidden="true">'
  + '<path d="M2.5 4.5h11M6 4.5V3h4v1.5M4 4.5l.7 9h6.6l.7-9M6.7 7v4.5M9.3 7v4.5"/></svg>';

function clearDeleteConfirm(row) {
  row.classList.remove("confirming-delete");
  row.querySelector(".document-row-confirm")?.remove();
}

function confirmDeleteDocument(row, name) {
  elements.documentsList.querySelectorAll(".document-row.confirming-delete").forEach(clearDeleteConfirm);
  row.classList.add("confirming-delete");

  const actions = document.createElement("span");
  actions.className = "document-row-confirm";

  const confirmBtn = document.createElement("button");
  confirmBtn.type = "button";
  confirmBtn.className = "btn btn-xs btn-danger-outline";
  confirmBtn.textContent = "Delete";
  confirmBtn.setAttribute("aria-label", `Confirm delete ${name}`);

  const cancelBtn = document.createElement("button");
  cancelBtn.type = "button";
  cancelBtn.className = "btn btn-xs btn-ghost";
  cancelBtn.textContent = "Cancel";

  const cancel = () => {
    clearDeleteConfirm(row);
    row.querySelector(".document-row-delete")?.focus();
  };
  cancelBtn.addEventListener("click", cancel);
  actions.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      event.preventDefault();
      cancel();
    }
  });
  confirmBtn.addEventListener("click", async () => {
    confirmBtn.disabled = true;
    cancelBtn.disabled = true;
    await deleteDocument(name, row);
  });

  actions.append(confirmBtn, cancelBtn);
  row.append(actions);
  confirmBtn.focus();
}

async function deleteDocument(name, row) {
  try {
    const data = await apiJson(`/api/documents/${encodeURIComponent(name)}${agentQuery()}`, { method: "DELETE" });
    setDocuments(data.documents || [], data.documents_default, data.folders || []);
  } catch (error) {
    console.error("Unable to delete document:", error);
    if (row?.isConnected) clearDeleteConfirm(row);
    showDocumentError(error.message);
  }
}

async function toggleIncluded(name, included, box) {
  try {
    await apiJson(`/api/documents/${encodeURIComponent(name)}/include${agentQuery()}`, {
      method: "POST",
      body: { included },
    });
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
    const data = await apiJson(`/api/documents/default${agentQuery()}`, {
      method: "POST",
      body: { enabled },
    });
    setDocuments(data.documents || [], data.documents_default, data.folders || []);
  } catch (error) {
    console.error("Unable to change the documents default:", error);
    elements.documentsDefaultToggle.checked = state.documentsDefault;
  }
}

function renderOpenDocument(name, data, { showView = true } = {}) {
  const nextFolder = parentFolder(name);
  if (nextFolder !== state.documentsFolder) {
    state.documentsFolder = nextFolder;
    listSignature = null;
  }
  state.openDocument = name;
  state.openDocumentContent = data.content;
  elements.documentsMarkdown.innerHTML = renderMarkdown(data.content || "", { breaks: false });
  rewriteEmbeddedImageUrls(elements.documentsMarkdown, data.path);
  rewriteDocumentLinks(elements.documentsMarkdown, data.path);
  highlightBlocks(elements.documentsMarkdown);
  renderMath(elements.documentsMarkdown);
  if (showView) showDocumentsView("read");
  elements.documentsMeta.textContent = `Updated: ${new Date(data.modified * 1000).toLocaleTimeString()} (${data.size} B)`;
}

export async function openDocument(name, quiet = false) {
  const res = await fetch(`/api/documents/${encodeURIComponent(name)}${agentQuery()}`);
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
  renderOpenDocument(name, data, { showView: !quiet });
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

export function startCreateFolder() {
  elements.documentsFolderName.value = "";
  elements.documentsFolderForm.classList.remove("hidden");
  elements.documentsError.classList.add("hidden");
  elements.documentsFolderName.focus();
}

export function cancelCreateFolder() {
  elements.documentsFolderForm.classList.add("hidden");
}

export async function saveFolder() {
  const name = elements.documentsFolderName.value.trim();
  if (!name || name.includes("/") || name.includes("\\")) {
    showDocumentError("Enter one folder name without a slash.");
    return;
  }
  const path = state.documentsFolder ? `${state.documentsFolder}/${name}` : name;
  try {
    const data = await apiJson(`/api/document-folders${agentQuery()}`, {
      method: "POST",
      body: { path },
    });
    setDocuments(data.documents || [], data.documents_default, data.folders || []);
    cancelCreateFolder();
    navigateFolder(path);
  } catch (error) {
    showDocumentError(error.message);
  }
}

export function startModify() {
  if (!state.openDocument) return;
  elements.documentsEditorText.value = state.openDocumentContent || "";
  lastSavedEditorText = elements.documentsEditorText.value;
  showDocumentsView("edit");
  elements.documentsMeta.textContent = "Autosave on";
  elements.documentsEditorText.focus();
}

export function scheduleAutosave() {
  if (state.documentsView !== "edit" || !state.openDocument) return;
  clearTimeout(autosaveTimer);
  elements.documentsMeta.textContent = "Unsaved…";
  autosaveTimer = setTimeout(() => {
    autosaveTimer = null;
    flushAutosave();
  }, AUTOSAVE_DELAY_MS);
}

export async function flushAutosave() {
  if (autosaveTimer !== null) {
    clearTimeout(autosaveTimer);
    autosaveTimer = null;
  }
  if (state.documentsView !== "edit" || !state.openDocument) return null;
  if (autosavePromise) {
    autosaveNeedsAnother = true;
    return autosavePromise;
  }

  const run = async () => {
    let lastData = null;
    do {
      autosaveNeedsAnother = false;
      const name = state.openDocument;
      if (!name || state.documentsView !== "edit") break;
      const text = elements.documentsEditorText.value;
      if (text === lastSavedEditorText || text === state.openDocumentContent) break;
      elements.documentsMeta.textContent = "Saving…";
      elements.documentsError.classList.add("hidden");
      try {
        const data = await apiJson(`/api/documents/${encodeURIComponent(name)}${agentQuery()}`, {
          method: "PUT",
          body: { text },
        });
        lastData = data;
        if (state.openDocument === name) {
          state.openDocumentContent = data.content;
          lastSavedEditorText = text;
          if (elements.documentsEditorText.value === text) {
            elements.documentsMeta.textContent = `Saved · ${new Date(data.modified * 1000).toLocaleTimeString()}`;
          }
        }
        fetchDocuments();
      } catch (error) {
        elements.documentsMeta.textContent = "Save failed";
        showDocumentError(error.message);
        throw error;
      }
    } while (autosaveNeedsAnother);
    return lastData;
  };

  autosavePromise = run().finally(() => {
    autosavePromise = null;
  });
  return autosavePromise;
}

export async function finishEdit(returnTo = "read") {
  if (state.documentsView !== "edit" || !state.openDocument) {
    showDocumentsView(returnTo === "list" ? "list" : (state.openDocument ? "read" : "list"));
    return;
  }
  elements.documentsSave.disabled = true;
  try {
    const saved = await flushAutosave();
    if (returnTo === "list") {
      showDocumentsView("list");
    } else if (saved) {
      renderOpenDocument(state.openDocument, saved, { showView: true });
    } else {
      await openDocument(state.openDocument);
    }
  } catch {
    // Error is already shown in #documents-error; stay in edit view.
  } finally {
    elements.documentsSave.disabled = false;
  }
}

export async function cancelEdit() {
  if (state.documentsView === "edit") {
    await finishEdit("read");
    return;
  }
  showDocumentsView(state.openDocument ? "read" : "list");
}

function showDocumentError(message) {
  elements.documentsError.textContent = message;
  elements.documentsError.classList.remove("hidden");
}

export async function saveEdit() {
  if (state.documentsView === "edit") {
    await finishEdit("read");
    return;
  }
  const text = elements.documentsEditorText.value;
  elements.documentsSave.disabled = true;
  try {
    const data = await apiJson(`/api/documents${agentQuery()}`, {
      method: "POST",
      body: { title: elements.documentsEditorTitle.value, text, folder: state.documentsFolder },
    });
    setDocuments(data.documents || [], data.documents_default, data.folders || []);
    await openDocument(data.name);
  } catch (error) {
    showDocumentError(error.message);
  } finally {
    elements.documentsSave.disabled = false;
  }
}
