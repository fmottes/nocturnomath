import { elements, state } from "./state.js?v=20260913-1";
import { highlightBlocks, renderMarkdown } from "./markdown.js?v=20260913-2";
import { openLightbox } from "./ui.js?v=20260913-1";
import { updateAgentStatus, updateKernelStatus } from "./status.js?v=20260913-1";
import { fetchDocuments, refreshOpenDocument, setDocuments, showDocumentsView } from "./documents.js?v=20260913-1";

// ============================================================================
// Workspace & Files Management
// ============================================================================
export function applyWorkspace(ws) {
  document.getElementById("app-loading").classList.add("hidden");
  setAuthStatus(ws?.auth);
  if (!ws || !ws.is_open) {
    showLanding(ws);
    return;
  }
  if (state.workspace?.path !== ws.path) {
    state.documentContent = null;
    state.documentPath = null;
    state.openDocumentContent = null;
    state.sessionDefaults = null;
    elements.modelSelect.value = "";
    elements.effortSelect.value = "";
    elements.sessionDefaultsStatus.textContent = "";
    showDocumentsView("list");
  }
  state.workspace = ws;
  document.getElementById("environment-info").textContent = ws.environment
    ? `Python ${ws.environment.version} — ${ws.environment.python}` : "";
  document.getElementById("environment-python").value = ws.environment?.python || "";
  recordRecentWorkspace(ws.path);
  elements.landingScreen.classList.add("hidden");
  elements.workspaceApp.classList.remove("hidden");

  elements.workspacePath.textContent = workspaceName(ws.path);
  elements.workspacePill.title = ws.path;
  setSessionId(ws.session_id);
  setModelCatalogue(
    ws.models || [],
    ws.model_labels || {},
    ws.model,
    ws.model_efforts || {},
    ws.effort,
  );
  setSessionDefaults(ws.session_defaults);

  updateKernelStatus(ws.kernel_alive, ws.kernel_busy);
  updateAgentStatus(ws.is_busy ? "thinking" : "idle");
  setCarryContext(ws.carry_chat_context);

  if (![ws.evidence_path, ws.thoughts_path].includes(state.activeDoc)) {
    state.activeDoc = ws.evidence_path;
  }
  setDocuments(ws.documents || [], ws.documents_default);

  // Refresh the record view without pulling the user off the figures or documents tab.
  loadDocument(state.activeDoc, "", ["plots", "documents"].includes(state.activeTab));

  // Refresh plots list
  loadPlots();
}

export function setSessionId(id) {
  state.currentSession = id || null;
  elements.sessionIdBadge.textContent = id || "";
}

function workspaceName(path) {
  const parts = String(path || "").split(/[\\/]+/).filter(Boolean);
  return parts.at(-1) || path;
}

export function showLanding(ws) {
  document.getElementById("app-loading").classList.add("hidden");
  state.workspace = null;
  state.navigatorRoot = ws?.navigator_root || state.navigatorRoot || ".";
  elements.workspaceApp.classList.add("hidden");
  elements.landingScreen.classList.remove("hidden");
  renderRecentWorkspaces();
}

export function setAuthStatus(auth) {
  if (!auth) return;
  state.auth = auth;
  const label = auth.label || "Claude authentication";
  elements.btnAuth.textContent = label;
  elements.btnAuthLanding.textContent = label;
  elements.authSummary.textContent = `Current: ${label}.${auth.note ? ` ${auth.note}` : ""}`;
  elements.authCurrent.textContent = `Current: ${label}`;
  elements.authNote.textContent = auth.note || "";
  elements.authNote.classList.toggle("hidden", !auth.note);
}

function fillModelSelect(select, models, labels, preferred) {
  select.replaceChildren(...models.map((model) => new Option(labels[model] || model, model)));
  if (!models.length) select.add(new Option("No model available", ""));
  select.disabled = !models.length;
  select.value = models.includes(preferred) ? preferred : models[0] || "";
}

export function setEffortForModel(select, model, preferred = null) {
  const efforts = state.modelEfforts[model] || [];
  select.replaceChildren(...efforts.map((effort) => new Option(effort, effort)));
  if (!efforts.length) select.add(new Option("No effort setting", ""));
  select.disabled = !efforts.length;
  select.value = efforts.includes(preferred)
    ? preferred
    : efforts.includes("high")
      ? "high"
      : efforts[0] || "";
}

export function setActiveModelAndEffort(model, effort) {
  if (state.models.includes(model)) elements.modelSelect.value = model;
  setEffortForModel(elements.effortSelect, elements.modelSelect.value, effort);
}

export function setSessionDefaults(defaults) {
  state.sessionDefaults = defaults || null;
  const preferredModel = defaults?.model || state.models[0] || "";
  fillModelSelect(elements.defaultModelSelect, state.models, state.modelLabels, preferredModel);
  setEffortForModel(elements.defaultEffortSelect, elements.defaultModelSelect.value, defaults?.effort);
  elements.sessionDefaultsSave.disabled = !state.models.length;
}

export function setModelCatalogue(
  models = [],
  labels = {},
  serverModel = null,
  modelEfforts = {},
  serverEffort = null,
) {
  // The dropdown decides the model of the next message, so an unsent choice
  // survives a catalogue refresh; otherwise follow the server's model.
  const currentModel = elements.modelSelect.value;
  const currentEffort = elements.effortSelect.value;
  state.models = models;
  state.modelLabels = labels;
  state.modelEfforts = modelEfforts;
  const preferred = [currentModel, serverModel].find((model) => models.includes(model));
  fillModelSelect(elements.modelSelect, models, labels, preferred);
  setEffortForModel(
    elements.effortSelect,
    elements.modelSelect.value,
    currentModel === elements.modelSelect.value ? currentEffort : serverEffort,
  );
  if (state.sessionDefaults) setSessionDefaults(state.sessionDefaults);
}

const RECENT_WORKSPACES_KEY = "nocturnomath.recent-workspaces";
const RECENT_WORKSPACE_LIMIT = 6;

function recentWorkspaces() {
  try {
    const stored = JSON.parse(localStorage.getItem(RECENT_WORKSPACES_KEY) || "[]");
    return Array.isArray(stored) ? stored.filter((path) => typeof path === "string") : [];
  } catch {
    return [];
  }
}

function recordRecentWorkspace(path) {
  if (!path) return;
  const projects = [path, ...recentWorkspaces().filter((item) => item !== path)]
    .slice(0, RECENT_WORKSPACE_LIMIT);
  localStorage.setItem(RECENT_WORKSPACES_KEY, JSON.stringify(projects));
}

function removeRecentWorkspace(path) {
  localStorage.setItem(
    RECENT_WORKSPACES_KEY,
    JSON.stringify(recentWorkspaces().filter((item) => item !== path))
  );
}

function renderRecentWorkspaces() {
  const projects = recentWorkspaces();
  elements.recentWorkspaces.innerHTML = "";
  if (!projects.length) return;

  const heading = document.createElement("span");
  heading.className = "recent-workspaces-label";
  heading.textContent = "Recent projects";
  elements.recentWorkspaces.appendChild(heading);

  const list = document.createElement("div");
  list.className = "recent-workspaces-list";
  projects.forEach((path) => {
    const project = document.createElement("button");
    project.type = "button";
    project.className = "recent-workspace";
    project.textContent = path;
    project.title = path;
    project.addEventListener("click", () => openRecentWorkspace(path));
    list.appendChild(project);
  });
  elements.recentWorkspaces.appendChild(list);
}

export async function openWorkspace(path, python = null) {
  const res = await fetch("/api/workspace", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, python }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Unable to prepare research environment");
  applyWorkspace(data.workspace);
  return data.workspace;
}

let loadingRecentWorkspace = false;
async function openRecentWorkspace(path) {
  if (loadingRecentWorkspace) return;
  loadingRecentWorkspace = true;
  const buttons = [...elements.recentWorkspaces.querySelectorAll("button"), document.getElementById("btn-open-workspace")];
  const loading = document.getElementById("recent-workspace-loading");
  const errorLabel = document.getElementById("recent-workspace-error");
  errorLabel.textContent = "";
  loading.classList.remove("hidden");
  buttons.forEach(button => button.disabled = true);
  try {
    await openWorkspace(path);
  } catch (error) {
    errorLabel.textContent = error.message;
  } finally {
    loadingRecentWorkspace = false;
    loading.classList.add("hidden");
    buttons.forEach(button => button.disabled = false);
  }
}

export function setCarryContext(enabled) {
  state.carryChatContext = Boolean(enabled);
  elements.contextToggle.checked = state.carryChatContext;
}

export function selectViewerTab(tab) {
  const resetDocuments = tab === "documents" && state.activeTab === "documents";
  setViewerTab(tab);
  if (tab === "plots") return loadPlots();
  if (tab === "documents") {
    if (resetDocuments) showDocumentsView("list");
    return fetchDocuments().then(refreshOpenDocument);
  }
  const path = tab === "evidence" ? state.workspace?.evidence_path : state.workspace?.thoughts_path;
  if (path) loadDocument(path, "", false, tab);
}

function setViewerTab(tab) {
  state.activeTab = tab;
  document.querySelectorAll("[data-tab]").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tab);
  });
  elements.viewDoc.classList.toggle("hidden", tab === "plots" || tab === "documents");
  elements.viewPlots.classList.toggle("hidden", tab !== "plots");
  elements.viewDocuments.classList.toggle("hidden", tab !== "documents");
}

// Which record tab a rendered file belongs to. Linked artifacts (probe code,
// output) stay under whichever record tab the reader was already on.
function tabForPath(filePath) {
  if (filePath === state.workspace?.evidence_path) return "evidence";
  if (filePath === state.workspace?.thoughts_path) return "thoughts";
  return ["evidence", "thoughts"].includes(state.activeTab) ? state.activeTab : "evidence";
}

let refreshing = false;
export async function refreshDocuments() {
  if (!state.workspace?.is_open || refreshing || state.exiting) return;
  refreshing = true;
  try {
    await fetchDocuments();
    if (state.activeTab === "plots") await loadPlots();
    else if (state.activeTab === "documents") await refreshOpenDocument();
    else await loadDocument(state.activeDoc, "", true);
  } finally { refreshing = false; }
}

export async function loadDocument(filePath, anchor = "", quiet = false, tab = null) {
  if (!state.workspace?.is_open) return;
  if (!filePath) filePath = state.workspace.evidence_path;
  state.activeDoc = filePath;
  if (!quiet) setViewerTab(tab || tabForPath(filePath));
  elements.docPathLabel.textContent = filePath;

  try {
    const res = await fetch(`/api/file?path=${encodeURIComponent(filePath)}`);
    if (state.activeDoc !== filePath) return;
    if (!res.ok) {
      elements.markdownContainer.innerHTML = `<p class="empty-state">File not found or empty: ${filePath}</p>`;
      state.documentContent = null;
      return;
    }
    const data = await res.json();
    const content = data.content || "";
    if (state.activeDoc !== filePath) return;
    if (quiet && window.getSelection()?.isCollapsed === false) return;
    if (state.documentPath === filePath && state.documentContent === content && !anchor) return;
    state.documentPath = filePath;
    state.documentContent = content;

    // Render markdown
    if (filePath.endsWith(".md")) {
      elements.markdownContainer.innerHTML = renderMarkdown(content, { breaks: false });
    } else {
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      code.textContent = content;
      pre.appendChild(code);
      elements.markdownContainer.replaceChildren(pre);
    }
    rewriteEmbeddedImageUrls(elements.markdownContainer, filePath);
    rewriteDocumentLinks(elements.markdownContainer, filePath);
    highlightBlocks(elements.markdownContainer);
    if (anchor) {
      const target = [...elements.markdownContainer.querySelectorAll("[id]")]
        .find((element) => element.id === anchor);
      target?.scrollIntoView({ block: "start" });
    }

    // Metadata
    const modifiedDate = new Date(data.modified * 1000).toLocaleTimeString();
    elements.docMetaLabel.textContent = `Updated: ${modifiedDate} (${data.size} B)`;
  } catch (e) {
    console.error("Error loading document:", e);
    elements.markdownContainer.innerHTML = `<p class="empty-state">Failed to load document: ${e.message}</p>`;
  }
}

export function rewriteDocumentLinks(container, documentPath) {
  container.querySelectorAll("a[href]").forEach((link) => {
    const source = link.getAttribute("href")?.trim();
    if (!source || source.startsWith("/") || /^[a-z][a-z0-9+.-]*:/i.test(source)) return;
    const target = new URL(source, `https://workspace.invalid/${documentPath}`);
    const path = decodeURIComponent(target.pathname.slice(1));
    const anchor = decodeURIComponent(target.hash.slice(1));
    link.href = `/api/asset?path=${encodeURIComponent(path)}`;
    link.addEventListener("click", (event) => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      if (/\.png$/i.test(path)) {
        openLightbox(link.href, path, state.plots || []);
      } else {
        loadDocument(path, anchor);
      }
    });
  });
}

export function rewriteEmbeddedImageUrls(container, documentPath) {
  const documentDir = documentPath.includes("/")
    ? documentPath.slice(0, documentPath.lastIndexOf("/"))
    : "";

  container.querySelectorAll("img[src]").forEach((image) => {
    const source = image.getAttribute("src")?.trim();
    if (!source || source.startsWith("/") || source.startsWith("//") ||
        source.startsWith("#") || /^[a-z][a-z0-9+.-]*:/i.test(source)) {
      return;
    }

    const assetPath = documentDir ? `${documentDir}/${source}` : source;
    image.src = `/api/asset?path=${encodeURIComponent(assetPath)}`;
  });
}

export async function loadPlots() {
  if (!state.workspace?.is_open) return;
  try {
    const res = await fetch("/api/plots");
    if (!res.ok) return;
    const data = await res.json();
    const plots = data.plots || [];
    const current = data.current_session || null;
    const changed = current !== state.currentSession
      || JSON.stringify(state.plots) !== JSON.stringify(plots);
    state.plots = plots;
    state.currentSession = current;

    if (changed) applyPlotsFilter();
  } catch (e) {
    console.error("Error loading plots:", e);
  }
}

// Session filter for the figures gallery. "all" shows everything, "current"
// follows whichever session is active, "selected" shows the checked sessions.
function plotSessions() {
  const counts = new Map();
  state.plots.forEach((p) => counts.set(p.session, (counts.get(p.session) || 0) + 1));
  return [...counts.entries()]
    .sort(([a], [b]) => b.localeCompare(a, undefined, { numeric: true }))
    .map(([id, count]) => ({ id, count }));
}

function visiblePlots() {
  const { mode, sessions } = state.plotsFilter;
  if (mode === "current") return state.plots.filter((p) => p.session === state.currentSession);
  if (mode === "selected") return state.plots.filter((p) => sessions.includes(p.session));
  return state.plots;
}

export function setPlotsFilter(mode, sessions = state.plotsFilter.sessions) {
  const known = new Set(plotSessions().map((s) => s.id));
  state.plotsFilter = { mode, sessions: sessions.filter((id) => known.has(id)) };
  applyPlotsFilter();
}

export function applyPlotsFilter() {
  const sessions = plotSessions();
  const known = new Set(sessions.map((s) => s.id));
  const filter = state.plotsFilter;
  filter.sessions = filter.sessions.filter((id) => known.has(id));

  elements.plotsFilterCurrent.textContent = state.currentSession ? `· ${state.currentSession}` : "";
  elements.plotsFilterMenu.querySelectorAll("input[name=plots-filter-mode]").forEach((input) => {
    input.checked = input.value === filter.mode;
  });

  elements.plotsFilterSessions.replaceChildren(...sessions.map((s) => {
    const label = document.createElement("label");
    label.className = "plots-filter-option plots-filter-session";
    const box = document.createElement("input");
    box.type = "checkbox";
    box.value = s.id;
    box.checked = filter.sessions.includes(s.id);
    const name = document.createElement("span");
    name.textContent = s.id + (s.id === state.currentSession ? " (current)" : "");
    const count = document.createElement("span");
    count.className = "plots-filter-note";
    count.textContent = s.count;
    label.append(box, name, count);
    return label;
  }));
  if (!sessions.length) {
    const empty = document.createElement("div");
    empty.className = "plots-filter-empty";
    empty.textContent = "No sessions with figures";
    elements.plotsFilterSessions.replaceChildren(empty);
  }

  const shown = visiblePlots();
  elements.plotsFilterLabel.textContent = filter.mode === "current"
    ? `Current session${state.currentSession ? ` · ${state.currentSession}` : ""}`
    : filter.mode === "selected"
      ? (filter.sessions.length === 1 ? filter.sessions[0] : `${filter.sessions.length} sessions`)
      : "All sessions";
  elements.plotsCount.textContent = shown.length;
  renderPlotsGallery(shown, filter.mode !== "all" && state.plots.length > 0);
}

export function togglePlotsFilterMenu(open = elements.plotsFilterMenu.classList.contains("hidden")) {
  elements.plotsFilterMenu.classList.toggle("hidden", !open);
  elements.plotsFilterBtn.setAttribute("aria-expanded", String(open));
}

export function renderPlotsGallery(plots, filtered = false) {
  elements.plotsGrid.innerHTML = "";
  if (!plots || !plots.length) {
    elements.plotsGrid.innerHTML = filtered
      ? `<div class="empty-state">No figures match the selected sessions.</div>`
      : `<div class="empty-state">No figures yet. Ask the agent to probe the system and produce a plot.</div>`;
    return;
  }

  plots.forEach((p) => {
    const card = document.createElement("div");
    card.className = "plot-card";

    const imgWrap = document.createElement("div");
    imgWrap.className = "plot-card-img-wrap";
    imgWrap.onclick = () => openLightbox(p.url, p.filename, plots);

    const img = document.createElement("img");
    img.src = p.url;
    img.alt = p.filename;
    img.loading = "lazy";
    imgWrap.appendChild(img);

    const footer = document.createElement("div");
    footer.className = "plot-card-footer";

    const name = document.createElement("span");
    name.className = "plot-card-name";
    name.textContent = p.filename;

    const time = document.createElement("span");
    time.textContent = new Date(p.modified * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    footer.appendChild(name);
    footer.appendChild(time);

    card.appendChild(imgWrap);
    card.appendChild(footer);
    elements.plotsGrid.appendChild(card);
  });
}

function showFolderPickerError(message = "") {
  elements.folderPickerError.textContent = message;
  elements.folderPickerError.classList.toggle("hidden", !message);
}

export async function browseFolders(path) {
  if (!path) return;
  showFolderPickerError();
  elements.folderBrowserPath.textContent = "Loading folders…";
  elements.folderList.innerHTML = "";
  try {
    const res = await fetch(`/api/folders?path=${encodeURIComponent(path)}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);

    state.folderParent = data.parent;
    elements.newFolderInput.value = data.path;
    elements.folderBrowserPath.textContent = data.path;
    elements.btnFolderUp.disabled = !data.parent;

    if (!data.folders?.length) {
      elements.folderList.innerHTML = '<div class="folder-empty">No subfolders in this directory.</div>';
      return;
    }
    data.folders.forEach((folder) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "folder-entry";
      button.textContent = folder.name;
      button.addEventListener("click", () => browseFolders(folder.path));
      elements.folderList.appendChild(button);
    });
  } catch (error) {
    state.folderParent = null;
    elements.btnFolderUp.disabled = true;
    elements.folderBrowserPath.textContent = "Folder unavailable";
    showFolderPickerError(error.message);
  }
}

export function openFolderPicker(mode = "open") {
  state.folderPickerMode = mode;
  const opening = mode === "open";
  elements.folderModalTitle.textContent = opening
    ? "Open a workspace folder"
    : "Change workspace folder";
  elements.folderModalHelp.innerHTML = opening
    ? "Browse to an existing folder. Once opened, the agent will create and manage its files under <code>.nocturnomath/</code>."
    : "Choose another existing folder. The agent will start a fresh kernel and session using that workspace’s research environment.";
  elements.modalSubmit.textContent = opening ? "Open folder" : "Switch workspace";
  elements.folderModal.classList.remove("hidden");
  browseFolders(state.workspace?.path || state.navigatorRoot || ".");
}

export function closeFolderPicker() {
  elements.folderModal.classList.add("hidden");
  showFolderPickerError();
}
