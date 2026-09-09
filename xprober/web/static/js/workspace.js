import { elements, state } from "./state.js?v=20260906-1";
import { highlightBlocks, renderMarkdown } from "./markdown.js?v=20260906-1";
import { openLightbox } from "./ui.js?v=20260906-1";
import { updateAgentStatus, updateKernelStatus } from "./status.js?v=20260906-1";

// ============================================================================
// Workspace & Files Management
// ============================================================================
export function applyWorkspace(ws) {
  if (!ws || !ws.is_open) {
    showLanding(ws);
    return;
  }
  if (state.workspace?.path !== ws.path) {
    state.documentContent = null;
    state.documentPath = null;
    fileListSignature = null;
  }
  state.workspace = ws;
  recordRecentWorkspace(ws.path);
  elements.landingScreen.classList.add("hidden");
  elements.workspaceApp.classList.remove("hidden");

  elements.workspacePath.textContent = workspaceName(ws.path);
  elements.workspacePill.title = ws.path;
  const models = ws.models || [];
  elements.modelSelect.replaceChildren(...models.map((model) => new Option(ws.model_labels?.[model] || model, model)));
  if (!models.length) elements.modelSelect.add(new Option("No model available", ""));
  elements.modelSelect.disabled = !models.length;
  elements.modelSelect.value = models.includes(ws.model) ? ws.model : (models[0] || "");

  updateKernelStatus(ws.kernel_alive, ws.kernel_busy);
  updateAgentStatus(ws.is_busy ? "thinking" : "idle");
  setCarryContext(ws.carry_chat_context);

  const notesPath = ws.evidence_path || "xprober/notes/evidence.md";
  if (!state.activeDoc || !ws.markdown_files?.some((f) => f.path === state.activeDoc)) {
    state.activeDoc = notesPath;
  }

  // Populate file selector
  updateFileSelector(ws.markdown_files || []);

  // Refresh active doc
  loadDocument(state.activeDoc);

  // Refresh plots list
  loadPlots();
}

function workspaceName(path) {
  const parts = String(path || "").split(/[\\/]+/).filter(Boolean);
  return parts.at(-1) || path;
}

export function showLanding(ws) {
  state.workspace = null;
  state.navigatorRoot = ws?.navigator_root || state.navigatorRoot || ".";
  elements.workspaceApp.classList.add("hidden");
  elements.landingScreen.classList.remove("hidden");
  renderRecentWorkspaces();
}

const RECENT_WORKSPACES_KEY = "xprober.recent-workspaces";
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

async function openRecentWorkspace(path) {
  try {
    const res = await fetch("/api/workspace", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Unable to open project");
    applyWorkspace(data.workspace);
  } catch (error) {
    console.error("Unable to open recent workspace:", error);
    removeRecentWorkspace(path);
    renderRecentWorkspaces();
  }
}

export function setCarryContext(enabled) {
  state.carryChatContext = Boolean(enabled);
  elements.contextToggle.checked = state.carryChatContext;
}

let fileListSignature = null;
export function updateFileSelector(files) {
  const signature = JSON.stringify(files.map((file) => [file.path, file.is_notes]));
  if (signature === fileListSignature) return;
  fileListSignature = signature;
  const currentVal = elements.fileSelect.value;
  elements.fileSelect.innerHTML = "";

  if (!files.length) {
    const opt = document.createElement("option");
    opt.value = state.workspace?.evidence_path || "xprober/notes/evidence.md";
    opt.textContent = opt.value;
    elements.fileSelect.appendChild(opt);
    return;
  }

  files.forEach((f) => {
    const opt = document.createElement("option");
    opt.value = f.path;
    opt.textContent = f.path + (f.is_notes ? " (notes)" : "");
    elements.fileSelect.appendChild(opt);
  });

  // Preserve selection if possible
  const exists = files.some((f) => f.path === currentVal);
  if (exists) {
    elements.fileSelect.value = currentVal;
  } else {
    elements.fileSelect.value = files[0].path;
  }
}

export function selectViewerTab(tab) {
  setViewerTab(tab);
  if (tab === "plots") return loadPlots();
  const path = tab === "evidence" ? state.workspace?.evidence_path
    : tab === "thoughts" ? state.workspace?.thoughts_path : elements.fileSelect.value;
  if (path) loadDocument(path, "", false, tab);
}

function setViewerTab(tab) {
  state.activeTab = tab;
  document.querySelectorAll("[data-tab]").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tab);
  });
  elements.viewDoc.classList.toggle("hidden", tab === "plots");
  elements.viewPlots.classList.toggle("hidden", tab !== "plots");
  elements.fileSelect.parentElement.classList.toggle("hidden", tab !== "doc");
}

let refreshing = false;
export async function refreshDocuments() {
  if (!state.workspace?.is_open || refreshing || state.exiting) return;
  refreshing = true;
  try {
    const response = await fetch("/api/files");
    if (response.ok) updateFileSelector((await response.json()).files || []);
    if (state.activeTab !== "plots") await loadDocument(state.activeDoc, "", true);
    else await loadPlots();
  } finally { refreshing = false; }
}

export async function loadDocument(filePath, anchor = "", quiet = false, tab = null) {
  if (!state.workspace?.is_open) return;
  if (!filePath) filePath = state.workspace.evidence_path;
  state.activeDoc = filePath;
  if (!quiet) setViewerTab(tab || (filePath === state.workspace.evidence_path ? "evidence"
    : filePath === state.workspace.thoughts_path ? "thoughts" : "doc"));
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
      elements.markdownContainer.innerHTML = renderMarkdown(content);
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
    if (![...elements.fileSelect.options].some((option) => option.value === filePath)) {
      elements.fileSelect.add(new Option(filePath, filePath));
    }
    elements.fileSelect.value = filePath;
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

function rewriteDocumentLinks(container, documentPath) {
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

function rewriteEmbeddedImageUrls(container, documentPath) {
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
    const changed = JSON.stringify(state.plots) !== JSON.stringify(plots);
    state.plots = plots;
    elements.plotsCount.textContent = plots.length;

    if (changed) renderPlotsGallery(plots);
  } catch (e) {
    console.error("Error loading plots:", e);
  }
}

export function renderPlotsGallery(plots) {
  elements.plotsGrid.innerHTML = "";
  if (!plots || !plots.length) {
    elements.plotsGrid.innerHTML = `<div class="empty-state">No figures yet. Ask the agent to probe the system and produce a plot.</div>`;
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
    ? "Browse to an existing folder. Once opened, the agent will create and manage its files under <code>xprober/</code>."
    : "Choose another existing folder. The agent will keep its kernel and begin a new workspace session there.";
  elements.modalSubmit.textContent = opening ? "Open folder" : "Switch workspace";
  elements.folderModal.classList.remove("hidden");
  browseFolders(state.workspace?.path || state.navigatorRoot || ".");
}

export function closeFolderPicker() {
  elements.folderModal.classList.add("hidden");
  showFolderPickerError();
}
