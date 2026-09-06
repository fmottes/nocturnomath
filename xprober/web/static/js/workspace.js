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
  state.workspace = ws;
  recordRecentWorkspace(ws.path);
  elements.landingScreen.classList.add("hidden");
  elements.workspaceApp.classList.remove("hidden");

  elements.workspacePath.textContent = workspaceName(ws.path);
  elements.modelBadge.textContent = ws.model || "claude-opus-5";

  updateKernelStatus(ws.kernel_alive, ws.kernel_busy);
  updateAgentStatus(ws.is_busy ? "thinking" : "idle");
  setCarryContext(ws.carry_chat_context);

  const notesPath = ws.notes_path || "xprober/notes/notes.md";
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

export function updateFileSelector(files) {
  const currentVal = elements.fileSelect.value;
  elements.fileSelect.innerHTML = "";

  if (!files.length) {
    const opt = document.createElement("option");
    opt.value = state.workspace?.notes_path || "xprober/notes/notes.md";
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
    state.activeDoc = files[0].path;
  }
}

export async function loadDocument(filePath) {
  if (!state.workspace?.is_open) return;
  if (!filePath) filePath = state.workspace.notes_path;
  state.activeDoc = filePath;
  elements.docPathLabel.textContent = filePath;

  try {
    const res = await fetch(`/api/file?path=${encodeURIComponent(filePath)}`);
    if (!res.ok) {
      elements.markdownContainer.innerHTML = `<p class="empty-state">File not found or empty: ${filePath}</p>`;
      elements.rawMarkdownContainer.querySelector("code").textContent = "";
      return;
    }
    const data = await res.json();
    const content = data.content || "";

    // Render markdown
    elements.markdownContainer.innerHTML = renderMarkdown(content);
    highlightBlocks(elements.markdownContainer);

    // Raw content
    elements.rawMarkdownContainer.querySelector("code").textContent = content;

    // Metadata
    const modifiedDate = new Date(data.modified * 1000).toLocaleTimeString();
    elements.docMetaLabel.textContent = `Updated: ${modifiedDate} (${data.size} B)`;
  } catch (e) {
    console.error("Error loading document:", e);
    elements.markdownContainer.innerHTML = `<p class="empty-state">Failed to load document: ${e.message}</p>`;
  }
}

export async function loadPlots() {
  if (!state.workspace?.is_open) return;
  try {
    const res = await fetch("/api/plots");
    if (!res.ok) return;
    const data = await res.json();
    const plots = data.plots || [];
    state.plots = plots;
    elements.plotsCount.textContent = plots.length;

    renderPlotsGallery(plots);
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
    imgWrap.onclick = () => openLightbox(p.url, p.filename);

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
