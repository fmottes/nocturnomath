import { elements, state } from "./state.js?v=20260913-1";

// ============================================================================
// Lightbox View
// ============================================================================
function normalizeLightboxItems(imgSrc, caption, items) {
  const normalized = (items || [])
    .map((item) => ({
      src: item.src || item.url,
      caption: item.caption || item.filename || "",
    }))
    .filter((item) => item.src);

  return normalized.length ? normalized : [{ src: imgSrc, caption: caption || "" }];
}

function renderLightboxItem() {
  const item = state.lightboxItems[state.lightboxIndex];
  if (!item) return;

  elements.lightboxImg.src = item.src;
  elements.lightboxImg.alt = item.caption || "Enlarged figure";
  const position = state.lightboxItems.length > 1
    ? ` (${state.lightboxIndex + 1}/${state.lightboxItems.length})`
    : "";
  elements.lightboxCaption.textContent = (item.caption || "Figure") + position;
}

export function openLightbox(imgSrc, caption, items) {
  state.lightboxItems = normalizeLightboxItems(imgSrc, caption, items);
  state.lightboxIndex = state.lightboxItems.findIndex((item) => item.src === imgSrc);
  if (state.lightboxIndex < 0) state.lightboxIndex = 0;
  renderLightboxItem();
  elements.lightboxModal.classList.remove("hidden");
}

export function closeLightbox() {
  elements.lightboxModal.classList.add("hidden");
  state.lightboxItems = [];
  state.lightboxIndex = -1;
}

export function isLightboxOpen() {
  return !elements.lightboxModal.classList.contains("hidden");
}

export function cycleLightbox(direction) {
  if (!isLightboxOpen() || state.lightboxItems.length < 2) return false;
  const count = state.lightboxItems.length;
  state.lightboxIndex = (state.lightboxIndex + direction + count) % count;
  renderLightboxItem();
  return true;
}
