import { elements } from "./state.js?v=20260906-1";

// ============================================================================
// Lightbox View
// ============================================================================
export function openLightbox(imgSrc, caption) {
  elements.lightboxImg.src = imgSrc;
  elements.lightboxCaption.textContent = caption || "";
  elements.lightboxModal.classList.remove("hidden");
}

export function closeLightbox() {
  elements.lightboxModal.classList.add("hidden");
}
