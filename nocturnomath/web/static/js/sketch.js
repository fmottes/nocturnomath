/**
 * Nocturnomath — hand-drafted chrome.
 *
 * rough.js draws the structural linework: pane dividers, callout boxes and
 * container frames. Roughness stays low so it reads as a steady hand rather
 * than a scribble, and every element keeps a stable seed so redraws never
 * "boil". Nothing here touches figures, tables or numbers — data is drawn
 * precisely, by the code that produced it.
 */
(function () {
  "use strict";

  var rough = window.rough;
  if (!rough) return; // CSS keeps precise 1px borders as the fallback

  var SVG_NS = "http://www.w3.org/2000/svg";
  var colorCache = {};

  function cssColor(name, fallback) {
    if (Object.prototype.hasOwnProperty.call(colorCache, name)) {
      return colorCache[name] || fallback;
    }
    var value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    colorCache[name] = value;
    return value || fallback;
  }

  // Chrome only. Order is irrelevant; each element is drawn independently.
  var CHROME = [
    { selector: ".app-header", kind: "rule", color: "--blue", weight: 1.1 },
    { selector: ".chat-header", kind: "rule" },
    { selector: ".viewer-header", kind: "rule" },
    { selector: ".chat-input-container", kind: "rule-top" },
    { selector: ".doc-header-info", kind: "rule" },
    { selector: ".plots-gallery-header", kind: "rule" },
    { selector: ".gutter", kind: "vline", color: "--blue", weight: 1 },
    { selector: ".message-card.system-welcome", kind: "box" },
    { selector: ".probe-card", kind: "box", color: "--blue" },
    { selector: ".chat-bubble.user .bubble-body", kind: "box" },
    { selector: ".note-notification", kind: "box", color: "--blue" },
    { selector: ".modal-dialog", kind: "box", color: "--blue" },
  ];

  var drawn = new WeakMap(); // element -> spec
  var pending = new Set();
  var frame = null;
  var fallback = null;

  function seedFor(el) {
    if (el.__roughSeed === undefined) {
      el.__roughSeed = (Math.random() * 2147483647) | 0;
    }
    return el.__roughSeed;
  }

  function layerFor(el) {
    var svg = el.__roughLayer;
    if (!svg || svg.parentNode !== el) {
      svg = document.createElementNS(SVG_NS, "svg");
      svg.setAttribute("class", "rough-layer");
      svg.setAttribute("aria-hidden", "true");
      svg.setAttribute("preserveAspectRatio", "none");
      el.insertBefore(svg, el.firstChild);
      el.__roughLayer = svg;
    }
    return svg;
  }

  function drawNow(el) {
    var spec = drawn.get(el);
    if (!spec || !el.isConnected) return;

    var w = el.clientWidth;
    var h = el.clientHeight;
    if (!w || !h) return;

    var svg = layerFor(el);
    svg.setAttribute("width", w);
    svg.setAttribute("height", h);
    svg.setAttribute("viewBox", "0 0 " + w + " " + h);
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    var rc = rough.svg(svg);
    var opts = {
      roughness: spec.kind === "box" ? 0.8 : 0.7,
      bowing: spec.kind === "box" ? 0.6 : 0.35,
      stroke: spec.color
        ? cssColor(spec.color, "#1e3a8a")
        : cssColor("--rule-strong", "#a8b0c2"),
      strokeWidth: spec.weight || 1,
      seed: seedFor(el),
      disableMultiStroke: true, // one steady pass, not a sketchy double line
    };

    var node = null;
    switch (spec.kind) {
      case "box":
        node = rc.rectangle(1, 1, w - 2, h - 2, opts);
        break;
      case "rule":
        node = rc.line(0, h - 1, w, h - 1, opts);
        break;
      case "rule-top":
        node = rc.line(0, 1, w, 1, opts);
        break;
      case "vline":
        node = rc.line(5.5, 4, 5.5, h - 4, opts);
        break;
    }
    if (node) svg.appendChild(node);
    el.setAttribute("data-rough", spec.kind);
  }

  function flush() {
    if (frame !== null) {
      cancelAnimationFrame(frame);
      frame = null;
    }
    if (fallback !== null) {
      clearTimeout(fallback);
      fallback = null;
    }
    pending.forEach(drawNow);
    pending.clear();
  }

  // Frame callbacks are the right moment to draw, but they are throttled in
  // background tabs, so a short timer guarantees the redraw still happens.
  function schedule(el) {
    pending.add(el);
    if (frame === null) frame = requestAnimationFrame(flush);
    if (fallback === null) fallback = setTimeout(flush, 80);
  }

  var resizeObserver =
    "ResizeObserver" in window
      ? new ResizeObserver(function (entries) {
          for (var i = 0; i < entries.length; i++) schedule(entries[i].target);
        })
      : null;

  function register(el, spec) {
    if (drawn.has(el)) return;
    drawn.set(el, spec);
    schedule(el);
    if (resizeObserver) resizeObserver.observe(el);
  }

  function scan(root) {
    for (var i = 0; i < CHROME.length; i++) {
      var spec = CHROME[i];
      if (root.matches && root.matches(spec.selector)) register(root, spec);
      if (!root.querySelectorAll) continue;
      var found = root.querySelectorAll(spec.selector);
      for (var j = 0; j < found.length; j++) register(found[j], spec);
    }
  }

  // --------------------------------------------------------------------------
  // Boot, and keep up with chat cards appended as a run proceeds.
  // --------------------------------------------------------------------------
  function start() {
    scan(document.body);

    var log = document.getElementById("chat-messages");
    if (log) {
      new MutationObserver(function (mutations) {
        for (var i = 0; i < mutations.length; i++) {
          var added = mutations[i].addedNodes;
          for (var j = 0; j < added.length; j++) {
            var node = added[j];
            if (node.nodeType === 1) scan(node);
          }
        }
      }).observe(log, { childList: true });
    }

    window.addEventListener("nocturnomath:themechange", function () {
      colorCache = {};
      redrawAll();
    });

    // Modals are hidden at load, so they have no measurable box until opened.
    var modals = document.querySelectorAll(".modal-backdrop");
    for (var i = 0; i < modals.length; i++) {
      (function (modal) {
        new MutationObserver(function () {
          if (!modal.classList.contains("hidden")) {
            var dialog = modal.querySelector(".modal-dialog");
            if (dialog) schedule(dialog);
          }
        }).observe(modal, { attributes: true, attributeFilter: ["class"] });
      })(modals[i]);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }

  // Web fonts change measured heights; redraw once they have settled.
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(redrawAll);
  }

  function redrawAll() {
    var all = document.querySelectorAll("[data-rough]");
    for (var i = 0; i < all.length; i++) schedule(all[i]);
  }
})();
