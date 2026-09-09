/**
 * Xprober — hand-drafted chrome.
 *
 * rough.js draws the structural linework: pane dividers, callout boxes and
 * container frames. Roughness stays low so it reads as a steady hand rather
 * than a scribble, and every element keeps a stable seed so redraws never
 * "boil". Nothing here touches figures, tables or numbers — data is drawn
 * precisely, by the code that produced it.
 *
 * rough-notation is used for one thing only: marking a genuinely important
 * result (a fact the agent recorded) with the single highlighter accent.
 */
(function () {
  "use strict";

  var rough = window.rough;
  var RoughNotation = window.RoughNotation;
  if (!rough) return; // CSS keeps precise 1px borders as the fallback

  var SVG_NS = "http://www.w3.org/2000/svg";
  function cssColor(name, fallback) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
  }

  // --------------------------------------------------------------------------
  // Motion preference. Only decorative motion is affected: the highlighter
  // drawing itself in. Structural linework is static either way.
  // --------------------------------------------------------------------------
  var MOTION_KEY = "xprober.motion";

  function systemPrefersReduced() {
    return !!(
      window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches
    );
  }

  function storedMotion() {
    try {
      return window.localStorage.getItem(MOTION_KEY);
    } catch (e) {
      return null;
    }
  }

  // The system preference is the default. A stored override is only honoured
  // while a control exists to reverse it — see start().
  var motionOn = !systemPrefersReduced();

  function applyMotion(on) {
    motionOn = on;
    document.documentElement.setAttribute("data-motion", on ? "on" : "off");
    try {
      window.localStorage.setItem(MOTION_KEY, on ? "on" : "off");
    } catch (e) {
      /* preference simply does not persist */
    }
    for (var i = 0; i < waiting.length; i++) {
      if (waiting[i].__annotation) waiting[i].__annotation.animate = on;
    }
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
  // Annotation: the single highlighter, drawn in once, on recorded facts only.
  // --------------------------------------------------------------------------
  var annotated = new WeakSet();
  var waiting = [];
  var revealFrame = null;
  var revealTimer = null;

  function inView(el) {
    var r = el.getBoundingClientRect();
    if (!r.height) return false;
    return r.top < window.innerHeight - 16 && r.bottom > 0;
  }

  function revealVisible() {
    revealFrame = null;
    for (var i = waiting.length - 1; i >= 0; i--) {
      var target = waiting[i];
      if (!target.isConnected) {
        waiting.splice(i, 1);
        continue;
      }
      if (!inView(target)) continue;
      waiting.splice(i, 1);
      target.__annotation.show();
    }
    if (!waiting.length && revealTimer !== null) {
      clearInterval(revealTimer);
      revealTimer = null;
    }
  }

  function scheduleReveal() {
    if (revealFrame === null) revealFrame = requestAnimationFrame(revealVisible);
    // Scroll events are the usual trigger, but a result can also come into
    // view because the pane was resized or the log re-laid out. A cheap poll,
    // alive only while something is still waiting, covers those cases.
    if (revealTimer === null && waiting.length) {
      revealTimer = setInterval(revealVisible, 300);
    }
  }

  function markResult(notification) {
    if (!RoughNotation || annotated.has(notification)) return;
    annotated.add(notification);

    var spans = notification.querySelectorAll("span");
    var target = spans.length > 1 ? spans[spans.length - 1] : null;
    if (!target) return;

    var annotation = RoughNotation.annotate(target, {
      type: "highlight",
      color: cssColor("--highlighter", "#fde68a"),
      multiline: true,
      animate: motionOn,
      animationDuration: 480,
      iterations: 1,
    });
    target.__annotation = annotation;
    waiting.push(target);
    scheduleReveal();
  }

  function scanAnnotations(root) {
    if (root.matches && root.matches(".note-notification.fact")) markResult(root);
    if (!root.querySelectorAll) return;
    var found = root.querySelectorAll(".note-notification.fact");
    for (var i = 0; i < found.length; i++) markResult(found[i]);
  }

  // --------------------------------------------------------------------------
  // Boot, and keep up with the nodes app.js appends as a run proceeds.
  // --------------------------------------------------------------------------
  function start() {
    scan(document.body);
    scanAnnotations(document.body);

    new MutationObserver(function (mutations) {
      for (var i = 0; i < mutations.length; i++) {
        var added = mutations[i].addedNodes;
        for (var j = 0; j < added.length; j++) {
          var node = added[j];
          if (node.nodeType !== 1 || node.classList.contains("rough-annotation")) continue;
          scan(node);
          scanAnnotations(node);
        }
      }
    }).observe(document.body, { childList: true, subtree: true });

    var motionToggle = document.getElementById("motion-toggle");
    if (motionToggle) {
      if (storedMotion() !== null) motionOn = storedMotion() === "on";
      motionToggle.checked = motionOn;
      motionToggle.addEventListener("change", function () {
        applyMotion(motionToggle.checked);
      });
    }
    applyMotion(motionOn);

    var log = document.getElementById("chat-messages");
    if (log) log.addEventListener("scroll", scheduleReveal, { passive: true });
    window.addEventListener("scroll", scheduleReveal, { passive: true });
    window.addEventListener("resize", scheduleReveal, { passive: true });
    window.addEventListener("xprober:themechange", redrawAll);

    // Modals are hidden at load, so they have no measurable box until opened.
    ["folder-modal", "history-modal"].forEach(function (id) {
      var modal = document.getElementById(id);
      if (!modal) return;
      new MutationObserver(function () {
        if (!modal.classList.contains("hidden")) {
          var dialog = modal.querySelector(".modal-dialog");
          if (dialog) schedule(dialog);
        }
      }).observe(modal, { attributes: true, attributeFilter: ["class"] });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }

  // Web fonts change measured heights; redraw once they have settled.
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(function () {
      redrawAll();
      scheduleReveal();
    });
  }

  function redrawAll() {
    var all = document.querySelectorAll("[data-rough]");
    for (var i = 0; i < all.length; i++) schedule(all[i]);
  }
})();
