const ALLOWED_TAGS = new Set([
  "a", "blockquote", "br", "code", "del", "em", "h1", "h2", "h3", "h4",
  "h5", "h6", "hr", "img", "li", "ol", "p", "pre", "strong", "table",
  "tbody", "td", "th", "thead", "tr", "ul",
]);

function safeUrl(value, image = false) {
  const source = value.trim();
  if (!source || source.startsWith("#") || source.startsWith("/") || source.startsWith("./") || source.startsWith("../")) {
    return source;
  }
  try {
    const protocol = new URL(source).protocol;
    return image
      ? (protocol === "https:" ? source : null)
      : (["http:", "https:", "mailto:"].includes(protocol) ? source : null);
  } catch (error) {
    return null;
  }
}

function sanitizeHtml(html) {
  const template = document.createElement("template");
  template.innerHTML = html;

  for (const element of [...template.content.querySelectorAll("*")]) {
    const tag = element.localName;
    if (!ALLOWED_TAGS.has(tag)) {
      element.replaceWith(document.createTextNode(element.textContent || ""));
      continue;
    }

    const attributes = Object.fromEntries(
      [...element.attributes].map((attribute) => [attribute.name, attribute.value])
    );
    for (const attribute of [...element.attributes]) element.removeAttribute(attribute.name);

    if (tag === "a") {
      const original = attributes.href ?? null;
      const href = original === null ? null : safeUrl(original);
      if (href !== null) element.setAttribute("href", href);
    } else if (tag === "img") {
      const source = attributes.src ?? null;
      const src = source === null ? null : safeUrl(source, true);
      if (src !== null) element.setAttribute("src", src);
      const alt = attributes.alt ?? null;
      if (alt !== null) element.setAttribute("alt", alt);
    } else if (tag === "code") {
      const className = attributes.class ?? null;
      if (/^language-[A-Za-z0-9_-]+$/.test(className || "")) {
        element.setAttribute("class", className);
      }
    }

    const id = attributes.id ?? null;
    if (/^[ET]\d+$/.test(id || "")) element.setAttribute("id", id);
  }
  return template.innerHTML;
}

function escapeHtml(text) {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function protectMath(markdown) {
  const formulas = [];
  const protectedText = markdown.replace(
    /\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|(?<!\\)\$(?!\s)(?:\\.|[^$\n])+?(?<!\s)\$/g,
    (formula) => {
      const token = `NocturnomathMathToken${formulas.length}End`;
      formulas.push(formula);
      return token;
    },
  );
  return {
    text: protectedText,
    restore: (html) => html.replace(/NocturnomathMathToken(\d+)End/g, (token, index) =>
      formulas[Number(index)] === undefined ? token : escapeHtml(formulas[Number(index)])),
  };
}

export function renderMarkdown(markdown, { breaks = true } = {}) {
  const math = protectMath(markdown);
  let rendered = null;
  if (window.marked && typeof window.marked.parse === "function") {
    try {
      rendered = window.marked.parse(math.text, { gfm: true, breaks });
    } catch (error) {
      console.warn("marked.js error, falling back:", error);
    }
  }
  if (rendered !== null) return math.restore(sanitizeHtml(rendered));

  let html = math.text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  html = html.replace(/```([a-z]*)\n([\s\S]*?)```/g, (match, lang, code) => {
    return `<pre><code class="language-${lang}">${code}</code></pre>`;
  });
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  // Keep scientific record anchors, strikes, and citations usable without the CDN renderer.
  html = html.replace(/&lt;a id="([ET]\d+)"&gt;&lt;\/a&gt;/g, '<a id="$1"></a>');
  html = html.replace(/&lt;(\/?del)&gt;/g, "<$1>");
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, label, target) => {
    if (/^(?:javascript|data|vbscript):/i.test(target.trim())) return label;
    return `<a href="${target.replace(/"/g, "&quot;")}">${label}</a>`;
  });
  html = html.replace(/^### (.*$)/gim, "<h3>$1</h3>");
  html = html.replace(/^## (.*$)/gim, "<h2>$1</h2>");
  html = html.replace(/^# (.*$)/gim, "<h1>$1</h1>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  html = html.replace(/^\- (.*$)/gim, "<li>$1</li>");
  html = html.replace(/(<li>.*<\/li>)/s, "<ul>$1</ul>");
  html = html.replace(/\n\n/g, "<p></p>");
  return math.restore(sanitizeHtml(html));
}

export function highlightBlocks(container) {
  if (window.hljs && typeof window.hljs.highlightElement === "function") {
    container.querySelectorAll("pre code").forEach((element) => {
      try {
        window.hljs.highlightElement(element);
      } catch (error) {
        // Highlighting is optional.
      }
    });
  }
}

export function renderMath(container) {
  if (typeof window.renderMathInElement !== "function") return;
  try {
    window.renderMathInElement(container, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
        { left: "$", right: "$", display: false },
      ],
      throwOnError: false,
    });
  } catch (error) {
    console.warn("KaTeX error, leaving formula source visible:", error);
  }
}
