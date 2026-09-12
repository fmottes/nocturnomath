export function renderMarkdown(markdown) {
  if (window.marked && typeof window.marked.parse === "function") {
    try {
      return window.marked.parse(markdown, { gfm: true, breaks: true });
    } catch (error) {
      console.warn("marked.js error, falling back:", error);
    }
  }

  let html = markdown
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
  return html.replace(/\n\n/g, "<p></p>");
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
