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
