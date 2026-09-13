(() => {
  const key = "nocturnomath.theme";
  let theme = null;
  try {
    theme = window.localStorage.getItem(key);
  } catch (error) {
    // Use the system preference when storage is unavailable.
  }
  if (theme !== "light" && theme !== "dark") {
    theme = window.matchMedia?.("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  }
  document.documentElement.dataset.theme = theme;
})();
