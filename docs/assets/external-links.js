// Open external links in a new tab.
// Required for HF static Spaces: the app runs in a sandboxed iframe that
// blocks same-frame navigation to other origins.
document.querySelectorAll("a[href^='http']").forEach((link) => {
  const url = new URL(link.href, window.location.href);
  if (url.origin !== window.location.origin) {
    link.target = "_blank";
    link.rel = "noopener";
  }
});
