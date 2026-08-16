(() => {
  const root = document.documentElement;
  const themeToggle = document.querySelector("#theme-toggle");
  const savedTheme = localStorage.getItem("gemm-optimization-blog-theme");
  const preferredTheme = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";

  root.dataset.theme = savedTheme || preferredTheme;

  themeToggle?.addEventListener("click", () => {
    const nextTheme = root.dataset.theme === "dark" ? "light" : "dark";
    root.dataset.theme = nextTheme;
    localStorage.setItem("gemm-optimization-blog-theme", nextTheme);
  });

  const progressBar = document.querySelector("#progress-bar");
  const updateProgress = () => {
    const scrollable = document.documentElement.scrollHeight - window.innerHeight;
    const progress = scrollable > 0 ? Math.min(window.scrollY / scrollable, 1) : 0;
    progressBar.style.width = String(progress * 100) + "%";
  };

  updateProgress();
  window.addEventListener("scroll", updateProgress, { passive: true });
  window.addEventListener("resize", updateProgress);

  const tocLinks = Array.from(document.querySelectorAll("#toc a"));
  const linksById = new Map(tocLinks.map((link) => [link.hash.slice(1), link]));
  const sections = Array.from(linksById.keys())
    .map((id) => document.getElementById(id))
    .filter(Boolean);

  const activateLink = (id) => {
    tocLinks.forEach((link) => link.classList.toggle("active", link.hash === "#" + id));
  };

  if ("IntersectionObserver" in window) {
    const visible = new Map();
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) visible.set(entry.target.id, entry.target.offsetTop);
        else visible.delete(entry.target.id);
      });
      const current = Array.from(visible.entries()).sort((a, b) => a[1] - b[1])[0];
      if (current) activateLink(current[0]);
    }, { rootMargin: "-18% 0px -70% 0px" });

    sections.forEach((section) => observer.observe(section));
  }
})();
