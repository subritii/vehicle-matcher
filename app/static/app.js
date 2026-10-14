// Theme toggle, review hotkeys and invoice field highlighting. No build step.

function toggleTheme() {
  const dark = !document.documentElement.classList.contains("dark");
  document.documentElement.classList.toggle("dark", dark);
  localStorage.setItem("theme", dark ? "dark" : "light");
}

// Hotkeys: any element with data-hotkey="c" (or "1", "n", "arrowright"...) is clicked
// when that key is pressed outside a form field.
document.addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const tag = (e.target.tagName || "").toLowerCase();
  if (["input", "select", "textarea"].includes(tag) || e.target.isContentEditable) return;
  const key = e.key.toLowerCase();
  if (key === "?") {
    document.getElementById("hotkey-help")?.classList.toggle("hidden");
    return;
  }
  const el = document.querySelector(`[data-hotkey~="${CSS.escape(key)}"]`);
  if (el && !el.disabled) {
    e.preventDefault();
    el.classList.add("ring-4");
    el.click();
  }
});

// Hovering a field row highlights where it was read on the invoice, and the reverse.
function setActiveField(name, on) {
  document.querySelectorAll(`[data-field="${CSS.escape(name)}"]`).forEach((el) => el.classList.toggle("is-active", on));
}
document.addEventListener("mouseover", (e) => {
  const el = e.target.closest("[data-field]");
  if (el) setActiveField(el.dataset.field, true);
});
document.addEventListener("mouseout", (e) => {
  const el = e.target.closest("[data-field]");
  if (el) setActiveField(el.dataset.field, false);
});

// Photo viewer: clicking a thumbnail with data-full shows it in the main frame.
document.addEventListener("click", (e) => {
  const t = e.target.closest("[data-full]");
  if (!t) return;
  e.preventDefault();
  const frame = document.getElementById(t.dataset.target || "photo-main");
  if (!frame) return;
  frame.querySelector("img").src = t.dataset.full;
  const cap = frame.querySelector("[data-caption]");
  if (cap) cap.textContent = t.dataset.caption || "";
  document.querySelectorAll(`[data-target="${frame.id}"]`).forEach((x) => x.classList.toggle("is-selected", x === t));
});
