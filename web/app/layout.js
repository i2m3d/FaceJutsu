// The editor's shell: the canvas, the timeline and the settings column on one grid. The 8px gaps
// between them are grid cells that double as resize handles: the side gap sets the right column's
// width (settings in landscape, settings and timeline in vertical), the lower gap the timeline's
// height. Each layout keeps its own sizes; double-click a gap to reset it.

const KEY = "facejutsu.layout";
const DEF = { land: { sw: 340, th: 240 }, port: { sw: 420, th: 260 } };

export function initLayout(app) {
  // the settings column stays 300–600 px (narrower squeezes its controls, wider leaves them empty
  // margins); in vertical, where the timeline shares it, up to 900 px. It's never narrower than
  // its header needs (minWidth): the window's own limit gives way to that, not the header.
  const LIM = {
    land: { sw: [300, () => Math.min(600, app.clientWidth * 0.5)], th: [110, () => app.clientHeight * 0.7] },
    port: { sw: [300, () => Math.min(900, app.clientWidth * 0.6)], th: [110, () => app.clientHeight * 0.7] },
  };
  let state = { mode: "land", land: { ...DEF.land }, port: { ...DEF.port } };
  try {
    const s = JSON.parse(localStorage.getItem(KEY));
    if (s) state = { mode: s.mode === "port" ? "port" : "land", land: { ...DEF.land, ...s.land }, port: { ...DEF.port, ...s.port } };
  } catch { /* private mode */ }

  // The settings' header: the logo and the app's name, then its buttons, all at their own widths,
  // with room to spare between the name and the buttons
  const head = app.querySelector(".project");
  const SPARE = 24;
  const headWidth = () => {
    const css = getComputedStyle(head), gap = parseFloat(css.columnGap) || 0;
    const kids = [...head.children].filter((c) => c.offsetParent !== null);
    const w = kids.reduce((a, c) => a + c.scrollWidth, 0) + gap * (kids.length - 1);
    return Math.ceil(w + parseFloat(css.paddingLeft) + parseFloat(css.paddingRight) + 2 + SPARE);
  };
  let onChange = () => {};

  function apply() {
    const m = state.mode, size = state[m];
    app.classList.toggle("portrait", m === "port");
    const need = headWidth();
    for (const k in size) {
      const [lo, hi] = LIM[m][k];
      const min = k === "sw" ? Math.max(lo, need) : lo;
      size[k] = Math.round(Math.max(min, Math.min(hi(), size[k])));
      app.style.setProperty("--" + k, size[k] + "px");
    }
    try { localStorage.setItem(KEY, JSON.stringify(state)); } catch { /* private mode */ }
    onChange(m);
  }
  const setMode = (m) => { state.mode = m; apply(); };
  const toggle = () => setMode(state.mode === "land" ? "port" : "land");
  addEventListener("keydown", (e) => {
    if (e.key.toLowerCase() !== "l" || e.ctrlKey || e.metaKey || e.altKey || typing(e.target)) return;
    toggle();
  });

  for (const sp of app.querySelectorAll(".split")) {
    const k = sp.classList.contains("x") ? "sw" : "th";
    sp.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      sp.setPointerCapture(e.pointerId);
      sp.classList.add("active");
      document.body.classList.add("dragging", k === "sw" ? "x" : "y");
      const size = state[state.mode];
      const start = k === "sw" ? e.clientX : e.clientY, base = size[k];
      const move = (ev) => { size[k] = base - ((k === "sw" ? ev.clientX : ev.clientY) - start); apply(); };
      const up = () => {
        sp.classList.remove("active");
        document.body.classList.remove("dragging", "x", "y");
        sp.removeEventListener("pointermove", move);
      };
      sp.addEventListener("pointermove", move);
      sp.addEventListener("pointerup", up, { once: true });
      sp.addEventListener("pointercancel", up, { once: true });
    });
    sp.addEventListener("dblclick", () => { state[state.mode][k] = DEF[state.mode][k]; apply(); });
  }
  addEventListener("resize", apply);
  apply();

  for (const name of ["work", "tl"]) {
    const el = app.querySelector("." + name);
    el.addEventListener("pointerover", () => { over = name; });
    el.addEventListener("pointerleave", () => { if (over === name) over = null; });
    // (capture: a panel's own handlers may stop the press)
    el.addEventListener("pointerdown", () => { clicked = name; }, true);
  }

  // mode: "land" or "port"; relayout: after the header's buttons change width (the Export badge);
  // onChange(fn): fn(mode) after each change
  return { mode: () => state.mode, setMode, relayout: apply, onChange: (fn) => { onChange = fn; } };
}

// The zoom keys (+ and -) go to one panel, "work" (the canvas) or "tl" (the timeline): the one
// under the pointer, else the one last clicked, else the canvas. The timeline only zooms with a
// video open; otherwise the canvas gets them.
let over = null, clicked = "work", timelineZooms = () => false;
export const setTimelineZooms = (fn) => { timelineZooms = fn; };
export function zoomKeysPanel() {
  const p = over || clicked;
  return p === "tl" && !timelineZooms() ? "work" : p;
}

// The settings' steps: a click on a step's header folds it; which are folded is remembered
export function initSteps(root) {
  const KEY_STEPS = "facejutsu.folded";
  let folded = [];
  try { folded = JSON.parse(localStorage.getItem(KEY_STEPS)) || []; } catch { /* private mode */ }
  for (const s of root.querySelectorAll(".step")) {
    const head = s.querySelector(".step-head");
    const set = (on) => { s.classList.toggle("folded", on); head.setAttribute("aria-expanded", String(!on)); };
    set(folded.includes(s.dataset.step));
    head.onclick = () => {
      set(!s.classList.contains("folded"));
      folded = [...root.querySelectorAll(".step.folded")].map((x) => x.dataset.step);
      try { localStorage.setItem(KEY_STEPS, JSON.stringify(folded)); } catch { /* private mode */ }
    };
  }
}

export const typing = (el) => el instanceof HTMLElement && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName));
