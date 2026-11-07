// Zoom and pan of the result on the canvas. The media fills the working area (fit), and zooms
// around the pointer with the wheel or a pinch; dragging pans it once zoomed. The tool pill zooms
// in steps, shows the zoom (displayed pixels per media pixel) and fits it back. Keys: + and -
// (when they go to the canvas, see layout.js), 0.

import { typing, zoomKeysPanel } from "./layout.js";

const $ = (id) => document.getElementById(id);
const STEP = 1.25;
const MIN = 0.25; // of the fitted size
const MAX_PX = 16; // displayed pixels per media pixel

export function initViewport(box, media, onChange = null) {
  let z = 1, x = 0, y = 0; // scale over the fitted size, offset of the centre in px

  // the shown element and its fitted scale (displayed px per media px)
  function shown() {
    const el = media.find((m) => !m.hidden);
    const w = el?.videoWidth ?? el?.width, h = el?.videoHeight ?? el?.height;
    return el && w && h ? { el, fit: Math.min(box.clientWidth / w, box.clientHeight / h), w, h } : null;
  }

  function apply() {
    const s = shown();
    const max = s ? MAX_PX / s.fit : 8;
    z = Math.min(max, Math.max(MIN, z));
    // keep the media on screen: zoomed in, its edges may not pass the box's
    if (s) {
      const mx = Math.max(0, (z * s.fit * s.w - box.clientWidth) / 2), my = Math.max(0, (z * s.fit * s.h - box.clientHeight) / 2);
      x = Math.min(mx, Math.max(-mx, x));
      y = Math.min(my, Math.max(-my, y));
    } else x = y = 0;
    box.style.setProperty("--zx", `${x}px`);
    box.style.setProperty("--zy", `${y}px`);
    box.style.setProperty("--z", z);
    box.classList.toggle("zoomed", z > 1);
    box.classList.toggle("pixels", !!s && z * s.fit >= 4);
    $("zoomLevel").textContent = s ? `${Math.round(100 * z * s.fit)}%` : "–";
    $("zoomOut").disabled = z <= MIN;
    $("zoomIn").disabled = z >= max;
    $("zoomFit").disabled = z === 1 && !x && !y;
    onChange?.();
  }

  // zoom by f around the point (cx, cy) in px from the box's centre
  function zoomAt(f, cx = 0, cy = 0) {
    const s = shown();
    const z2 = Math.min(s ? MAX_PX / s.fit : 8, Math.max(MIN, z * f));
    x = cx - ((cx - x) * z2) / z;
    y = cy - ((cy - y) * z2) / z;
    z = z2;
    apply();
  }
  const fromCentre = (e) => {
    const r = box.getBoundingClientRect();
    return [e.clientX - r.left - r.width / 2, e.clientY - r.top - r.height / 2];
  };
  const fit = () => { z = 1; x = y = 0; apply(); };

  box.addEventListener("wheel", (e) => {
    if (!shown()) return;
    e.preventDefault();
    // a pinch arrives as a wheel with ctrlKey and small deltas
    const d = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
    zoomAt(Math.exp(-d * (e.ctrlKey ? 0.01 : 0.0015)), ...fromCentre(e));
  }, { passive: false });

  box.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || z <= 1 || !shown() || e.target.closest("button, a, .panel-center, .progress-card")) return;
    e.preventDefault();
    box.setPointerCapture(e.pointerId);
    box.classList.add("panning");
    let px = e.clientX, py = e.clientY;
    const move = (ev) => { x += ev.clientX - px; y += ev.clientY - py; px = ev.clientX; py = ev.clientY; apply(); };
    const up = () => { box.classList.remove("panning"); box.removeEventListener("pointermove", move); };
    box.addEventListener("pointermove", move);
    box.addEventListener("pointerup", up, { once: true });
    box.addEventListener("pointercancel", up, { once: true });
  });
  box.addEventListener("dblclick", (e) => { if (shown() && !e.target.closest(".panel-center")) fit(); });

  $("zoomIn").onclick = () => zoomAt(STEP);
  $("zoomOut").onclick = () => zoomAt(1 / STEP);
  $("zoomFit").onclick = fit;
  addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey || typing(e.target) || !shown()) return;
    const keys = zoomKeysPanel() === "work";
    if (keys && (e.key === "+" || e.key === "=")) zoomAt(STEP);
    else if (keys && e.key === "-") zoomAt(1 / STEP);
    else if (e.key === "0") fit();
  });
  new ResizeObserver(apply).observe(box);
  for (const m of media) m.addEventListener("loadedmetadata", apply);

  // where a point of the shown media is in the box: { k (box px per media px), x0, y0 } with
  // box = x0 + k * media; null with none shown
  function map() {
    const s = shown();
    if (!s) return null;
    const k = z * s.fit;
    return { k, x0: box.clientWidth / 2 + x - (k * s.w) / 2, y0: box.clientHeight / 2 + y - (k * s.h) / 2 };
  }

  // fit: new media; update: the shown media changed size (keeps the zoom)
  return { fit, update: apply, map };
}
