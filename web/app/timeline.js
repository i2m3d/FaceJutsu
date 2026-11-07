// The timeline under (or beside) the canvas, where the photo or video to change is dropped: a
// transport, a ruler, a filmstrip of the clip and one track per face, each with segments where
// that face is on screen. It drives the Player: press and drag on the ruler or the filmstrip to
// scrub (frame-exact, the canvas follows at once), wheel to zoom around the pointer, Shift+wheel or
// a right/middle drag to pan, and the overview bar under a zoomed-in view to move it. Clicking a
// track's header selects that person (Ctrl adds it, Shift selects the tracks from the one last
// clicked); dragging it moves the track up or down, or onto another track merges it into that one;
// its × deletes the track.
// A track is cut into segments, each with its own settings: a click on one selects it (Ctrl adds
// it, Shift the segments from the one last clicked; a click between segments clears it), and a
// drag across the tracks selects the segments in its box (Ctrl or Shift adds them); a press there
// moves the playhead too. S splits the selected tracks at the playhead, J joins, Shift+J merges the
// selected tracks into one and Delete (or Backspace) deletes what's selected.
// I and O set the in and out points at the playhead (Alt+I, Alt+O, Alt+X clear them); once set,
// their markers on the ruler can be dragged, and the clip outside them is shaded. + and - zoom
// when the keys go to the timeline (see layout.js).
//
// The rows are laid out by the DOM (headers on the left, lanes on the right); one canvas over the
// lane column draws what's in them each animation frame, reading the lanes' boxes as it goes.

import { typing, zoomKeysPanel, setTimelineZooms } from "./layout.js";
import { rawToDisplay } from "./player.js";

const $ = (id) => document.getElementById(id);
const STEPS = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000, 20000, 50000, 100000, 200000, 500000];
const MAX_PPF = 60; // CSS pixels per frame at the closest zoom

const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");
const easeOut = (t) => 1 - (1 - t) ** 4;

export const timecode = (sec) => {
  const s = Math.max(0, sec);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = (s % 60).toFixed(2).padStart(5, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${r}` : `${m}:${r}`;
};

// a view glide from (left edge a0, span s0) to (a1, s1) that scales geometrically around the one
// point that stays put, as a real zoom looks; a plain pan when the span barely changes
function zoomPath(a0, s0, a1, s1) {
  const r = s1 / s0;
  if (Math.abs(1 - r) < 1e-3) return (k) => ({ left: a0 + (a1 - a0) * k, span: s0 + (s1 - s0) * k });
  const fixed = (a1 - a0 * r) / (1 - r);
  return (k) => {
    const span = s0 * r ** k;
    return { left: fixed - (fixed - a0) * (span / s0), span };
  };
}

export function initTimeline(player) {
  const grid = $("tlGrid"), canvas = $("tlCanvas"), ctx = canvas.getContext("2d");
  const rulerEl = $("tlRuler"), filmEl = $("tlFilm"), tracksEl = $("tlTracks"), overRow = $("tlOverRow"), overEl = $("tlOver");
  const play = $("tlPlay"), mute = $("tlMute"), auto = $("tlAuto"), time = $("tlTime");
  time.innerHTML = `<span></span><span class="dur"></span>`; // the duration hides in the narrow vertical layout
  const font = getComputedStyle(document.body).fontFamily || "system-ui, sans-serif";
  const mono = getComputedStyle(document.body).getPropertyValue("--font-mono").trim() || "ui-monospace, monospace";

  let tracks = []; // [{ id, label, color, segments: [[f0, f1) frames], parts: [{ id, f0, f1, url }] }]
  let rawTracks = [];
  let lastHeads = ""; // the tracks' headers as last drawn (JSON)
  let picked = new Set(); // the selected segments' ids
  const faces = new Map(); // url -> Image of a segment's face to use
  let media = null; // the player's media the view was fitted to
  let w = 0, h = 0;
  let start = 0; // frame at the left edge
  let ppf = 1; // CSS pixels per frame
  let anim = null; // { t0, dur, path, to: { start, ppf } }
  let drag = null;
  let transport = "";

  const total = () => Math.max(1, player.total);
  const x = (f) => (f - start) * ppf;
  const frameAt = (px) => start + px / ppf;

  function limits(s, p) {
    const n = total();
    p = Math.max(w / n / 1.05, Math.min(MAX_PPF, p));
    const visible = w / p;
    return { start: Math.max(-visible * 0.02, Math.min(n - visible * 0.98, s)), ppf: p };
  }
  function clamp() { ({ start, ppf } = limits(start, ppf)); }

  // move the view, gliding there unless `smooth` is off
  function goTo(s, p, smooth) {
    const to = limits(s, p);
    anim = null;
    if (smooth && w && !reduceMotion.matches) anim = { t0: performance.now(), dur: 280, path: zoomPath(start, w / ppf, to.start, w / to.ppf), to };
    else ({ start, ppf } = to);
    player.invalidate();
  }
  const fit = (smooth = false) => { if (w) goTo(0, w / total(), smooth); };
  function zoomBy(k, anchor = w / 2, smooth = true) {
    // repeated presses build on where a running zoom is heading, not where it is
    const from = anim?.to ?? { start, ppf };
    const f = from.start + anchor / from.ppf;
    const p = limits(from.start, from.ppf * k).ppf;
    goTo(f - anchor / p, p, smooth);
  }

  new ResizeObserver(() => {
    const r = canvas.getBoundingClientRect();
    const dpr = devicePixelRatio || 1;
    const fitted = w && Math.abs(w / ppf - total()) < 1e-6 * total() + 0.5;
    w = r.width;
    h = r.height;
    canvas.width = Math.max(1, Math.round(w * dpr));
    canvas.height = Math.max(1, Math.round(h * dpr));
    if (fitted || !media) fit();
    clamp();
    player.invalidate();
  }).observe(canvas);
  tracksEl.addEventListener("scroll", () => player.invalidate());

  // ------------------------------------------------------------ rows

  // the lanes' boxes in canvas coordinates
  function rows() {
    const c = canvas.getBoundingClientRect();
    const box = (el) => { const r = el.getBoundingClientRect(); return { y: r.top - c.top, h: r.height }; };
    const lanes = [...tracksEl.querySelectorAll(".tl-lane")].map(box);
    return { ruler: box(rulerEl), film: box(filmEl), tracks: box(tracksEl), lanes, over: overRow.hidden ? null : box(overEl) };
  }

  // ------------------------------------------------------------ interaction

  // the segment under a point of the canvas, where its person is on screen: { track, part } or null
  function partAt(r, p) {
    const i = r.lanes.findIndex((l) => p.y >= l.y && p.y < l.y + l.h);
    const tr = tracks[i];
    if (!tr?.parts || p.y < r.tracks.y || p.y > r.tracks.y + r.tracks.h) return null;
    const f = frameAt(p.x);
    if (!tr.segments.some(([a, b]) => f >= a && f < b)) return null;
    const part = tr.parts.find((q) => f >= q.f0 && f < q.f1);
    return part ? { track: tr, part } : null;
  }

  // how a click selects: "" only it, "toggle" (Ctrl/Cmd), "range" (Shift)
  const how = (e) => (e.shiftKey ? "range" : e.ctrlKey || e.metaKey ? "toggle" : "");
  const pos = (e) => { const r = canvas.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
  const scrub = (px) => player.seek(Math.floor(frameAt(px)));
  const BOX_SLOP = 4; // CSS pixels a press on the tracks moves before it's a drag, not a click

  // a selection box: its corners in frames and in the track list's content (so it stays put when
  // the view zooms, pans or scrolls under it), clamped to the list
  function boxRect(r) {
    const top = r.tracks.y - tracksEl.scrollTop;
    const ya = Math.max(r.tracks.y, Math.min(drag.y0, drag.y1) + top), yb = Math.min(r.tracks.y + r.tracks.h, Math.max(drag.y0, drag.y1) + top);
    return { fa: Math.min(drag.f0, drag.f1), fb: Math.max(drag.f0, drag.f1), ya, yb };
  }
  // the segments a box touches where their person is on screen
  function partsIn(r, b) {
    const ids = [];
    tracks.forEach((tr, i) => {
      const lane = r.lanes[i];
      if (!tr.parts || !lane || lane.y + lane.h - 4 <= b.ya || lane.y + 4 >= b.yb) return;
      for (const q of tr.parts) {
        if (tr.segments.some(([a, e]) => Math.max(a, q.f0, b.fa) < Math.min(e, q.f1, b.fb))) ids.push(q.id);
      }
    });
    return ids;
  }
  function moveBox(p) {
    const r = rows();
    drag.px = p;
    drag.f1 = frameAt(Math.max(0, Math.min(w, p.x)));
    drag.y1 = p.y - (r.tracks.y - tracksEl.scrollTop);
    const ids = partsIn(r, boxRect(r));
    const key = ids.join();
    if (key !== drag.key) { drag.key = key; api.onBox?.(ids, drag.how, !drag.started); drag.started = true; }
    player.invalidate();
  }

  // the in or out marker under a point on the ruler: "in", "out" or null
  function markerAt(r, p) {
    if (!player.range || p.y < r.ruler.y || p.y > r.ruler.y + r.ruler.h) return null;
    const di = Math.abs(p.x - x(player.inPoint)), dout = Math.abs(p.x - x(player.outPoint + 1));
    if (Math.min(di, dout) > 6) return null;
    return di < dout || (di === dout && p.x < x(player.inPoint)) ? "in" : "out";
  }
  function moveMarker(which, px) {
    if (which === "in") player.setRange(Math.min(player.outPoint, Math.round(frameAt(px))), player.outPoint);
    else player.setRange(player.inPoint, Math.max(player.inPoint, Math.round(frameAt(px)) - 1));
  }
  // The overview bar maps what the view can scroll over (the video and the little limits() lets
  // it go past each end) onto the lane's width, so the view's thumb always fits inside it: { lo
  // (frame at its left), k (px per frame), vx, vw (the thumb) }
  function overview() {
    const visible = w / ppf, lo = Math.min(0, start, -visible * 0.02), hi = Math.max(total(), start + visible, total() + visible * 0.02);
    const k = w / (hi - lo), vw = Math.min(w, Math.max(6, visible * k));
    return { lo, k, vw, vx: Math.max(0, Math.min(w - vw, (start - lo) * k)) };
  }
  function overviewTo(px, grab) {
    const o = overview();
    start = o.lo + (px - grab) / o.k;
    clamp();
    player.invalidate();
  }

  canvas.addEventListener("pointerdown", (e) => {
    if (!player.media) return;
    e.preventDefault();
    anim = null;
    const p = pos(e);
    canvas.setPointerCapture(e.pointerId);
    if (e.button === 1 || e.button === 2) { drag = { kind: "pan", x0: p.x, start0: start }; return; }
    const r = rows();
    if (r.over && p.y >= r.over.y) {
      const { vx, vw } = overview();
      drag = { kind: "overview", grab: p.x >= vx && p.x <= vx + vw ? p.x - vx : vw / 2 };
      overviewTo(p.x, drag.grab);
      return;
    }
    const mark = markerAt(r, p);
    if (mark) { drag = { kind: "marker", which: mark }; return; }
    // a press scrubs; while playing it keeps playing from there. On the tracks it selects the
    // segment under it at once (a plain press between segments clears the selection), and a drag
    // from there selects what its box touches instead.
    scrub(p.x);
    if (p.y >= r.tracks.y && p.y <= r.tracks.y + r.tracks.h && tracks.length) {
      const hit = partAt(r, p);
      if (hit) api.onPart?.(hit.track.id, hit.part.id, how(e));
      else if (!how(e)) api.onBox?.([], "", true);
      const y0 = p.y - (r.tracks.y - tracksEl.scrollTop);
      drag = { kind: "press", how: how(e), x0: p.x, y0: p.y, f0: frameAt(p.x), f1: frameAt(p.x), cy0: y0, y1: y0 };
      return;
    }
    drag = { kind: "scrub" };
  });
  canvas.addEventListener("pointermove", (e) => {
    const p = pos(e);
    if (!drag) {
      const r = player.media && rows();
      const onTracks = r && p.y >= r.tracks.y && p.y <= r.tracks.y + r.tracks.h && tracks.length;
      const f = r && Math.floor(frameAt(p.x));
      canvas.title = r && p.y >= railY(r.ruler) - 5 && p.y < r.ruler.y + r.ruler.h + 2 && f >= 0 && f < player.total ? statusText(f) : "";
      canvas.style.cursor = r && markerAt(r, p) ? "ew-resize" : (r?.over && p.y >= r.over.y) || (r && partAt(r, p)) ? "pointer" : onTracks ? "default" : player.media ? "col-resize" : "";
      return;
    }
    if (drag.kind === "pan") { start = drag.start0 - (p.x - drag.x0) / ppf; clamp(); player.invalidate(); }
    else if (drag.kind === "overview") overviewTo(p.x, drag.grab);
    else if (drag.kind === "marker") moveMarker(drag.which, p.x);
    else if (drag.kind === "press") {
      if (Math.hypot(p.x - drag.x0, p.y - drag.y0) < BOX_SLOP) return;
      Object.assign(drag, { kind: "box", y0: drag.cy0 });
      moveBox(p);
    } else if (drag.kind === "box") moveBox(p);
    else scrub(p.x);
  });
  const up = () => {
    if (drag?.kind === "box") player.invalidate();
    drag = null;
  };
  canvas.addEventListener("pointerup", up);
  canvas.addEventListener("pointercancel", up);
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("wheel", (e) => {
    if (!player.media) return;
    e.preventDefault();
    anim = null;
    if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
      start += (e.shiftKey ? e.deltaY || e.deltaX : e.deltaX) / ppf;
      clamp();
      player.invalidate();
      return;
    }
    zoomBy(Math.exp(-e.deltaY * 0.002), pos(e).x, false);
  }, { passive: false });

  setTimelineZooms(() => !!player.media);
  play.onclick = () => player.toggle();
  $("tlStart").onclick = () => { player.stop(); player.seek(player.inPoint); };
  mute.onclick = () => { player.muted = !player.muted; };
  addEventListener("keydown", (e) => {
    if (!player.media || e.ctrlKey || e.metaKey || typing(e.target) || e.target.closest?.("dialog")) return;
    const k = e.key;
    if (e.altKey) {
      // (by e.code: Alt changes e.key on some layouts)
      const clear = { KeyI: () => player.setIn(null), KeyO: () => player.setOut(null), KeyX: () => player.setRange(0, player.total - 1) }[e.code];
      if (clear) { e.preventDefault(); clear(); }
      return;
    }
    const act = {
      " ": () => player.toggle(),
      ArrowLeft: () => player.step(e.shiftKey ? -10 : -1),
      ArrowRight: () => player.step(e.shiftKey ? 10 : 1),
      Home: () => { player.stop(); player.seek(player.inPoint); },
      End: () => { player.stop(); player.seek(player.outPoint); },
      i: () => player.setIn(player.playhead), I: () => player.setIn(player.playhead),
      o: () => player.setOut(player.playhead), O: () => player.setOut(player.playhead),
      f: () => fit(true), F: () => fit(true),
      s: () => api.onSplit?.(), S: () => api.onSplit?.(),
      j: () => api.onJoin?.(), J: () => (e.shiftKey ? api.onMerge?.() : api.onJoin?.()),
      Delete: () => api.onDelete?.(), Backspace: () => api.onDelete?.(),
      m: () => mute.click(), M: () => mute.click(),
      w: () => auto.click(), W: () => auto.click(),
      ...(zoomKeysPanel() === "tl" && {
        "+": () => zoomBy(1.6), "=": () => zoomBy(1.6),
        "-": () => zoomBy(1 / 1.6), _: () => zoomBy(1 / 1.6),
      }),
    }[k];
    // Space on a focused button presses that button instead
    if (!act || (k === " " && e.target.closest?.("button, a"))) return;
    e.preventDefault();
    act();
  });

  // ------------------------------------------------------------ drawing

  function draw() {
    if (player.media !== media) {
      const prev = media;
      media = player.media;
      grid.hidden = !media;
      $("tlEmpty").hidden = !!media;
      play.disabled = $("tlStart").disabled = mute.disabled = auto.disabled = !media;
      tracks = toFrames(rawTracks);
      drawTracks();
      // a new video is shown whole; its swapped result keeps the view
      if (media && (!prev || prev.count !== media.count)) { fit(); anim = null; }
    }
    syncTransport();
    const dpr = devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!media || !w) return;
    if (anim) {
      const t = Math.min(1, (performance.now() - anim.t0) / anim.dur);
      const p = anim.path(easeOut(t));
      start = p.left;
      ppf = w / p.span;
      if (t >= 1) ({ start, ppf } = anim.to), anim = null;
      else player.invalidate();
    }
    if (player.playing || player.tracking) follow();
    if (drag?.kind === "box") edgeScroll();
    clamp();
    const zoomed = w / ppf < total() * 0.995;
    if (overRow.hidden === zoomed) { overRow.hidden = !zoomed; player.invalidate(); } // the boxes move: next frame

    const css = getComputedStyle(grid);
    const col = (n, d) => css.getPropertyValue(n).trim() || d;
    const c = { fg: col("--fg", "#eee"), muted: col("--muted", "#aaa"), faint: col("--faint", "#777"), line: col("--line-2", "#444"), bg: col("--panel", "#18181b"), good: col("--good", "#8fbf9f"), mark: col("--warn", "#d6b56e"), accent: col("--accent", "#d93a22"), accentInk: col("--accent-ink", "#ffffff"), ring: col("--ring", "#f2ede6"), ink: col("--ink", "#121014") };
    const r = rows();
    const n = player.total;
    const f0 = Math.max(0, Math.floor(start)), f1 = Math.min(n, Math.ceil(start + w / ppf));
    const endX = x(n);

    drawRuler(r.ruler, f0, f1, c);
    drawStatus(r.ruler, f0, f1, c);
    drawFilm(r.film, f0, f1, c);
    // face tracks, clipped to the (scrolling) track list
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, r.tracks.y, w, r.tracks.h);
    ctx.clip();
    drawn = [];
    tracks.forEach((tr, i) => {
      const lane = r.lanes[i];
      if (!lane || lane.y > r.tracks.y + r.tracks.h || lane.y + lane.h < r.tracks.y) return;
      // where the person is on screen, cut where their segments meet; a selected segment is lit
      const parts = tr.parts || [{ id: null, f0: 0, f1: Infinity }];
      for (const [a, b] of tr.segments) {
        for (const q of parts) {
          const pa = Math.max(a, q.f0), pb = Math.min(b, q.f1);
          if (pb <= pa) continue;
          const cutL = pa > a, cutR = pb < b; // a cut, not where the person comes or goes
          const sx = Math.max(-4, x(pa) + (cutL ? 1 : 0)), ex = Math.min(w + 4, x(pb) - (cutR ? 1 : 0));
          if (ex <= sx) continue;
          const on = q.id !== null && picked.has(q.id);
          // A sticker inset in the lane: the face's colour, solid, with an ink edge, sitting on a
          // shadow. Without a character (or still being scouted) it's hatched and outlined in the
          // colour instead. Selected: a paper outline and a red seal on its corner.
          const bare = q.id === null || !q.url;
          const bw = Math.max(2, ex - sx), by = lane.y + 3, bh = lane.h - 6;
          if (on) drawn.push([Math.max(sx, 0), Math.max(by, r.tracks.y), Math.min(ex, w), Math.min(by + bh, r.tracks.y + r.tracks.h)]);
          if (!bare) {
            round(sx, by + 2, bw, bh, 4);
            ctx.fillStyle = "rgba(0, 0, 0, 0.45)";
            ctx.fill();
          }
          round(sx, by, bw, bh, 4);
          ctx.fillStyle = bare ? hatch(c.fg) : tr.color;
          ctx.fill();
          const sw = on ? 2 : 1.5;
          round(sx + sw / 2, by + sw / 2, bw - sw, bh - sw, 4 - sw / 2);
          ctx.strokeStyle = on ? c.ring : bare ? tr.color : "rgba(0, 0, 0, 0.55)";
          ctx.lineWidth = sw;
          ctx.stroke();
          if (on && sx > -4) {
            ctx.fillStyle = c.accent;
            ctx.fillRect(sx - 1, by - 1, 8, 8);
          }
          // the character it becomes, at its start (kept in view) as far as there's room: its
          // picture (3:4) and its name, or "Not swapped"
          if (q.id !== null) drawBecomes(q, Math.max(sx, 0) + 3, Math.min(ex, w) - 4, by + 3, bh - 6, bare ? tr.color : c.ink);
        }
      }
    });
    if (drag?.kind === "box") drawBox(r, c);
    ctx.restore();
    // past the end of the video
    const bottom = r.over ? r.over.y - 2 : h;
    if (endX < w) {
      ctx.fillStyle = mix(c.bg, 0.6);
      ctx.fillRect(endX, r.ruler.y, w - endX, bottom - r.ruler.y);
    }
    if (player.range) drawRange(r, bottom, c);
    // the playhead: the frame on screen is shaded when it's wide enough to see
    const px = x(player.playhead);
    if (ppf >= 3) {
      ctx.fillStyle = mix(c.fg, 0.1);
      ctx.fillRect(px, r.film.y, ppf, bottom - r.film.y);
    }
    const cx = px + ppf / 2;
    if (cx >= -1 && cx <= w + 1) {
      ctx.strokeStyle = c.accent;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(cx, r.ruler.y + 10);
      ctx.lineTo(cx, bottom);
      ctx.stroke();
      // a badge with the frame number on the ruler, kept inside the lane
      ctx.font = `600 10px ${mono}`;
      const label = String(player.playhead), bw = ctx.measureText(label).width + 10;
      const bx = Math.max(0, Math.min(w - bw, cx - bw / 2));
      ctx.fillStyle = c.accent;
      round(bx, r.ruler.y + 1, bw, 15, 4);
      ctx.fill();
      ctx.fillStyle = c.accentInk;
      ctx.fillText(label, bx + 5, r.ruler.y + 12);
    }
    if (r.over) drawOverview(r.over, c);
  }
  player.onDraw(draw);
  let drawn = []; // the selected segments as last drawn, in view: [[x0, y0, x1, y1]] on the canvas

  // a selection box held past an edge of the tracks pans or scrolls them, faster further out
  function edgeScroll() {
    const p = drag.px, r = rows();
    if (!p) return;
    const by = (d) => Math.sign(d) * Math.min(40, Math.abs(d)) / 2;
    const dx = p.x < 0 ? p.x : p.x > w ? p.x - w : 0;
    const dy = p.y < r.tracks.y ? p.y - r.tracks.y : p.y > r.tracks.y + r.tracks.h ? p.y - r.tracks.y - r.tracks.h : 0;
    if (!dx && !dy) return;
    const s0 = start, top0 = tracksEl.scrollTop;
    if (dx) { start += by(dx) / ppf; clamp(); }
    if (dy) tracksEl.scrollTop += by(dy);
    if (start === s0 && tracksEl.scrollTop === top0) return;
    moveBox(p);
  }

  function drawBox(r, c) {
    const b = boxRect(r);
    const sx = x(b.fa), ex = x(b.fb);
    ctx.fillStyle = mix(c.fg, 0.08);
    ctx.fillRect(sx, b.ya, ex - sx, b.yb - b.ya);
    ctx.strokeStyle = mix(c.fg, 0.7);
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    ctx.strokeRect(Math.round(sx) + 0.5, Math.round(b.ya) + 0.5, Math.max(0, Math.round(ex - sx) - 1), Math.max(0, Math.round(b.yb - b.ya) - 1));
    ctx.setLineDash([]);
  }

  // keep the playhead in view while playing (or marking an export's place): page along with it
  function follow() {
    const visible = w / ppf, p = player.playhead;
    if (p < start || p > start + visible - 1) { anim = null; start = p - visible * 0.1; clamp(); }
  }

  // the stripes of a segment without a character, in colour (cached per colour)
  let hatchFor = "", hatchPat = null;
  function hatch(color) {
    if (hatchFor !== color) {
      const t = document.createElement("canvas");
      t.width = t.height = 8;
      const g = t.getContext("2d");
      g.strokeStyle = mix(color, 0.07);
      g.lineWidth = 3;
      g.beginPath();
      g.moveTo(-2, 10); g.lineTo(10, -2);
      g.moveTo(6, 10); g.lineTo(10, 6);
      g.moveTo(-2, 2); g.lineTo(2, -2);
      g.stroke();
      hatchPat = ctx.createPattern(t, "repeat");
      hatchFor = color;
    }
    return hatchPat;
  }

  // a segment's character between x0 and x1 at y (h tall): its picture, then its name as far as it
  // fits, in the colour given
  function drawBecomes(q, x0, x1, y, h, ink) {
    if (h <= 0) return;
    const img = q.url && face(q.url);
    let tx = x0 + 3;
    if (q.url) {
      const pw = Math.round((h * 3) / 4);
      if (x0 + pw > x1) return;
      if (img) {
        // the picture's middle, at 3:4
        const k = Math.min(img.naturalWidth / 3, img.naturalHeight / 4);
        ctx.save();
        round(x0, y, pw, h, 2);
        ctx.clip();
        ctx.drawImage(img, (img.naturalWidth - 3 * k) / 2, (img.naturalHeight - 4 * k) / 2, 3 * k, 4 * k, x0, y, pw, h);
        ctx.restore();
        round(x0 + 0.5, y + 0.5, pw - 1, h - 1, 2);
        ctx.strokeStyle = "rgba(0, 0, 0, 0.5)";
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      tx = x0 + pw + 6;
    }
    const text = q.url ? q.name : "Not swapped";
    const room = x1 - tx - 3;
    if (room < 24) return;
    ctx.font = `${q.url ? 600 : 500} 11px ${font}`;
    ctx.fillStyle = ink;
    ctx.textBaseline = "middle";
    let t = text;
    if (ctx.measureText(t).width > room) {
      while (t.length > 1 && ctx.measureText(t + "…").width > room) t = t.slice(0, -1);
      t += "…";
    }
    if (t.length > 2) ctx.fillText(t, tx, y + h / 2 + 0.5);
  }

  // (a lane squeezed below its padding gives a negative size: nothing is drawn, and roundRect
  // would throw on the negative radius)
  function round(rx, ry, rw, rh, rad) {
    ctx.beginPath();
    if (rw <= 0 || rh <= 0) return;
    ctx.roundRect(rx, ry, rw, rh, Math.min(rad, rw / 2, rh / 2));
  }
  // a #rgb / #rrggbb colour at alpha a
  function mix(color, a) {
    let hex = color.replace("#", "");
    if (hex.length === 3) hex = [...hex].map((d) => d + d).join("");
    const v = parseInt(hex.slice(0, 6), 16);
    return Number.isNaN(v) ? color : `rgba(${v >> 16}, ${(v >> 8) & 255}, ${v & 255}, ${a})`;
  }

  function drawRuler(row, f0, f1, c) {
    const y = row.y, rh = row.h;
    const step = STEPS.find((s) => s * ppf >= 70) ?? 1000000;
    const minor = STEPS.find((s) => s * ppf >= 8 && step % s === 0) ?? step;
    ctx.strokeStyle = c.line;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let f = Math.floor(f0 / minor) * minor; f <= f1; f += minor) {
      const tx = Math.round(x(f)) + 0.5;
      if (tx < 0) continue;
      const end = rh - RAIL - 3;
      ctx.moveTo(tx, y + end - (f % step === 0 ? 8 : 3));
      ctx.lineTo(tx, y + end);
    }
    ctx.stroke();
    for (let f = Math.floor(f0 / step) * step; f <= f1; f += step) {
      const tx = x(f);
      if (tx < 0 || f >= player.total) continue;
      ctx.font = `500 10px ${mono}`;
      ctx.fillStyle = c.muted;
      ctx.fillText(String(f), tx + 4, y + 13);
      const fw = ctx.measureText(String(f)).width;
      ctx.font = `10px ${mono}`;
      const tc = timecode(player.time(f));
      if (fw + ctx.measureText(tc).width + 22 <= step * ppf) {
        ctx.fillStyle = c.faint;
        ctx.fillText(tc, tx + 10 + fw, y + 13);
      }
    }
  }

  // one rail under the ruler's ticks for where each frame is at, the furthest along showing: the
  // faces found (only the keyframes: dimmer), the video decoded, then the swapped preview (to make:
  // dim, made: lit). A pixel's frames together, so a few missing frames between two made parts show.
  const RAIL = 3;
  const railY = (row) => row.y + row.h - RAIL - 1;
  function frameStatus(f) {
    const k = api.madeState?.(f) || 0;
    if (k) return 3 + k;
    if (media.frames.has(player.src(f))) return 3;
    return api.scanState?.(f) || 0;
  }
  function drawStatus(row, f0, f1, c) {
    const fill = [mix(c.fg, 0.08), mix(c.fg, 0.18), mix(c.fg, 0.3), mix(c.fg, 0.55), mix(c.good, 0.45), c.good];
    const y = railY(row), x0 = Math.max(0, x(0)), x1 = Math.min(w, x(player.total));
    if (x1 <= x0) return;
    ctx.save();
    round(x0, y, x1 - x0, RAIL, RAIL / 2);
    ctx.clip();
    const step = Math.max(1, 1 / ppf);
    let runStart = -1, run = 0;
    const flush = (end) => { ctx.fillStyle = fill[run]; ctx.fillRect(x(runStart), y, x(end) - x(runStart), RAIL); };
    for (let f = f0; f < f1; f += step) {
      // the swapped preview: to make where any frame still is; the rest: the most any frame has
      let s = 0;
      for (let g = Math.floor(f), end = Math.max(Math.min(f1, Math.floor(f + step)), g + 1); g < end; g++) {
        const k = frameStatus(g);
        if (k === 4) { s = 4; break; }
        if (k > s) s = k;
      }
      if (runStart < 0 || s !== run) { if (runStart >= 0) flush(f); runStart = f; run = s; }
    }
    if (runStart >= 0) flush(f1);
    ctx.restore();
  }
  // what the rail says at a frame, for its tooltip
  function statusText(f) {
    const k = frameStatus(f);
    return `Frame ${f}: ${["faces not looked for yet", "faces found at the keyframes", "faces found", "decoded", "to swap for the preview", "swapped"][k]}`;
  }

  function drawFilm(row, f0, f1, c) {
    const th = media.thumbs, info = media.info;
    const y = row.y + 3, fh = row.h - 6;
    const rot = info.rotation % 180 !== 0;
    const dispW = rot ? th.height : th.width, dispH = rot ? th.width : th.height;
    const slotW = Math.max(24, (fh * dispW) / dispH);
    ctx.save();
    ctx.beginPath();
    ctx.roundRect(Math.max(0, x(0)), y, Math.min(w, x(player.total)) - Math.max(0, x(0)), fh, 4);
    ctx.clip();
    ctx.fillStyle = "#000";
    ctx.fill();
    const want = [];
    const dpr = devicePixelRatio || 1;
    for (let s = Math.max(0, Math.floor(x(f0) / slotW)); ; s++) {
      const sx = s * slotW;
      if (sx > w) break;
      const f = Math.floor(frameAt(sx + slotW / 2));
      if (f < 0) continue;
      if (f >= f1 || f >= player.total) break;
      const src = player.src(f);
      const b = th.get(src);
      if (!b) { want.push(src); continue; }
      const scale = fh / dispH;
      const dw = dispW * scale;
      ctx.save();
      ctx.beginPath();
      ctx.rect(sx, y, slotW, fh);
      ctx.clip();
      ctx.setTransform(new DOMMatrix([dpr, 0, 0, dpr, 0, 0]).translate(sx + (slotW - dw) / 2, y).scale(scale)
        .multiply(rawToDisplay({ rawW: b.width, rawH: b.height, rotation: info.rotation, flip: info.flip })));
      ctx.drawImage(b, 0, 0);
      ctx.restore();
    }
    ctx.restore();
    if (want.length) th.request(want);
  }

  // the clip outside the in and out points shaded, and a bracket on the ruler at each
  function drawRange(r, bottom, c) {
    const ix = x(player.inPoint), ox = x(player.outPoint + 1);
    const top = r.ruler.y;
    ctx.fillStyle = mix(c.bg, 0.6);
    if (ix > 0) ctx.fillRect(Math.max(0, x(0)), top, Math.min(w, ix) - Math.max(0, x(0)), bottom - top);
    if (ox < w) ctx.fillRect(Math.max(0, ox), top, Math.min(w, x(player.total)) - Math.max(0, ox), bottom - top);
    ctx.fillStyle = mix(c.mark, 0.25);
    ctx.fillRect(ix, top, ox - ix, 3);
    ctx.strokeStyle = c.mark;
    ctx.lineWidth = 2;
    for (const [mx, dir] of [[ix, 1], [ox, -1]]) {
      if (mx < -8 || mx > w + 8) continue;
      ctx.beginPath();
      ctx.moveTo(mx + dir * 6, top + 1);
      ctx.lineTo(mx, top + 1);
      ctx.lineTo(mx, bottom);
      ctx.stroke();
    }
  }

  function drawOverview(row, c) {
    const y = row.y, oh = row.h;
    const { lo, k, vx, vw } = overview(), fx = (f) => (f - lo) * k;
    ctx.fillStyle = mix(c.line, 0.5);
    round(fx(0), y + oh / 2 - 1, fx(total()) - fx(0), 2, 1);
    ctx.fill();
    ctx.fillStyle = mix(c.fg, 0.16);
    ctx.strokeStyle = mix(c.fg, 0.4);
    ctx.lineWidth = 1;
    round(vx + 0.5, y + 0.5, vw - 1, oh - 1, 3);
    ctx.fill();
    ctx.stroke();
    if (player.range) {
      ctx.fillStyle = c.mark;
      ctx.fillRect(fx(player.inPoint), y + oh - 2, (player.outPoint + 1 - player.inPoint) * k, 2);
    }
    ctx.fillStyle = c.fg;
    ctx.fillRect(Math.min(w - 1.5, fx(player.playhead)), y, 1.5, oh);
  }

  function syncTransport() {
    const on = player.playing;
    const key = `${player.playhead}|${on}|${player.muted}|${player.total}|${player.inPoint}|${player.outPoint}`;
    if (key === transport) return;
    transport = key;
    play.classList.toggle("playing", on);
    play.title = on ? "Pause (Space)" : "Play (Space)";
    play.setAttribute("aria-label", on ? "Pause" : "Play");
    mute.classList.toggle("muted", player.muted);
    mute.title = player.muted ? "Unmute (M)" : "Mute (M)";
    mute.setAttribute("aria-label", player.muted ? "Unmute" : "Mute");
    time.firstChild.textContent = player.total ? timecode(player.time(player.playhead)) : "";
    time.lastChild.textContent = player.total ? ` / ${timecode(player.duration)}` : "";
    const range = player.range ? `\nIn ${player.inPoint} · out ${player.outPoint} (${timecode(player.time(player.outPoint + 1) - player.time(player.inPoint))})` : "";
    time.title = player.total ? `Frame ${player.playhead} of ${player.total - 1} · ${timecode(player.duration)} at ${+player.fps.toFixed(3)} fps${range}\n←/→ step a frame (Shift: 10) · wheel zooms, and +/− with the pointer over the timeline · Shift+wheel pans · F fits\nI / O set the in and out points · Alt+I, Alt+O, Alt+X clear them` : "";
    $("tlStart").title = player.range ? "Back to the in point (Home)" : "Back to the start (Home)";
    $("tlStart").setAttribute("aria-label", player.range ? "Back to the in point" : "Back to the start");
  }

  const toFrame = (t) => (t >= player.duration ? player.total : t <= 0 ? 0 : player.frameAt(t));
  const toFrames = (list) => player.total ? list.map((tr) => ({
    ...tr,
    segments: tr.segments.map(([a, b]) => [toFrame(a), toFrame(b)]),
    parts: tr.parts?.map((q) => ({ ...q, f0: toFrame(q.t0), f1: toFrame(q.t1) })),
  })) : [];

  // a face to use as an image, loaded once (drawn when it's in)
  function face(url) {
    let img = faces.get(url);
    if (!img) {
      img = new Image();
      img.onload = () => player.invalidate();
      img.src = url;
      faces.set(url, img);
    }
    return img.complete && img.naturalWidth ? img : null;
  }

  // a header per track: the person's name (a click selects them), a handle that shows the header
  // can be dragged and a × that deletes the track
  function drawTracks() {
    endMove();
    tracksEl.innerHTML = "";
    for (const tr of tracks) {
      const row = document.createElement("div");
      row.className = "tl-row track";
      row.dataset.id = tr.id;
      row.style.setProperty("--fc", tr.color);
      const hd = document.createElement("div");
      hd.className = "tl-h";
      if (tr.parts) {
        const grip = document.createElement("span");
        grip.className = "grip";
        grip.innerHTML = `<svg viewBox="0 0 6 10" aria-hidden="true"><circle cx="1" cy="1" r="1"/><circle cx="5" cy="1" r="1"/><circle cx="1" cy="5" r="1"/><circle cx="5" cy="5" r="1"/><circle cx="1" cy="9" r="1"/><circle cx="5" cy="9" r="1"/></svg>`;
        hd.title = "Drag to move the track, or onto another track to merge into it";
        hd.addEventListener("pointerdown", (e) => startMove(e, tr.id));
        hd.append(grip);
      }
      const who = document.createElement("button");
      who.type = "button";
      who.className = "who";
      who.innerHTML = `${tr.thumb ? `<img class="fc" alt="">` : ""}<span class="nm"></span>`;
      if (tr.thumb) who.querySelector(".fc").src = tr.thumb;
      who.querySelector(".nm").textContent = tr.label;
      who.onclick = (e) => api.onSelect?.(tr.id, how(e));
      hd.append(who);
      if (tr.parts) {
        const del = document.createElement("button");
        del.type = "button";
        del.className = "tbtn del";
        del.title = "Delete this track: its face is left as it is";
        del.setAttribute("aria-label", `Delete ${tr.label}`);
        del.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 7l10 10M17 7 7 17"/></svg>`;
        del.addEventListener("pointerdown", (e) => e.stopPropagation()); // (not a drag of the header)
        del.onclick = () => api.onRemove?.(tr.id);
        hd.append(del);
      }
      const lane = document.createElement("div");
      lane.className = "tl-lane";
      row.append(hd, lane);
      tracksEl.append(row);
    }
    markTracks();
  }

  // Dragging a track by its header: between two rows moves it there, onto the middle of another
  // row merges it into that one (which keeps its name and place). A row it can't merge into
  // (api.canMerge) shows as blocked, and a drop there does nothing. Escape cancels. A press that
  // doesn't move is the header's click (select) or double-click (rename); one that does isn't.
  let move = null; // { id, pointer, y0, on (moved past the threshold), to: { at } | { into, ok }, can }
  function startMove(e, id) {
    if (e.button !== 0 || move) return;
    move = { id, pointer: e.pointerId, y0: e.clientY, on: false, to: null, can: new Map() };
    addEventListener("pointermove", moving);
    addEventListener("pointerup", dropped);
    addEventListener("pointercancel", endMove);
    addEventListener("keydown", moveKey, true);
  }
  function moving(e) {
    if (e.pointerId !== move.pointer) return;
    if (!move.on) {
      if (Math.abs(e.clientY - move.y0) < 4) return;
      move.on = true;
      tracksEl.classList.add("moving");
      tracksEl.querySelector(`.track[data-id="${CSS.escape(String(move.id))}"]`)?.classList.add("lifted");
    }
    // scroll the tracks near their top and bottom edges
    const box = tracksEl.getBoundingClientRect();
    if (e.clientY < box.top + 16) tracksEl.scrollTop -= 8;
    else if (e.clientY > box.bottom - 16) tracksEl.scrollTop += 8;
    const rowsEl = [...tracksEl.children];
    let to = { at: rowsEl.length };
    for (const [i, row] of rowsEl.entries()) {
      const r = row.getBoundingClientRect();
      if (e.clientY >= r.bottom) continue;
      const k = (e.clientY - r.top) / r.height;
      const other = tracks[i].id !== move.id;
      if (other && k > 0.25 && k < 0.75) {
        const into = tracks[i].id;
        if (!move.can.has(into)) move.can.set(into, !!api.canMerge?.(move.id, into));
        to = { into, ok: move.can.get(into) };
      } else to = { at: k < 0.5 ? i : i + 1 };
      break;
    }
    move.to = to;
    showMove();
  }
  // the drop's mark: a line where the track would go, or the row it would merge into
  function showMove() {
    const from = tracks.findIndex((tr) => tr.id === move.id);
    for (const [i, row] of [...tracksEl.children].entries()) {
      const { at, into, ok } = move.to ?? {};
      // a drop next to where it is already doesn't move it
      const line = at !== undefined && at !== from && at !== from + 1;
      row.classList.toggle("ins-before", line && at === i);
      row.classList.toggle("ins-after", line && at === tracks.length && i === tracks.length - 1);
      row.classList.toggle("merge-into", into === tracks[i].id && ok);
      row.classList.toggle("no-merge", into === tracks[i].id && !ok);
    }
  }
  function dropped(e) {
    if (e.pointerId !== move.pointer) return;
    const { id, on, to } = move;
    endMove();
    if (!on) return;
    // the click this release makes isn't one on the header
    const eat = (c) => { c.stopPropagation(); c.preventDefault(); };
    addEventListener("click", eat, { capture: true, once: true });
    setTimeout(() => removeEventListener("click", eat, true));
    if (!to) return;
    if (to.into !== undefined) { if (to.ok) api.onMergeInto?.(id, to.into); return; }
    const from = tracks.findIndex((tr) => tr.id === id);
    const at = to.at > from ? to.at - 1 : to.at;
    if (at !== from) api.onMove?.(id, at);
  }
  function moveKey(e) {
    if (e.key !== "Escape") return;
    e.preventDefault();
    e.stopPropagation();
    endMove();
  }
  function endMove() {
    if (!move) return;
    move = null;
    removeEventListener("pointermove", moving);
    removeEventListener("pointerup", dropped);
    removeEventListener("pointercancel", endMove);
    removeEventListener("keydown", moveKey, true);
    tracksEl.classList.remove("moving");
    for (const row of tracksEl.children) row.classList.remove("lifted", "ins-before", "ins-after", "merge-into", "no-merge");
  }

  // a track with all its segments selected shows as selected, one with some of them lighter
  let wholeSet = new Set(); // the tracks selected as a whole
  function markTracks() {
    tracks.forEach((tr, i) => {
      const row = tracksEl.children[i];
      const n = tr.parts?.filter((q) => picked.has(q.id)).length ?? 0;
      row?.classList.toggle("sel", !!n && wholeSet.has(tr.id));
      row?.classList.toggle("some", !!n && !wholeSet.has(tr.id));
    });
  }

  const api = {
    madeState: null, // (frame) -> 0 nothing to swap, 1 to make, 2 made: on the status rail
    scanState: null, // (frame) -> 0 not looked at, 1 its keyframe looked at, 2 looked at: on the status rail
    onSelect: null, // (id, how) after a click on a track's header; how: "", "toggle" (Ctrl) or "range" (Shift)
    onPart: null, // (track id, segment id, how) after a press on a segment
    onBox: null, // (segment ids, how, first) as a drag's box changes what it touches; first: a new drag
    onSplit: null, // () on S
    onJoin: null, // () on J
    onMerge: null, // () on Shift+J
    onDelete: null, // () on Delete or Backspace
    onRemove: null, // (id) after a click on a track's ×
    onMove: null, // (id, index) after a track is dragged to another place in the list
    onMergeInto: null, // (id, into) after a track is dragged onto another one
    canMerge: null, // (id, into) -> whether the track can merge into the other (asked once per drag)
    // tracks: [{ id, label, color, thumb (the face, a URL), segments: [[t0, t1] seconds on screen], parts (the track's
    // segments: [{ id, t0, t1 (seconds), url and name (the character it becomes; "" for none) }];
    // leave it out for no handle) }]
    // (the headers stay, and a drag of one goes on, when only the segments changed)
    setTracks(list = []) {
      const heads = JSON.stringify(list.map((t) => [t.id, t.label, t.color, !!t.parts, t.thumb || ""]));
      rawTracks = list;
      tracks = toFrames(list);
      if (heads !== lastHeads) drawTracks();
      else markTracks();
      lastHeads = heads;
      player.invalidate();
    },
    // where the selected segments in view are on the window, as last drawn: a DOMRect, or null
    pickedRect() {
      const v = drawn.filter(([x0, y0, x1, y1]) => x1 > x0 && y1 > y0);
      if (!player.media || !v.length) return null;
      const c = canvas.getBoundingClientRect();
      const x0 = Math.min(...v.map((d) => d[0])), y0 = Math.min(...v.map((d) => d[1]));
      const x1 = Math.max(...v.map((d) => d[2])), y1 = Math.max(...v.map((d) => d[3]));
      return new DOMRect(c.left + x0, c.top + y0, x1 - x0, y1 - y0);
    },
    // the segment at a point of the window: { track, part } (ids) or null
    partAtClient(cx, cy) {
      if (!player.media) return null;
      const c = canvas.getBoundingClientRect();
      const hit = partAt(rows(), { x: cx - c.left, y: cy - c.top });
      return hit && { track: hit.track.id, part: hit.part.id };
    },
    // the selected segments' ids; shown: a track to scroll into view (the one last clicked)
    // whole: the tracks selected as a whole (their header lit), not only their segments
    pick(ids, shown = null, whole = []) {
      picked = new Set(ids);
      wholeSet = new Set(whole);
      markTracks();
      const i = tracks.findIndex((tr) => tr.id === shown);
      if (i >= 0 && picked.size) tracksEl.children[i]?.scrollIntoView({ block: "nearest" });
      player.invalidate();
    },
  };
  return api;
}
