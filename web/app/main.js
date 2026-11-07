import { fetchManifest, resolveFiles, listSwappers, listEnhancers, listOccluders } from "./pipeline.js";
import { createEngine } from "./engine.js";
import * as workspace from "./workspace.js";
import { hasWebNN, WEBNN_FLAG } from "./webnn/support.js";
import { NEAR_SECONDS } from "./queue.js";
import { initLayout, initSteps, typing } from "./layout.js";
import { initTimeline, timecode } from "./timeline.js";
import { initViewport } from "./viewport.js";
import { Player } from "./player.js";
import { Preview } from "./preview.js";
import { normalize, nearest, stretches, MIN_FRAMES } from "./tracks.js";
import { GAP } from "./scanner.js";
import * as saved from "./saved.js";
import { History } from "./history.js";
import { canEncodeVideo } from "./vendor/mediabunny.min.mjs";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
// Model files: the models folder, which the server serves here.
const BASE = new URL("./models/", location.href);

// ------------------------------------------------------------------ choices (plain language)

// the builds the app loads (precisions in order of preference): int8, which looks the same as the
// original models
const BUILDS = { swapper: "int8", recognizer: "int8", enhancer: ["int8", "f16"], occluder: "int8" };
// the people's colours on the timeline
const COLORS = ["#5aa9ff", "#ffb547", "#3ddc97", "#ff7eb6", "#a78bfa", "#2fd3d0", "#ff9a4d", "#c2e34a"];
const BACKENDS = [
  { id: "webgpu", label: "WebGPU", tip: "Works in any WebGPU browser" },
  { id: "webnn", label: "WebNN (faster)", tip: "Runs the face swapper through the browser's ML API on fp16 tensor cores" },
];
// pixelshuffle: the swapper works on crops of its own size (128 px); larger crops are split
// into sub-images of that size. With several picked (Ctrl / Shift click), each face gets the
// one of them its size in the frame needs (pipeline.js autoPixelshuffle)
// what the enhancer may change: only detail finer than the swap's (keeping its face, colours and
// shapes), or the whole face (the face enhancer)
const RESTORE = [{ id: "detail", label: "Subtle" }, { id: "full", label: "Strong" }];
// a segment's pixelshuffle: one of 1 to 4, or the sorted list of those picked; "auto" (older saves): all
const pixelshuffleSet = (v) => (v === "auto" ? [1, 2, 3, 4] : [v].flat());
const DETAIL = [
  { id: 1, label: "Fast" },
  { id: 2, label: "Balanced" },
  { id: 3, label: "High" },
  { id: 4, label: "Max" },
];
// the output (its popup, opened from the timeline), the same for the whole project. A format's codecs are Mediabunny's;
const OUT_FORMATS = [
  { id: "mp4", label: "MP4", tip: "Plays everywhere", ext: ".mp4", mime: "video/mp4", codecs: ["avc", "hevc", "av1"] },
];
const OUT_CODECS = [
  { id: "avc", label: "H.264", tip: "H.264: plays everywhere" },
  { id: "hevc", label: "H.265", tip: "H.265 (HEVC): smaller files at the same quality, but not every player has it" },
  { id: "vp9", label: "VP9", tip: "VP9: WebM's usual codec" },
  { id: "av1", label: "AV1", tip: "AV1: the smallest files, for recent players" },
];
const OUT_QUALITY = [
  { id: "medium", label: "Good", tip: "The smallest files" },
  { id: "high", label: "High", tip: "Larger files, a little more detail" },
  { id: "very-high", label: "Very high", tip: "Keeps the detail the swap adds: the largest files" },
];
const OUT_RANGE = [
  { id: "inout", label: "In to out", tip: "Only the part between the in and out points" },
  { id: "all", label: "Whole video", tip: "All of it, in and out points or not" },
];
// a photo is saved in its own format (other formats: PNG); JPEG and WebP at PHOTO_QUALITY
const PHOTO_QUALITY = 0.92;
const PHOTO_FORMATS = [
  { id: "png", label: "PNG", tip: "Lossless", ext: ".png", mime: "image/png" },
  { id: "jpeg", label: "JPEG", tip: "Smaller, lossy", ext: ".jpg", mime: "image/jpeg" },
  { id: "webp", label: "WebP", tip: "Smaller still, lossy", ext: ".webp", mime: "image/webp" },
];
// with several photos open: save the one open, or each of them
const PHOTO_SCOPE = [
  { id: "one", label: "This one", tip: "Only the photo open now" },
  { id: "all", label: "All", tip: "Each photo with a face given a character (the others are left out)" },
];

// ------------------------------------------------------------------ state

// a segment's settings as a fresh photo or video starts (and a reset leaves them)
const SEGMENT_DEFAULTS = { pixelshuffle: 2, blur: 0.3, enhance: "", restore: "detail", enhanceBlend: 0.8, occlusion: false, maskSize: 0 };
const store = {
  get(k, d) { try { const v = localStorage.getItem("facejutsu." + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("facejutsu." + k, JSON.stringify(v)); } catch { /* private mode */ } },
};
// enhance: the enhancer (a manifest component), "" for none; occlusion: keep what covers a face
// (off by default), its mask grown or shrunk by maskSize -1..1. pixelshuffle to
// maskSize are each segment's own (SEGMENT): here, what the next segments start with, the last
// ones set. enhancer: the one enhancer of earlier versions, for every face, which their segments'
// enhance (true / false) turned on or off (enhanceOf).
// out* and photo*: the output step's choices
const ui = { backend: "webgpu", ...SEGMENT_DEFAULTS, enhancer: "",
  outQuality: "very-high", outRange: "inout", photoScope: "all", saveMode: "download", theme: "system", ...store.get("ui", {}) };
const SEGMENT = ["pixelshuffle", "blur", "enhance", "restore", "enhanceBlend", "occlusion", "maskSize"];
// their names in the undo buttons' tooltips
const SEGMENT_NAMES = { pixelshuffle: "Detail", blur: "Edges", enhance: "Quality", restore: "Style", enhanceBlend: "Strength", occlusion: "Objects", maskSize: "Coverage" };
// a mask size as it's shown: signed, and plain at 0
const fmtMaskSize = (v) => (v ? `${v > 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}` : "0");
// a segment's enhancer from what it saved: earlier versions' on / off for the project's one
const enhanceOf = (v) => (typeof v === "boolean" ? (v ? ui.enhancer || "" : "") : v || "");
ui.enhance = enhanceOf(ui.enhance);
delete ui.preset;
delete ui.recognize; // videos' faces are always told apart by recognizing them now
delete ui.renderAhead; // the preview always swaps the rest of the video when idle now
delete ui.waitPreview; // playback always waits for the swap now, while there's one
// the output choices of earlier versions: a video is always MP4 (H.264 where it can be), at its
// own size, with its sound; a photo in its own format
for (const k of ["outFormat", "outCodec", "outSize", "outAudio", "photoFormat", "photoQuality"]) delete ui[k];
// auto: the GPU works ahead in the background (every face of the window around the playhead,
// and swapping the rest); off, only the frames around the playhead and a skim of the video, to
// keep the load low. (Earlier versions kept a mode name here: off now, as by default.)
if (typeof ui.auto !== "boolean") ui.auto = false;
if (ui.backend === "webnn" && !hasWebNN()) ui.backend = "webgpu";
const saveUi = () => store.set("ui", ui);

let manifest = null;
let engine = null;
let ready = false; // the detector and the recognizer loaded: faces can be found
let busy = false;
let exporting = false; // a video's export runs: the editor is locked, the playhead marks its place
let abort = null;
// the characters: identities made from one or more images dropped together (the mean of their
// largest faces' normalized ArcFace embeddings). Only those faces are kept, as small crops with
// their embeddings; the picture shows the ones closest to the mean. A character's key names its
// source in the engine; loaded: the engine has it.
const characters = []; // [{ key, name, embed, url, faces: [{ url, embed }], loaded }]
let nextChar = 1;
let making = 0; // characters whose images are being read
// the photos open together, in order; target is the one open. Each one's faces and segments are
// saved under its hash, like any photo's, so opening it again brings them back. url: its
// thumbnail; faces, swapped: how many faces it has and how many of them have a character, as it
// was last open (null: not opened yet)
const photos = []; // [{ file, url, width, height, faces, swapped, chars (their keys) }]
let saving = null; // saving every photo: { ac }
let working = false; // saving the open photo
let layout = null; // initLayout's: the layout and the settings column's width
// hash: of the file's content (saved.js), what its faces and segments are saved under
const target = { file: null, kind: "", hash: null, bitmap: null, width: 0, height: 0, duration: 0 };
// finding the faces of the target: status "" (not yet) | "running" (a photo's, or a video's until
// the first of them come) | "done" (a video's: the ones found so far, more as its scan goes on);
// people: one per person found, their track cut into segments (parts, in order of time), each
// swapped with its own settings: { id, t0 (seconds; it lasts until the next one), s: { source
// (the key of the face swapped in, null to keep theirs), ...SEGMENT }, off (deleted: not swapped
// nor shown) }. The first one also covers what's before it.
// faces: the people on screen in each analysed frame ([person ids] per frame, none in a frame not
// looked at yet), at times (seconds from the first frame shown)
// A video's: analysis, its faces as the engine has them (scanner.js), from its updates; scanning:
// a scan of the window around `center` (seconds) going on, kept between the in and out points as
// they were then (`span`), `coverage` of it done (0..1);
// finished: the last scan wasn't cut short; gone: the analysis's people whose tracks were
// deleted; thumbs: its people's thumbnails (data URLs)
const freshScan = () => ({ ac: null, status: "", frac: 0, frames: 0, people: [], faces: [], boxes: [], times: [],
  analysis: null, scanning: false, center: 0, span: "", coverage: 0, finished: false, gone: new Set(), thumbs: new Map(), imageAnalysis: null });
const scan = { id: 0, ...freshScan() };
let nextPart = 1;
let picked = new Set(); // the segments selected: the settings show and change theirs (none: every face's)
// the tracks (or a photo's faces) selected as a whole, by their header (or Ctrl+A): ident()s. A
// track counts as selected only while all its segments are too; segments clicked or boxed stay
// segments, even all of a track's.
let wholeTracks = new Set();
let lastTrack = null; // the person and the segment last clicked, where a Shift-click's range starts
let lastPart = null;
// the models loaded besides the detector and the recognizer: { swapper, enhancer (component), occluder }
let have = {};
// each one's load / unload button in its card: true loaded by hand (before a segment wants it), false
// unloaded by hand (until a segment wants it again); unset: loaded once a segment wants it, and kept
const keep = {};
let modelLoad = null; // loading or unloading them: { ac }
let modelSkip = ""; // the models whose load failed: not tried again by itself
let modelState = {}; // part -> its card's state while it loads { phase ("wait", "load"), frac }, or why it failed { failed }
let timeline = null;
let viewport = null;
let player = null; // plays the video on the canvas, driven by the timeline
let preview = null; // the swapped faces over the photo or video (preview.js)
let plan = null; // what the engine swaps with (swapOpts), as the preview last took it
let planJson = "";
let epoch = 1; // changes with the models and the characters: every frame's swap changes
let opened = null; // the scan whose photo or video the engine previews
const edits = new History(); // the open photo or video's edits, to undo (edit())

// ------------------------------------------------------------------ helpers

const fmtSize = (n) => (n >= 2 ** 30 ? `${(n / 2 ** 30).toFixed(1)} GB` : n >= 2 ** 20 ? `${Math.round(n / 2 ** 20)} MB` : `${Math.round(n / 2 ** 10)} KB`);
const fmtTime = (s) => (s < 60 ? `${Math.round(s)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`);

// Draws the middle of an image, or the area around a face's box (its shorter side 1.5x the box's,
// as the engine's thumbnails), kept inside the image, into the w x h area of ctx at x, y.
function drawCrop(ctx, img, box, x, y, w, h) {
  const fit = Math.min(img.width / w, img.height / h); // the largest w:h area the image holds
  let k = fit, cx = img.width / 2, cy = img.height / 2;
  if (box) {
    const [x1, y1, x2, y2] = box;
    k = Math.min(fit, (Math.max(x2 - x1, y2 - y1) * 1.5) / Math.min(w, h));
    cx = (x1 + x2) / 2;
    cy = (y1 + y2) / 2;
  }
  const sw = w * k, sh = h * k;
  const sx = Math.min(Math.max(cx - sw / 2, 0), img.width - sw), sy = Math.min(Math.max(cy - sh / 2, 0), img.height - sh);
  ctx.drawImage(img, sx, sy, sw, sh, x, y, w, h);
}

function canvas2d(w, h) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c;
}

// a w x h data URL of the middle of an image, or of the area around a face's box
function thumbUrl(img, box = null, w = 96, h = w) {
  const c = canvas2d(w, h);
  drawCrop(c.getContext("2d"), img, box, 0, 0, w, h);
  return c.toDataURL("image/jpeg", 0.85);
}

// A character's w x h picture of up to 9 of its faces ([{ bitmap, box }], the closest to its
// identity first): rows of about equal length, the shorter ones on top, so the closest face is the
// largest. One face fills it; 4 are 2 x 2, 9 are 3 x 3.
function facesUrl(faces, w = 192, h = 256) {
  faces = faces.slice(0, 9);
  const rows = Math.ceil(Math.sqrt(faces.length));
  const c = canvas2d(w, h);
  const ctx = c.getContext("2d");
  let i = 0;
  for (let r = 0; r < rows; r++) {
    const n = Math.floor((faces.length + r) / rows); // the rows' lengths add up to the count, the longest last
    const y0 = Math.round((r * h) / rows) + (r ? 1 : 0), y1 = Math.round(((r + 1) * h) / rows);
    for (let k = 0; k < n; k++, i++) {
      // a 1 px gap between the faces
      const x0 = Math.round((k * w) / n) + (k ? 1 : 0), x1 = Math.round(((k + 1) * w) / n);
      drawCrop(ctx, faces[i].bitmap, faces[i].box, x0, y0, x1 - x0, y1 - y0);
    }
  }
  return c.toDataURL("image/jpeg", 0.85);
}

const charByKey = (key) => characters.find((c) => c.key === key) || null;

// the models' state: what's missing or the loading (in place of the whole layout), or
// ready (on the timeline)
function showPanel(id) {
  for (const p of ["welcome", "loading", "tlReady"]) $(p).hidden = p !== id;
  const setup = id === "welcome" || id === "loading";
  $("setup").hidden = !setup;
  const was = $("app").classList.contains("setup");
  $("app").classList.toggle("setup", setup);
  if (was && !setup) layout?.relayout(); // the header is measured now it shows
}

// Says when the server stops, for as long as it's stopped, and for a moment when it's back. What was
// saved meanwhile didn't reach it, though it counts as saved (lastSaved, a photo's saved), so all of
// it is saved again then: with the faces, as what's found meanwhile isn't there for the edits to join.
// A server that comes back as another build (FaceJutsu was updated) is told apart from this page's:
// the page has to be reloaded, as it may not suit that server any more.
let serverTimer = 0, serverDown = false, pageBuild = null, outdated = false;
function renderServer() {
  if (!serverDown) return;
  $("serverText").textContent = exporting || saving || working
    ? "FaceJutsu has stopped, so saving is paused. Start it again and keep this page open: it carries on from where it got to."
    : "FaceJutsu has stopped. Start it again and keep this page open to carry on where you left off. Nothing is saved until then.";
}
function showServer(up) {
  clearTimeout(serverTimer);
  serverDown = !up;
  const el = $("serverCard");
  el.hidden = false;
  $("serverReload").hidden = true;
  if (!up) { renderServer(); return; }
  $("serverText").textContent = "FaceJutsu is back. Picking up where you left off.";
  for (const c of characters) saveCharacter(c);
  for (const ph of photos) ph.saved = null;
  saveOtherPhotos();
  saveAnalysis(true);
  serverTimer = setTimeout(() => { el.hidden = true; }, 4000);
  workspace.status().then((s) => {
    if (serverDown || !pageBuild || s.build === pageBuild) return;
    outdated = true;
    clearTimeout(serverTimer);
    $("serverText").textContent = "FaceJutsu has been updated. Reload this page to keep going.";
    $("serverReload").hidden = false;
    render();
  }).catch(() => {});
}

function showError(msg) {
  const el = $("errorCard");
  el.textContent = msg;
  el.hidden = !msg;
  if (msg) setTimeout(() => { if (el.textContent === msg) el.hidden = true; }, 12000);
}

function seg(el, items, get, set) {
  el.innerHTML = "";
  for (const it of items) {
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("role", "radio");
    b.textContent = it.label;
    b.dataset.label = it.label;
    b.title = typeof it.tip === "string" ? it.tip : "";
    b.onclick = () => { set(it.id); render(); };
    b.dataset.id = it.id;
    el.append(b);
  }
  el.sync = () => { for (const b of el.children) b.setAttribute("aria-checked", String(b.dataset.id === String(get()))); placeThumb(el); };
  segThumbs.observe(el);
}

// A control's thumb slides to the option picked, or stretches over a run of them (a multi-choice
// one's picks side by side; picks with gaps get a key each instead): the CSS draws it
// (.seg.thumb::before), this places it, in the next frame, reading every waiting control before writing any (no layout
// per control). It follows the control as the column resizes, and the first placing doesn't slide.
const thumbsDue = new Set();
const segThumbs = new ResizeObserver((es) => { for (const e of es) placeThumb(e.target); });
function placeThumb(el) {
  thumbsDue.add(el);
  if (thumbsDue.size === 1) requestAnimationFrame(placeThumbs);
}
function placeThumbs() {
  const due = [...thumbsDue].map((el) => {
    const kids = [...el.children].filter((b) => !b.hidden && /^(radio|checkbox)$/.test(b.getAttribute("role")));
    const picked = kids.flatMap((b, i) => (b.getAttribute("aria-checked") === "true" ? [i] : []));
    const run = picked.length && picked.at(-1) - picked[0] === picked.length - 1;
    const a = kids[picked[0]], z = kids[picked.at(-1)];
    return { el, on: run, at: run && el.offsetWidth ? [a.offsetLeft, z.offsetLeft + z.offsetWidth - a.offsetLeft] : null };
  });
  thumbsDue.clear();
  for (const { el, on, at } of due) {
    el.classList.toggle("thumb", !!on);
    if (!at) { el.classList.remove("anim"); continue; }
    el.style.setProperty("--tx", `${at[0]}px`);
    el.style.setProperty("--tw", `${at[1]}px`);
    if (!el.classList.contains("anim")) requestAnimationFrame(() => requestAnimationFrame(() => el.classList.add("anim")));
  }
}

// An on/off setting as a switch. items: [off, on] with their tips; get() is undefined when the
// segments differ ("mixed": a click turns it on for all of them).
function toggle(el, items, get, set, label) {
  el.innerHTML = "";
  const b = document.createElement("button");
  b.type = "button";
  b.className = "switch";
  b.setAttribute("role", "checkbox");
  b.setAttribute("aria-label", label);
  b.onclick = () => { set(get() !== true); render(); };
  el.append(b);
  el.sync = () => {
    const v = get();
    b.setAttribute("aria-checked", v === undefined ? "mixed" : String(!!v));
    b.title = v === undefined ? "Mixed: click to turn it on for all" : "";
  };
}

// A slider shows how far it's set (the CSS fills its track up to --v)
const fillRange = (el) => el.style.setProperty("--v", `${(100 * (el.value - el.min)) / (el.max - el.min || 1)}%`);

// A segmented control where Ctrl / Cmd click adds or takes out a choice and Shift click picks the
// ones from the last clicked to it; a plain click picks only that one. get, set: the list picked.
function multiSeg(el, items, get, set) {
  let last = null;
  el.innerHTML = "";
  for (const it of items) {
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("role", "checkbox");
    b.textContent = it.label;
    b.dataset.label = it.label;
    b.dataset.id = it.id;
    b.onclick = (e) => {
      const how = clickMode(e), cur = get() || [], ids = items.map((x) => x.id);
      let v;
      if (how === "toggle") v = cur.includes(it.id) ? cur.filter((x) => x !== it.id) : [...cur, it.id];
      else if (how === "range" && last != null) {
        const [a, z] = [ids.indexOf(last), ids.indexOf(it.id)].sort((p, q) => p - q);
        v = ids.slice(a, z + 1);
      } else v = [it.id];
      if (!v.length) return; // keep one
      if (how !== "range") last = it.id;
      set(ids.filter((x) => v.includes(x)));
      render();
    };
    el.append(b);
  }
  el.sync = () => { const v = get() || []; for (const b of el.children) b.setAttribute("aria-checked", String(v.some((x) => String(x) === b.dataset.id))); placeThumb(el); };
  segThumbs.observe(el);
}

// the face swapper: { id (manifest component), model, label, size }
function swapperInfo() {
  const all = manifest ? listSwappers(manifest) : [];
  return all.find((s) => s.id === "swapper") || all[0] || { id: "swapper", model: "inswapper", label: "inswapper_128", size: 128 };
}

// the face enhancers, smallest first: { id (manifest component), model, label, size, template }
const enhancers = () => (manifest ? listEnhancers(manifest).sort((a, b) => a.size - b.size) : []);
const enhancerById = (id) => (id && enhancers().find((e) => e.id === id)) || null;
// what the settings call each: Fast and Best (one: On)
const enhancerLabel = (id) => { const all = enhancers(), i = all.findIndex((e) => e.id === id); return i < 0 ? "Off" : all.length === 1 ? "On" : i === all.length - 1 ? "Best" : i === 0 ? "Fast" : "Better"; };
// the enhancer the settings' segments use (none: null; also while they differ)
const enhancerInfo = () => enhancerById(one("enhance"));
// several enhancers as have.enhancer and the like have them: their components, sorted, joined
// by commas ("" for none)
const enhancerIds = (v) => (v ? v.split(",") : []);
const enhancerSet = (ids) => [...new Set(ids)].filter((id) => enhancerById(id)).sort().join(",");

// the occlusion mask: { id (manifest component), model, label, size, template }, or null
function occluderInfo() {
  return (manifest && listOccluders(manifest)[0]) || null;
}

// the occlusion mask when a segment (or the next one) uses it and the manifest has one, else null
function occlusionOn() {
  return ui.occlusion || scan.people.some((p) => p.parts.some((q) => q.s.occlusion)) ? occluderInfo() : null;
}

// The models besides the detector and the recognizer that the segments swapping a face (with a
// character) want: { swapper, enhancer (components: enhancerSet), occluder }
function wanted() {
  const qs = scan.people.flatMap(live).filter((q) => charByKey(q.s.source));
  return {
    swapper: qs.length > 0,
    enhancer: enhancerSet(qs.map((q) => q.s.enhance)),
    occluder: !!occluderInfo() && qs.some((q) => q.s.occlusion),
  };
}

// whether a segment wants a model that isn't loaded
function modelsMissing() {
  const w = wanted();
  return (w.swapper && !have.swapper) || enhancerIds(w.enhancer).some((id) => !enhancerIds(have.enhancer).includes(id)) || (w.occluder && !have.occluder);
}

// The models to have: those the segments want; else as their buttons say, else those loaded. The
// enhancers: those the segments want, with the settings' one when its button loaded it, and those
// loaded while none is wanted and the button didn't unload them.
function modelTarget() {
  const w = wanted();
  const pick = (k, on, had) => (w[k] ? w[k] : keep[k] === false ? false : keep[k] ? on : had);
  return {
    swapper: !!pick("swapper", true, have.swapper),
    enhancer: keep.enhancer === false ? w.enhancer : enhancerSet([...enhancerIds(w.enhancer), ...(keep.enhancer ? [one("enhance")] : []),
      ...(!w.enhancer && keep.enhancer === undefined ? enhancerIds(have.enhancer) : [])]),
    occluder: !!pick("occluder", !!occluderInfo(), have.occluder),
  };
}

// What the engine loads: the detector and the recognizer, and the models of parts ({ swapper,
// enhancer (components: enhancerSet), occluder }).
function selection(parts = {}) {
  const e = enhancerIds(enhancerSet(enhancerIds(parts.enhancer)));
  const o = parts.occluder && occluderInfo();
  return { detector: "bf16", recognizer: BUILDS.recognizer, model: swapperInfo().id, ...(parts.swapper ? { swapper: BUILDS.swapper } : {}),
    ...(e.length ? { enhancer: BUILDS.enhancer, enhancerModels: e } : {}), ...(o ? { occluder: BUILDS.occluder, occluderModel: o.id } : {}) };
}

// `full` (the manifest in use) with only the extras (enhancers, the occlusion mask) the models folder
// has, so the settings don't offer the others. -> { manifest, missing (the files it can't start
// without), extras (those it hasn't of the others): each { name, size, what: what it does, optional } }
const MODEL_ROLES = {
  detector: "Finds the faces",
  recognizer: "Recognizes each person",
  swapper: "Swaps the face",
  "enhancer-256": "Sharpens faces, fast",
  "enhancer-512": "Sharpens faces, best",
  xseg: "Keeps hands and glasses in front",
};
function modelsThere(full, files) {
  const there = new Set(files.map((f) => f.name));
  const needed = new Set(Object.values(resolveFiles(full, selection({ swapper: true }))).map((f) => f.path));
  const missing = [];
  const extras = [];
  const components = {};
  for (const [id, c] of Object.entries(full.components)) {
    const builds = Object.values(c.builds);
    const file = (b, optional) => ({ name: b.path, size: b.size, what: MODEL_ROLES[id] || id, optional });
    if (builds.some((b) => needed.has(b.path))) {
      components[id] = c;
      missing.push(...builds.filter((b) => needed.has(b.path) && !there.has(b.path)).map((b) => file(b, false)));
    } else if (builds.some((b) => there.has(b.path))) components[id] = c;
    else extras.push(...builds.map((b) => file(b, true)));
  }
  return { manifest: { ...full, components }, missing, extras };
}

// ------------------------------------------------------------------ segments

const EPS = 1e-3; // seconds: a segment starts on a frame's own time

const newPart = (t0, s) => ({ id: nextPart++, t0, s: { ...s } });
const segmentDefaults = () => ({ source: null, ...Object.fromEntries(SEGMENT.map((k) => [k, ui[k]])) });

// the person's segment at t seconds
function partAt(p, t) {
  let q = p.parts[0];
  for (const x of p.parts) if (x.t0 <= t + EPS) q = x;
  return q;
}

// where a segment's person is on screen in it: [t0, t1] seconds, or null
function partSpan(p, q) {
  const i = p.parts.indexOf(q);
  const a = i === 0 ? -Infinity : q.t0, b = p.parts[i + 1]?.t0 ?? Infinity;
  const on = p.segments.map(([x, y]) => [Math.max(x, a), Math.min(y, b)]).filter(([x, y]) => y > x);
  return on.length ? [on[0][0], on[on.length - 1][1]] : null;
}

// The people the selection and the settings work on: a video's tracks, or the faces of every
// selected photo (the open one's are scan.people), each keyed by its photo and its id.
function people() {
  if (target.kind !== "image") return scan.people;
  const cur = openPhotoOf(), out = [];
  for (const ph of photos) {
    if (!ph.sel && (ph !== cur || opening)) continue; // (the one being left doesn't count)
    for (const x of ph === cur ? scan.people : ph.people || []) {
      x.key = `${ph.serial}:${x.id}`;
      out.push(x);
    }
  }
  return out;
}
// what names a person in the page (a track's or a face chip's data-id)
const pkey = (x) => x.key ?? String(x.id);
const findPerson = (id) => (id == null ? null : people().find((x) => pkey(x) === String(id)) || null);
// what lastTrack holds: a track's id, or a face's key
const ident = (x) => (target.kind === "image" ? pkey(x) : x.id);

const personOf = (partId) => people().find((p) => p.parts.some((q) => q.id === partId)) || null;
// a person's segments that aren't deleted
const live = (p) => p.parts.filter((q) => !q.off);
// whether a person's track is selected as a whole (its header), not only its segments
const isWhole = (p) => wholeTracks.has(ident(p)) && live(p).length > 0 && live(p).every((q) => picked.has(q.id));
const pickedParts = () => people().flatMap((p) => p.parts.filter((q) => picked.has(q.id)));
// the segments the settings are of: the selected ones, or every face's
const scopeParts = () => (picked.size ? pickedParts() : people().flatMap(live));

const same = (a, b) => a === b || (typeof a === "object" && typeof b === "object" && JSON.stringify(a) === JSON.stringify(b));
// a setting as the settings' segments have it: { v, mixed }; with none, the next ones'
function shown(k) {
  const ps = scopeParts();
  if (!ps.length) return { v: k === "source" ? null : ui[k], mixed: false };
  const v = ps[0].s[k];
  return { v, mixed: ps.some((q) => !same(q.s[k], v)) };
}
const val = (k) => shown(k).v;
const one = (k) => { const x = shown(k); return x.mixed ? undefined : x.v; };

// Sets a setting of the settings' segments (and of the segments to come); coalesce: a slider's
// drag, one undo step.
function setSegment(k, v, { coalesce = false } = {}) {
  if (busy) return;
  ui[k] = v;
  saveUi();
  edit(SEGMENT_NAMES[k], () => { for (const q of scopeParts()) q.s[k] = v; }, { coalesce });
  render();
}

// the character a person becomes: { src (a character, or null), mixed (their segments differ) }
function personSource(p) {
  const keys = [...new Set(live(p).map((q) => (charByKey(q.s.source) ? q.s.source : null)))];
  return { src: charByKey(keys.find((k) => k) ?? null), mixed: keys.length > 1 };
}

// how a click selects: "" only what was clicked, "toggle" (Ctrl/Cmd) adds it or takes it out
// again, "range" (Shift) from the one last clicked to it
const clickMode = (e) => (e.shiftKey ? "range" : e.ctrlKey || e.metaKey ? "toggle" : "");

// Selects the whole track of a person (null: nobody); how: as clickMode. A range is the tracks
// from the one last clicked to this one, in row order.
function selectPerson(id, how = "") {
  const all = people();
  const p = findPerson(id);
  if (p && how === "toggle") {
    const on = isWhole(p);
    for (const q of live(p)) if (on) picked.delete(q.id); else picked.add(q.id);
    if (on) wholeTracks.delete(ident(p)); else wholeTracks.add(ident(p));
  } else if (p && how === "range") {
    const a = all.findIndex((x) => ident(x) === lastTrack), b = all.indexOf(p);
    const who = a < 0 ? [p] : all.slice(Math.min(a, b), Math.max(a, b) + 1);
    picked = new Set(who.flatMap((x) => live(x).map((q) => q.id)));
    wholeTracks = new Set(who.map(ident));
  } else {
    picked = new Set(p ? live(p).map((q) => q.id) : []);
    wholeTracks = new Set(p ? [ident(p)] : []);
  }
  // a range keeps where it started, so the next Shift-click starts there too
  if (how !== "range" || !all.some((x) => ident(x) === lastTrack)) lastTrack = p ? ident(p) : null;
  selectionChanged();
}

// A click on a track's header or a photo's face card; a plain click on the only track selected
// (all of it) goes back to every face.
function clickPerson(id, how = "") {
  const p = findPerson(id);
  const only = p && isWhole(p) && picked.size === live(p).length;
  selectPerson(!how && only ? null : id, how);
}

// Selects a segment; how: as clickMode. A range is the segments from the one last clicked to this
// one when they're on the same track, added to the selection; on another track it toggles.
function selectPart(partId, how = "") {
  const p = personOf(partId);
  if (!p) return;
  const from = how === "range" && personOf(lastPart) === p ? live(p).findIndex((q) => q.id === lastPart) : -1;
  if (from >= 0) {
    const to = live(p).findIndex((q) => q.id === partId);
    for (const q of live(p).slice(Math.min(from, to), Math.max(from, to) + 1)) picked.add(q.id);
  } else if (!how) { picked = new Set([partId]); wholeTracks = new Set(); }
  else if (picked.has(partId) && picked.size > 1) picked.delete(partId);
  else picked.add(partId);
  if (from < 0) lastPart = partId;
  lastTrack = ident(p);
  selectionChanged();
}

// Selects the segments a drag's box on the timeline touches: only them, or with how ("toggle" or
// "range") added to what was selected when the drag began (first).
let boxBase = new Set();
function selectBox(ids, how, first) {
  if (first) {
    boxBase = how ? new Set(picked) : new Set();
    if (!how) wholeTracks = new Set();
  }
  picked = new Set([...boxBase, ...ids]);
  const last = ids.at(-1);
  if (last != null) { lastPart = last; lastTrack = personOf(last)?.id ?? lastTrack; }
  // (no scrolling a track into view: the list would move under the box)
  timeline.pick(picked, null, wholeIds());
  render();
}

// Selects every track (Ctrl/Cmd+A).
function selectAll() {
  picked = new Set(people().flatMap(live).map((q) => q.id));
  wholeTracks = new Set(people().map(ident));
  selectionChanged();
}

// the tracks selected as a whole, by id (for the timeline's headers)
const wholeIds = () => (target.kind === "video" ? scan.people.filter(isWhole).map((p) => p.id) : []);

function selectionChanged() {
  timeline.pick(picked, lastTrack, wholeIds());
  render();
}

// Splits the tracks with a selected segment at the playhead: the part after the cut starts with the
// settings it had, and is the one selected.
function split() {
  if (busy || target.kind !== "video" || scan.status !== "done") return;
  const t = player.time(player.playhead);
  const who = scan.people.filter((p) => p.parts.some((q) => picked.has(q.id)));
  const made = [];
  edit("split", () => {
    for (const p of who) {
      const q = partAt(p, t);
      // not in a deleted segment, where a cut is already, nor off screen (a segment there would
      // hold nothing)
      if (q.off || t <= q.t0 + EPS || !p.segments.some(([a, b]) => t > a + EPS && t < b)) continue;
      const n = newPart(t, q.s);
      p.parts.splice(p.parts.indexOf(q) + 1, 0, n);
      made.push(n.id);
    }
  });
  if (!made.length) { showError("Move the playhead inside a segment to split it."); return; }
  picked = new Set(made);
  renderTimeline();
  selectionChanged();
}

// Joins the selected segments of each person into one, with the first one's settings (the ones
// between them too); a person with only one selected is left as it is. A deleted segment between
// them stays deleted, and where the person comes back on screen stays a cut: the ones on each side
// are joined apart.
function join() {
  if (busy) return;
  const keep = [];
  edit("join", () => {
    for (const p of scan.people) {
      const sel = p.parts.filter((q) => picked.has(q.id));
      if (!sel.length) continue;
      if (sel.length === 1) { keep.push(sel[0].id); continue; }
      const first = sel[0], last = sel[sel.length - 1];
      // each run of segments between deleted ones becomes its first
      const a = p.parts.indexOf(first), b = p.parts.indexOf(last);
      const kept = [first];
      for (const q of p.parts.slice(a + 1, b + 1)) if (q.off || kept[kept.length - 1].off || atGap(p, q.t0)) kept.push(q);
      p.parts.splice(a, b - a + 1, ...kept);
      for (const q of kept) if (!q.off) keep.push(q.id);
    }
  });
  picked = new Set(keep);
  renderTimeline();
  selectionChanged();
}

// whether join() has something to join: two selected segments of one person, with a cut between
// them inside a stretch on screen
const canJoin = () => scan.people.some((p) => {
  const sel = p.parts.filter((q) => picked.has(q.id));
  if (sel.length < 2) return false;
  const a = p.parts.indexOf(sel[0]), b = p.parts.indexOf(sel[sel.length - 1]);
  return p.parts.slice(a + 1, b + 1).some((q, i) => !q.off && !p.parts[a + i].off && !atGap(p, q.t0));
});

// the tracks with a selected segment, in row order
const pickedPeople = () => people().filter((p) => p.parts.some((q) => picked.has(q.id)));

// where two of these people are on screen in the same frame: { i (the analysed frame), on (two of
// them) }, or null when never
function together(who) {
  const owner = new Map(who.flatMap((p) => p.ids.map((id) => [id, p])));
  for (const [i, ids] of scan.faces.entries()) {
    const on = [...new Set(ids.map((id) => owner.get(id)).filter(Boolean))];
    if (on.length > 1) return { i, on };
  }
  return null;
}

const canMergeTracks = () => !busy && target.kind === "video" && scan.status === "done";

// Merges the selected tracks into one, for a person found as several: the topmost keeps its name,
// colour and picture, and each stretch keeps the character and settings it had. Only for tracks
// never on screen in the same frame; otherwise the playhead goes to where they meet.
function merge() {
  if (!canMergeTracks()) return;
  const who = pickedPeople();
  if (who.length < 2) return;
  const met = together(who);
  if (met) {
    player.seek(player.frameAt(scan.times[met.i]));
    showError(`${met.on[0].label} and ${met.on[1].label} are both on screen at ${timecode(scan.times[met.i])}, so they can't be one track.`);
    return;
  }
  mergeTracks(who, who[0]);
}

// whether track `id` can merge into track `into`: other tracks, never on screen together
const canMergeInto = (id, into) => {
  const p = scan.people.find((x) => x.id === id), top = scan.people.find((x) => x.id === into);
  return canMergeTracks() && !!p && !!top && p !== top && !together([top, p]);
};

// Merges track `id` into track `into` (dragged onto it), which keeps its name, colour, picture and
// place.
function mergeInto(id, into) {
  if (!canMergeInto(id, into)) return;
  const top = scan.people.find((x) => x.id === into);
  mergeTracks([top, scan.people.find((x) => x.id === id)], top);
}

// Moves track `id` to place `at` in the list (as the list is after it's taken out).
function moveTrack(id, at) {
  const p = scan.people.find((x) => x.id === id);
  if (busy || !p) return;
  edit(`moving ${p.label}`, () => {
    const rest = scan.people.filter((x) => x !== p);
    rest.splice(Math.max(0, Math.min(rest.length, at)), 0, p);
    scan.people = rest;
  });
  renderTimeline();
  render();
}

// whether no segment of track p has a character
const neverGiven = (p) => p.parts.every((q) => !q.s.source);
// The character tracks `who` have, the one their segments are given (some may have it taken off);
// the tracks never given one get it on being merged with them. null when none, or several.
function trackChar(who) {
  const chars = new Set(who.flatMap(live).map((q) => q.s.source).filter(Boolean));
  return chars.size === 1 ? [...chars][0] : null;
}

// Merges `who` (never on screen together) into one track, as `top`.
function mergeTracks(who, top) {
  const merged = mergedTrack(who, top);
  if (!merged) return;
  edit("merging tracks", () => { scan.people = scan.people.filter((p) => p === top || !who.includes(p)).map((p) => (p === top ? merged : p)); });
  picked = new Set(merged.parts.map((q) => q.id));
  lastTrack = top.id;
  renderTimeline();
  selectionChanged();
}

// `who` (never on screen together) as one track, as `top`, each stretch with the settings it had;
// with coalesce, a segment with the same settings as the one before it is part of that one. A
// character given to the tracks goes to the stretches of the ones never given one (trackChar).
// -> the track, or null when none of them is on screen
function mergedTrack(who, top, { coalesce = false } = {}) {
  const owner = new Map(who.flatMap((p) => p.ids.map((id) => [id, p])));
  const char = trackChar(who);
  // frame by frame, the segment each face had on its own track (as the engine picks it): a new
  // one starts where that changes, so every stretch keeps its settings
  const from = [];
  for (const [i, ids] of scan.faces.entries()) {
    const p = owner.get(ids.find((id) => owner.has(id)));
    if (!p) continue;
    const q = partAt(p, scan.times[i]);
    if (from[from.length - 1]?.q !== q) from.push({ t0: scan.times[i], q, s: char && neverGiven(p) ? { ...q.s, source: char } : q.s });
  }
  if (!from.length) return null;
  // the stretches on screen, joined where they touch (new arrays: the history's snapshots share
  // a person's segments and ids)
  const runs = who.flatMap((p) => p.segments).sort((x, y) => x[0] - y[0]);
  const segments = [];
  for (const [a, b] of runs) {
    const last = segments[segments.length - 1];
    if (last && a <= last[1] + EPS) last[1] = Math.max(last[1], b);
    else segments.push([a, b]);
  }
  const used = new Set();
  const same = (x, y) => JSON.stringify([x.s, !!x.off]) === JSON.stringify([y.s, !!y.off]);
  const parts = [];
  for (const { t0, q, s } of from) {
    if (coalesce && parts.length && same(parts[parts.length - 1], { s, off: q.off })) continue;
    // a segment back after another track's stretch is a new one, with the same settings
    const n = used.has(q.id) ? newPart(t0, s) : { id: q.id, t0, s: { ...s } };
    if (q.off) n.off = true;
    used.add(q.id);
    parts.push(n);
  }
  const merged = { ...top, ids: who.flatMap((p) => p.ids), segments, frames: who.reduce((n, p) => n + p.frames, 0), parts };
  cutAtGaps(merged);
  return merged;
}

// Deletes what's selected on the timeline: a track selected by its header goes, of the others only
// the selected segments. Their faces are left as they are; undo brings them back.
function remove() {
  if (busy || target.kind !== "video" || scan.status !== "done") return;
  const who = pickedPeople();
  if (!who.length) return;
  const gone = who.filter(isWhole);
  const n = pickedParts().length - gone.reduce((k, p) => k + live(p).length, 0);
  const what = [
    gone.length && (gone.length === 1 ? gone[0].label : `${gone.length} tracks`),
    n && (n === 1 ? "a segment" : `${n} segments`),
  ].filter(Boolean).join(" and ");
  edit(`deleting ${what}`, () => {
    for (const p of who) if (!gone.includes(p)) for (const q of p.parts) if (picked.has(q.id)) q.off = true;
    scan.people = scan.people.filter((p) => !gone.includes(p));
    for (const p of gone) for (const id of p.ids) scan.gone.add(id);
  });
  picked = new Set();
  wholeTracks = new Set();
  lastPart = null;
  if (gone.some((p) => p.id === lastTrack)) lastTrack = null;
  renderTimeline();
  selectionChanged();
}

// Deletes one track, the × on its header, whatever is selected: the only way to delete a track with
// no segments left, which can't be selected. The rest of the selection stays.
function removeTrack(id) {
  if (busy || target.kind !== "video" || scan.status !== "done") return;
  const p = scan.people.find((x) => x.id === id);
  if (!p) return;
  edit(`deleting ${p.label}`, () => {
    scan.people = scan.people.filter((x) => x !== p);
    for (const x of p.ids) scan.gone.add(x);
  });
  for (const q of p.parts) {
    picked.delete(q.id);
    if (q.id === lastPart) lastPart = null;
  }
  wholeTracks.delete(p.id);
  if (lastTrack === p.id) lastTrack = null;
  renderTimeline();
  selectionChanged();
}

// What the settings change, in the card over them: every face, or what's selected on the timeline.
// Who (their faces, and their colours as on their tracks), a title and a line of detail; with a
// selection, the paper outline the selection has on the timeline and a way back to every face.
// And whether split, join and merge can.
// It never flickers: what a click changes (the title, the outline) changes at once, but while
// another photo's faces are being found the last faces stay, and only a slow scan (over 500 ms)
// says so ("Scouting faces…", the faces dimmed), and then for at least 400 ms; new faces are
// swapped in once decoded; and nothing on it is touched unless it changed.
let scopeHold = 0; // when the faces started being found (0: they aren't)
let scopeSlowAt = 0; // when it started saying so (0: it isn't)
const SCOPE_SLOW = 500, SCOPE_MIN = 400;
function renderScope() {
  const ps = pickedParts();
  const all = people();
  const who = all.filter((p) => p.parts.some((q) => picked.has(q.id)));
  const sel = target.kind === "image" ? selectedPhotos().length : 1;
  const p = who[0];
  const video = target.kind === "video";
  const finding = !all.length && (scan.status !== "done" || !!opening);
  const now = performance.now();
  if (!finding) scopeHold = 0;
  else if (!scopeHold) {
    scopeHold = now;
    setTimeout(() => { if (scopeHold) renderScope(); }, SCOPE_SLOW + 10);
  }
  const slow = finding && now - scopeHold >= SCOPE_SLOW;
  if (slow && !scopeSlowAt) scopeSlowAt = now;
  // (once shown, "Scouting faces…" stays a moment: a scan ending just after it never flashes it)
  let linger = false;
  if (!slow && scopeSlowAt) {
    const left = SCOPE_MIN - (now - scopeSlowAt);
    if (left > 0) { linger = true; setTimeout(renderScope, left + 10); } else scopeSlowAt = 0;
  }
  let title, sub = "";
  if (!ps.length) {
    title = sel > 1 ? `Every face in ${sel} photos` : "Every face";
    sub = slow || linger ? "Scouting faces…"
      : !all.length && !finding ? `No faces found in this ${video ? "video" : "photo"}` : `Select a face${video ? " on the timeline" : ""} to change only that one`;
  } else if (who.length > 1) {
    title = `${who.length} faces`;
    sub = video ? `${ps.length} segments` : "Selected together";
  } else if (!video) {
    title = p.label;
    sub = "This face only";
  } else if (isWhole(p)) {
    const n = live(p).length;
    title = p.label;
    sub = `The whole track${n > 1 ? ` · ${n} segments` : ""}`;
  } else if (ps.length === 1) {
    const span = partSpan(p, ps[0]);
    title = p.label;
    sub = `Segment ${live(p).indexOf(ps[0]) + 1} of ${live(p).length}${span ? ` · <span class="num">${timecode(span[0])} – ${timecode(span[1])}</span>` : ""}`;
  } else {
    title = p.label;
    sub = `${ps.length} of ${live(p).length} segments`;
  }
  // (it stays while the faces are being found, the same size, so nothing under it jumps when
  // another photo opens)
  const box = $("scope");
  const set = (el, k, v) => { if (el[k] !== v) el[k] = v; };
  set(box, "hidden", !target.file);
  if (target.file) {
    const shown = ps.length ? who : all;
    if (box.classList.contains("on") !== !!ps.length) box.classList.toggle("on", !!ps.length);
    // the faces' colours down its edge, as on their tracks (a band each, with several)
    const cols = ps.length ? who.map((x) => x.color) : [];
    const sc = !cols.length ? "var(--line-2)" : cols.length === 1 ? cols[0]
      : `linear-gradient(${cols.map((c, i) => `${c} ${(100 * i) / cols.length}% ${(100 * (i + 1)) / cols.length}%`).join(", ")})`;
    if (box.style.getPropertyValue("--sc") !== sc) box.style.setProperty("--sc", sc);
    // the faces: while new ones are being found, the last ones stay (dimmed, once it's slow)
    // (one face, and how many more on a badge: the slot keeps its size, so the text never moves)
    const withThumb = shown.filter((x) => x.thumb);
    const faces = linger ? [] : withThumb.slice(0, 1);
    const more = linger ? 0 : Math.max(0, shown.length - 1);
    const el = $("scopeWho");
    const hold = (finding || linger) && !faces.length && !!el.querySelector("img");
    if (el.classList.contains("held") !== (hold && (slow || linger))) el.classList.toggle("held", hold && (slow || linger));
    const key = faces.length ? `${faces[0].thumb}|${more}` : finding ? (slow ? "finding slowly" : "finding") : "none";
    if (!hold && el.dataset.key !== key) {
      el.dataset.key = key;
      if (faces.length) {
        const img = Object.assign(new Image(), { src: faces[0].thumb, alt: "" });
        const badge = more ? [Object.assign(document.createElement("b"), { className: "more", textContent: `+${more}` })] : [];
        img.decode().catch(() => {}).then(() => { if (el.dataset.key === key) el.replaceChildren(img, ...badge); });
      } else el.innerHTML = `<span class="ph">${finding ? (slow ? `<i class="spin"></i>` : "") : `<svg viewBox="0 0 24 24"><circle cx="12" cy="9" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/></svg>`}</span>`;
    }
    set($("scopeTitle"), "textContent", title); // (a name the user gave: as text)
    set($("scopeSub"), "innerHTML", sub); // (numbers and fixed words only)
    set($("scopeAll"), "hidden", !ps.length);
  }
  renderSelBar();
}

// whether split() has something to cut: the playhead inside a selected segment, where its person
// is on screen and no cut is already
function canSplit() {
  if (target.kind !== "video" || scan.status !== "done" || !player.media) return false;
  const t = player.time(player.playhead);
  return pickedPeople().some((p) => {
    const q = partAt(p, t);
    return picked.has(q.id) && !q.off && t > q.t0 + EPS && p.segments.some(([a, b]) => t > a + EPS && t < b);
  });
}

// The bar over the selection: a video's selected segments (split, join, merge, delete, the ones
// that apply), or the selected photos when there are several (remove them from the set). Placed
// just above it, kept inside the timeline (below it when there's no room above); hidden with no
// selection, or when the selection is out of view.
function renderSelBar() {
  const bar = $("selBar"), panel = $("tgtDrop");
  let rect = null, label = "";
  const video = target.kind === "video" && scan.status === "done" && !$("tlGrid").hidden;
  if (!busy && !saving && video && picked.size) {
    const ps = pickedParts(), who = pickedPeople();
    const whole = who.filter(isWhole);
    label = whole.length === who.length ? (who.length === 1 ? who[0].label : `${who.length} tracks`) : `${ps.length} segment${ps.length > 1 ? "s" : ""}`;
    $("splitBtn").hidden = !canSplit();
    $("joinBtn").hidden = !canJoin();
    $("mergeBtn").hidden = who.length < 2;
    $("deleteBtn").hidden = false;
    $("psDelete").hidden = true;
    rect = timeline.pickedRect();
  } else if (!busy && !saving && target.kind === "image" && !$("photoSet").hidden && selectedPhotos().length > 1) {
    const sel = selectedPhotos();
    label = `${sel.length} photos`;
    for (const id of ["splitBtn", "joinBtn", "mergeBtn", "deleteBtn"]) $(id).hidden = true;
    $("psDelete").hidden = false;
    const strip = $("psStrip").getBoundingClientRect();
    const rs = [...$("psStrip").querySelectorAll('.ps-pick[aria-selected="true"]')].map((b) => b.getBoundingClientRect())
      .map((r) => [Math.max(r.left, strip.left), Math.max(r.top, strip.top), Math.min(r.right, strip.right), Math.min(r.bottom, strip.bottom)])
      .filter(([x0, y0, x1, y1]) => x1 > x0 && y1 > y0);
    if (rs.length) {
      const x0 = Math.min(...rs.map((r) => r[0])), y0 = Math.min(...rs.map((r) => r[1]));
      rect = new DOMRect(x0, y0, Math.max(...rs.map((r) => r[2])) - x0, Math.max(...rs.map((r) => r[3])) - y0);
    }
  }
  bar.hidden = !rect;
  if (!rect) return;
  if ($("selLab").textContent !== label) $("selLab").textContent = label;
  const pr = panel.getBoundingClientRect(), m = 6;
  const bw = bar.offsetWidth, bh = bar.offsetHeight;
  let top = rect.top - pr.top - bh - m;
  if (top < m) top = Math.min(rect.bottom - pr.top + m, pr.height - bh - m);
  bar.style.left = `${Math.round(Math.max(m, Math.min(pr.width - bw - m, rect.left - pr.left)))}px`;
  bar.style.top = `${Math.round(Math.max(m, top))}px`;
}

// ------------------------------------------------------------------ rendering

function render() {
  renderServer();
  $("detailSeg").sync();
  for (const b of $("detailSeg").children) b.disabled = busy;
  const blur = shown("blur");
  $("blurRange").value = blur.v;
  $("blurRange").disabled = busy;
  $("blurOut").textContent = blur.mixed ? "mixed" : blur.v.toFixed(2);
  $("occlusionRow").hidden = !occluderInfo();
  $("occlusionSeg").sync?.();
  for (const b of $("occlusionSeg").children) b.disabled = busy;
  const maskSize = shown("maskSize");
  $("maskSizeRow").hidden = !occluderInfo() || one("occlusion") === false;
  $("maskSizeRange").value = maskSize.v;
  $("maskSizeRange").disabled = busy;
  $("maskSizeOut").textContent = maskSize.mixed ? "mixed" : fmtMaskSize(maskSize.v);
  $("enhanceStep").hidden = !enhancers().length;
  $("enhanceSeg").sync?.();
  for (const b of $("enhanceSeg").children) b.disabled = busy;
  const enhOn = !!val("enhance") || shown("enhance").mixed;
  $("enhanceModeRow").hidden = !enhOn;
  $("enhanceModeSeg").sync();
  for (const b of $("enhanceModeSeg").children) b.disabled = busy;
  $("enhanceBlendRow").hidden = !enhOn;
  const blend = shown("enhanceBlend");
  $("enhanceBlendRange").value = blend.v;
  $("enhanceBlendRange").disabled = busy;
  $("enhanceBlendOut").textContent = blend.mixed ? "mixed" : `${Math.round(100 * blend.v)}%`;

  const assigned = scan.people.filter((p) => personSource(p).src).length;
  const some = assigned || (allPhotos() && photos.some((p) => p.swapped && p.file !== target.file));
  // why Export can't run, in its tooltip
  const why = outdated ? "Reload this page to keep going: FaceJutsu has been updated"
    : !target.file ? "Open a photo or video first"
    : scan.status !== "done" ? (scan.scanning ? `Scouting faces… ${Math.round(100 * scan.coverage)}%` : "Scouting faces…")
      : !some ? "Give a face a character first"
        : !ready || !have.swapper || modelsMissing() ? "The models are loading" : "";
  const go = $("goBtn"), run = busy || !!saving || working;
  go.disabled = !!why || run;
  $("goLabel").textContent = target.kind === "image" ? "Save" : "Export";
  const badge = goBadge();
  if ($("goBadge").textContent !== badge) {
    $("goBadge").textContent = badge;
    layout?.relayout(); // the header's width changed
  }
  go.title = why || (target.kind === "image" ? (allPhotos() ? `Save ${photos.length} photos: ${outSummary()}` : `Save the photo: ${outSummary()}`) : `Export the video: ${outSummary()}`);
  if (go.hidden !== run) {
    go.hidden = run;
    $("goRun").hidden = !run;
    layout?.relayout();
  }
  $("stopBtn").hidden = !abort;
  $("compareBtn").hidden = !assigned || !target.file;
  renderScanStatus();
  $("app").classList.toggle("saving", !!saving);
  $("app").classList.toggle("exporting", exporting);
  for (const b of document.querySelectorAll(".tgt-clear")) b.hidden = busy;
  $("tlTracks").classList.toggle("locked", busy);
  $("tgtInput").disabled = busy || !!saving;
  $("resetBtn").disabled = busy || !hasEdits();
  $("resetBtn").lastChild.textContent = target.kind === "image" ? "Reset this photo" : "Reset this video";

  const occ = occlusionOn();
  $("readyText").textContent = "Everything runs on this device. Your face never leaves the dojo.";

  renderSteps();
  renderOutput();
  renderScope();
  renderHistory();
  renderPeople();
  renderPhotos();
  renderSelBar();
  renderChars();

  const showingSomething = !$("view").hidden || !$("player").hidden;
  $("toolPill").hidden = !showingSomething;
  viewport?.update();
  // until there's a photo or video, the viewport is where it's dropped; until there's a character,
  // the settings' first part is where its images go
  $("empty").hidden = !!target.file;
  $("app").classList.toggle("no-media", !target.file);
  $("srcStart").hidden = characters.length > 0 || making > 0;
  $("facesBody").hidden = !$("srcStart").hidden;
  if (ready) showPanel(target.file ? "" : "tlReady");
  saveSoon();
  updatePlan();
  // a model a segment just came to want (a character given to a face, the enhancer or the
  // occlusion mask turned on), or one loaded or unloaded with its card's button, is loaded or unloaded now
  renderModels();
  syncModels();
  for (const r of document.querySelectorAll('input[type="range"]')) fillRange(r);
}

// ------------------------------------------------------------------ preview

// Hands the preview what's swapped now, when it changed.
function updatePlan() {
  if (!preview) return;
  const o = swapOpts();
  const json = JSON.stringify([epoch, o]);
  if (json !== planJson) {
    planJson = json;
    plan = o;
    preview.invalidate();
  }
  preview.kick();
  renderBadge();
}

// the analysed frame of frame f of the target
const frameIndex = (f) => (target.kind === "video" ? nearest(scan.times, player.time(f)) : 0);

// The faces of frame f the engine swaps, as it picks their settings (settingsAt in session.js):
// [[person, their segment's settings]] of the faces there with a character.
function swappedFaces(f) {
  if (!plan || scan.status !== "done" || plan.analysis !== scan.id) return [];
  const i = frameIndex(f);
  const t = (scan.times[i] ?? 0) + EPS;
  const faces = [];
  for (const id of scan.faces[i] || []) {
    const parts = plan.plan[id];
    if (!parts?.length) continue;
    let q = parts[0];
    for (const p of parts) if (p.t0 <= t) q = p;
    if (q.source) faces.push([id, q]);
  }
  return faces;
}

// what frame f shows swapped; null: nothing. Where a segment starts isn't part of it, so a merge
// (which starts segments elsewhere) doesn't swap again frames whose settings stay the same.
function frameKey(f) {
  const faces = swappedFaces(f).map(([id, { t0, ...s }]) => [id, s]);
  return faces.length ? `${epoch}:${scan.id}:${JSON.stringify(faces)}` : null;
}

// whether the preview can swap now
const previewRuns = () => ready && !busy && !pausePreview && !modelsMissing() && scan.status === "done" && !!target.file && (target.kind !== "image" || !!target.bitmap);

// Opens the photo or video in the engine for the preview. -> false when it changed meanwhile
async function openPreviewTarget() {
  const id = scan.id;
  if (opened === id) return true;
  const file = target.file;
  const key = target.hash || `${file.name}:${file.size}:${file.lastModified}`;
  await engine.openTarget(key, target.kind === "video" ? file : await createImageBitmap(target.bitmap));
  if (scan.id !== id || target.file !== file) return false;
  opened = id;
  return true;
}

// a new photo or video (or none): nothing of the last one's preview is kept here
function resetPreview() {
  opened = null;
  preview?.reset();
  renderBadge();
}

// the badge on the canvas while the frame on screen is being swapped
function renderBadge() {
  if (!preview) return;
  const f = target.kind === "video" ? player.playhead : 0;
  const shown = !!target.file && (target.kind === "image" || !$("player").hidden);
  $("renderBadge").hidden = !shown || preview.original || !previewRuns() || preview.ready(f) || !preview.inRange(f) || !!preview.error;
}

// Outlines on the viewport the boxes of the faces of the selected segments in the shown frame, at
// the same width whatever the zoom.
function drawMarks() {
  const svg = $("marks");
  const m = viewport?.map();
  const f = target.kind === "video" ? player.playhead : 0;
  const i = scan.status === "done" && picked.size && m && !busy ? frameIndex(f) : -1;
  const rects = [];
  for (const [j, id] of (scan.faces[i] || []).entries()) {
    const p = scan.people.find((x) => x.ids.includes(id));
    if (!p || !picked.has(partAt(p, scan.times[i] ?? 0).id)) continue;
    const [x1, y1, x2, y2] = scan.boxes[i][j];
    const x = Math.round(m.x0 + m.k * x1), y = Math.round(m.y0 + m.k * y1);
    const w = Math.round(m.x0 + m.k * x2) - x, h = Math.round(m.y0 + m.k * y2) - y;
    rects.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" style="--fc:${p.color}"/><rect class="corner" x="${x - 1}" y="${y - 11}" width="8" height="8" transform="rotate(-6 ${x + 3} ${y - 7})"/>`);
  }
  svg.innerHTML = rects.join("");
}

// The swap lands in a puff of smoke: over each face just given a character, once the frame on
// screen shows it swapped (the preview's onChange). Once per giving, within a few seconds of it, and
// never with reduced motion. The cloud is circles in two layers (an inked edge under a paper fill),
// sized to the face, in box pixels like the marks.
let poofFor = null; // { parts: the segment ids given a character, t: when }
const PUFFS = [[-0.33, -0.08, 0.54], [-0.12, -0.29, 0.58], [0.17, -0.27, 0.52], [0.33, 0, 0.56], [0.19, 0.25, 0.54], [-0.1, 0.29, 0.56], [-0.29, 0.19, 0.48], [0, 0, 0.67]];
function poof() {
  const f = target.kind === "video" ? player.playhead : 0;
  if (!poofFor || !preview?.ready(f)) return;
  const { parts, t } = poofFor;
  poofFor = null;
  if (performance.now() - t > 8000 || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const m = viewport?.map(), i = scan.status === "done" && m ? frameIndex(f) : -1;
  for (const [j, id] of (scan.faces[i] || []).entries()) {
    const p = scan.people.find((x) => x.ids.includes(id));
    if (!p || !parts.has(partAt(p, scan.times[i] ?? 0).id)) continue;
    const [x1, y1, x2, y2] = scan.boxes[i][j];
    const d = Math.max(x2 - x1, y2 - y1) * m.k * 1.15;
    const layer = (grow) => PUFFS.map(([px, py, ps]) => {
      const s = ps * d + grow;
      return `<i style="width:${s}px;height:${s}px;margin:${-s / 2}px 0 0 ${-s / 2}px;--x:${px * d}px;--y:${py * d}px"></i>`;
    }).join("");
    const el = document.createElement("div");
    el.className = "poof";
    el.style.left = `${m.x0 + (m.k * (x1 + x2)) / 2}px`;
    el.style.top = `${m.y0 + (m.k * (y1 + y2)) / 2}px`;
    el.innerHTML = `<div class="ol">${layer(Math.max(3, d * 0.05))}</div><div class="fl">${layer(0)}</div>`;
    el.addEventListener("animationend", (e) => { if (e.target === el) el.remove(); });
    $("canvasBox").append(el);
  }
}

// the photo with its swapped faces (or as it is, while comparing)
function drawPhoto() {
  if (target.kind !== "image" || !target.bitmap || busy) return;
  const cv = $("view");
  if (cv.width !== target.width || cv.height !== target.height) { cv.width = target.width; cv.height = target.height; }
  const ctx = cv.getContext("2d");
  ctx.drawImage(target.bitmap, 0, 0);
  preview?.draw(ctx, 0);
}

function initPreview() {
  preview = new Preview(engine, {
    count: () => (target.kind === "video" ? player.total : target.kind === "image" ? 1 : 0),
    range: () => (target.kind === "video" ? [player.inPoint, player.outPoint] : [0, 0]),
    fps: () => player.fps,
    index: frameIndex,
    key: frameKey,
    opts: () => plan,
    prepare: openPreviewTarget,
    canRun: previewRuns,
    background: () => ui.auto,
    focus: () => ({ frame: target.kind === "video" ? player.playhead : 0, playing: player.playing }),
    onChange: (f) => {
      if (target.kind === "image") drawPhoto();
      else player.invalidate();
      if (f === (target.kind === "video" ? player.playhead : 0)) { renderBadge(); poof(); }
    },
  });
  player.overlay = { token: (f) => preview.token(f), draw: (ctx, f) => preview.draw(ctx, f) };
  player.canShow = (f) => !previewRuns() || !!preview.error || !preview.inRange(f) || preview.ready(f);
  timeline.madeState = (f) => preview.state(f);
  timeline.scanState = (f) => (scan.analysis ? scan.analysis.state[frameIndex(f)] : 0);
  // the playhead moved, playback started or stopped, or the in and out points moved: the frames
  // around the playhead come first, and the background waits while playing
  let at = -1, playing = false, range = "";
  player.onDraw(() => {
    const r = `${player.inPoint}:${player.outPoint}`;
    if (target.kind === "video") renderSelBar();
    if (player.playhead !== at) { drawMarks(); keepScanning(); }
    if (player.playhead === at && player.playing === playing && r === range) return;
    at = player.playhead;
    playing = player.playing;
    // (the in and out points are saved with the work, the output's Range row shows with them,
    // and the faces are found between them)
    if (r !== range) { saveSoon(); renderOutput(); if (range) keepScanning(); }
    range = r;
    preview.kick();
    renderBadge();
  });
}

// the steps' one-line summaries
function renderSteps() {
  const label = (items, id) => items.find((x) => x.id === id)?.label || "";
  const enh = one("enhance");
  const enhName = enh === undefined ? "Mixed" : enhancerLabel(enh);
  const sums = {
    swap: `${one("pixelshuffle") === undefined ? "Mixed" : pixelshuffleSet(one("pixelshuffle")).map((b) => label(DETAIL, b)).join(" / ")}`,
    enhance: enh === "" || shown("enhanceBlend").mixed ? enhName : `${enhName} · ${Math.round(100 * val("enhanceBlend"))}%`,
    mask: `Edges ${shown("blur").mixed ? "mixed" : val("blur").toFixed(2)}${!occluderInfo() ? "" : one("occlusion") === undefined ? " · objects mixed" : one("occlusion") ? (shown("maskSize").mixed ? " · coverage mixed" : val("maskSize") ? ` · coverage ${fmtMaskSize(val("maskSize"))}` : " · objects") : ""}`,
  };
  // while its model loads or when it failed, a card's head says so instead
  const failed = {};
  for (const [step, k] of [["swap", "swapper"], ["enhance", "enhancer"], ["mask", "occluder"]]) {
    const st = modelState[k];
    if (!ready || !modelInfo(k)) continue;
    if (st?.failed) { sums[step] = "Couldn't load"; failed[step] = st.failed; }
    else if (st) sums[step] = st.phase === "wait" ? "Waiting…" : `Loading ${Math.round(100 * st.frac)}%`;
  }
  for (const s of document.querySelectorAll(".step")) {
    const el = s.querySelector(".step-sum");
    el.textContent = sums[s.dataset.step] || "";
    el.classList.toggle("failed", !!failed[s.dataset.step]);
    el.title = failed[s.dataset.step] || "";
  }
}

// the timeline's tracks: one per person found, or the clip's progress while faces are being found
function renderTimeline() {
  if (target.kind !== "video") { timeline.setTracks([]); return; }
  if (scan.status === "running") {
    timeline.setTracks([{ id: "scan", label: "Scouting faces…", color: "#aba49f", segments: [[0, scan.frac * target.duration]] }]);
    return;
  }
  timeline.setTracks(scan.people.map((p) => {
    // (a deleted segment isn't one: nothing shows there)
    const parts = p.parts.map((q, i) => ({ id: q.id, t0: i ? q.t0 : 0, t1: p.parts[i + 1]?.t0 ?? Infinity, url: charByKey(q.s.source)?.url || "", name: charByKey(q.s.source)?.name || "" }))
      .filter((_, i) => !p.parts[i].off);
    return { id: p.id, label: p.label, color: p.color, thumb: p.thumb, segments: p.segments, parts };
  }));
  timeline.pick(picked, null, wholeIds());
}

// the video (or its swapped result) on the canvas in place of the image view
function showPlayer() {
  $("player").hidden = false;
  $("view").hidden = true;
}

// the character the settings' faces become (null: kept), or undefined when they differ
function currentChar() {
  const x = shown("source");
  return x.mixed ? undefined : charByKey(x.v);
}

// the characters: a card per character, a card per one being made, then New (which chooses the
// images to make one from). A click gives a character to the selected faces (every face, with none
// selected), and takes it off them again when they all have it; a drag gives it to a track on the
// timeline.
function renderChars() {
  const grid = $("charGrid");
  const cur = currentChar();
  const key = JSON.stringify([characters.map((c) => [c.key, c.name, c.url ? c.url.length + c.url.slice(-24) : "", c.loaded, c.faces.length]), making, busy, cur === undefined ? "mixed" : cur?.key ?? null]);
  if (grid.dataset.key === key) return;
  grid.dataset.key = key;
  grid.innerHTML = "";
  for (const c of characters) {
    const t = document.createElement("div");
    t.className = "char";
    t.dataset.key = c.key;
    t.draggable = !busy;
    t.innerHTML = `<button type="button" class="pick" role="radio"><img alt=""><span class="nm"></span></button><button type="button" class="x"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 7l10 10M17 7 7 17"/></svg></button>`;
    const pick = t.querySelector(".pick");
    const from = ` (${c.faces.length} image${c.faces.length > 1 ? "s" : ""})`;
    pick.setAttribute("aria-checked", String(cur === c));
    pick.setAttribute("aria-label", c.name);
    pick.title = cur === c ? `${c.name}: click to take it off, so these faces aren't swapped`
      : `${c.name}${from}: click to use, or drag onto a track`;
    pick.disabled = busy || !c.loaded;
    pick.onclick = () => assign(null, cur === c ? null : c.key);
    t.querySelector("img").src = c.url;
    t.querySelector(".nm").textContent = c.name;
    t.querySelector(".nm").title = `${c.name}: double-click to rename`;
    const x = t.querySelector(".x");
    x.title = `Remove ${c.name}`;
    x.setAttribute("aria-label", `Remove ${c.name}`);
    x.hidden = busy;
    x.onclick = () => removeCharacter(c);
    grid.append(t);
  }
  for (let i = 0; i < making; i++) {
    const t = document.createElement("div");
    t.className = "char making";
    t.innerHTML = `<div class="pick"><span class="ph"></span><span class="nm">Reading…</span></div>`;
    grid.append(t);
  }
  const add = document.createElement("button");
  add.type = "button";
  add.className = "char new";
  add.title = "New character: choose one image or several of the same person (or drop them here)";
  add.disabled = busy;
  add.innerHTML = `<span class="ph"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 6v12M6 12h12"/></svg></span><span class="nm">New</span>`;
  add.onclick = () => $("srcInput").click();
  grid.append(add);
  // in a narrow column, one row
  grid.style.setProperty("--cols", characters.length + making + 1);
}

// The faces of the selected photos on the timeline, in place of tracks: a chip per face (its face,
// its name and the character it becomes, with that character's picture), numbered by photo when
// several are selected. A click selects it. While a photo's are being found, or with none, a line
// says so.
function renderPeople() {
  const el = $("people");
  if (target.kind !== "image") {
    if (el.dataset.key) { el.innerHTML = ""; el.dataset.key = ""; }
    return;
  }
  people(); // (their keys)
  const cur = openPhotoOf();
  const sel = photos.filter((ph) => ph.sel || (ph === cur && !opening));
  const many = sel.length > 1;
  // what each chip shows. A photo opened again shows its faces as last known while they're
  // reloaded (not "Scouting faces…" for a moment), and each chip keeps its picture (faceCrop).
  const chips = [];
  let waiting = false;
  for (const ph of sel) {
    const open = ph === cur;
    const list = open ? (scan.status === "done" ? scan.people : ph.people || null) : ph.people || (ph.failed ? [] : null);
    if (!list) { waiting = true; continue; }
    for (const p of list) {
      const { src, mixed } = personSource(p);
      const on = p.parts.filter((q) => picked.has(q.id)).length;
      chips.push({ p, src, mixed, pressed: on === p.parts.length ? "true" : on ? "mixed" : "false", av: faceCrop(p, ph), n: photos.indexOf(ph) + 1 });
    }
  }
  const any = chips.length > 0;
  const note = !ready && !any ? "The faces are found once the models are loaded."
    : waiting ? "Scouting faces…" : !any ? (many ? "No faces found in these photos." : "No faces found in this photo.") : "";
  // (built again only when something on it changed: the row is rendered on every change)
  const tag = (url) => (url ? url.length + url.slice(-24) : "");
  const key = JSON.stringify([chips.map((c) => [pkey(c.p), c.p.label, c.src?.name ?? null, tag(c.src?.url), c.mixed, c.pressed, tag(c.av), c.n]), many, note]);
  if (el.dataset.key === key) return;
  el.dataset.key = key;
  el.innerHTML = "";
  for (const { p, src, mixed, pressed, av, n } of chips) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "face-card";
    b.dataset.id = pkey(p);
    b.setAttribute("aria-pressed", pressed);
    const where = many ? ` (photo ${n})` : "";
    b.title = `${p.label}${where}${mixed ? "" : src ? ` becomes ${src.name}` : " isn't swapped"}: click to select (Ctrl adds it, Shift selects a range), or drop a character on it`;
    b.innerHTML = `<span class="fwrap"><img class="av" alt="">${many ? `<span class="pn"></span>` : ""}</span><span class="txt"><span class="nm"></span><span class="as"></span></span>${src ? `<img class="to" alt="">` : ""}`;
    b.querySelector(".av").src = av;
    if (many) b.querySelector(".pn").textContent = n;
    b.querySelector(".nm").textContent = p.label;
    b.querySelector(".as").textContent = mixed ? "Several characters" : src ? `→ ${src.name}` : "Not swapped";
    if (src) b.querySelector(".to").src = src.url;
    const id = pkey(p);
    b.onclick = (e) => clickPerson(id, clickMode(e));
    el.append(b);
  }
  if (note) {
    const t = document.createElement("span");
    t.className = "ps-empty";
    t.textContent = note;
    el.append(t);
  }
}

// A face of a photo as its chip shows it: cut from the photo itself while it's open (sharper than
// the engine's thumbnail), once per face of each photo, and kept, so the chip's picture is the same
// when the photo is opened again, and while it opens. Until it's cut: the thumbnail.
const faceCrops = new Map(); // `${photo serial}:${person id}` -> url
function faceCrop(p, ph) {
  const k = `${ph?.serial}:${p.id}`;
  if (!faceCrops.has(k) && ph && ph === openPhotoOf() && scan.status === "done" && target.bitmap) {
    const j = (scan.faces[0] || []).findIndex((id) => p.ids.includes(id));
    const box = scan.boxes[0]?.[j];
    if (box) faceCrops.set(k, thumbUrl(target.bitmap, box, 96, 96));
  }
  return faceCrops.get(k) || p.thumb;
}

// Gives the character `key` (null: none) to segments: all of a person's (personId), or the ones the
// settings change.
function assign(personId, key) {
  if (busy) return;
  const p = findPerson(personId);
  const parts = p ? live(p) : scopeParts();
  const c = charByKey(key);
  edit(c ? `giving ${c.name}` : "taking the character off", () => { for (const q of parts) q.s.source = key; });
  poofFor = c ? { parts: new Set(parts.map((q) => q.id)), t: performance.now() } : null;
  renderTimeline();
  render();
}

// ------------------------------------------------------------------ renaming

const NAME_MAX = 40;

// Opens a text field over a name (el) to change it. Enter or leaving the field keeps it (trimmed and
// cut to NAME_MAX; an empty one keeps the old name), Escape cancels. The field floats over the page,
// so a render meanwhile doesn't take it away. done(name) gets a new name; back() runs after it
// closes (to give the keyboard focus back).
function renameField(el, name, done, back) {
  if (busy || document.querySelector(".rename")) return;
  const r = el.getBoundingClientRect();
  const f = document.createElement("input");
  f.className = "rename";
  f.value = name;
  f.maxLength = NAME_MAX;
  f.spellcheck = false;
  f.setAttribute("aria-label", `Rename ${name}`);
  // (a character's card is narrower than the usual field: it stays on the card, and its text scrolls)
  const card = el.closest(".char")?.getBoundingClientRect();
  const w = card ? card.width : Math.min(Math.max(r.width + 16, 140), innerWidth - 16);
  f.style.left = `${Math.max(8, Math.min(innerWidth - w - 8, card ? card.left : r.left - 5))}px`;
  f.style.top = `${r.top + r.height / 2 - 12}px`;
  f.style.width = `${w}px`;
  f.style.font = getComputedStyle(el).font;
  let open = true;
  const close = (keep) => {
    if (!open) return;
    open = false;
    removeEventListener("resize", leave);
    removeEventListener("scroll", leave, true);
    const v = f.value.trim().slice(0, NAME_MAX);
    f.remove();
    if (keep && v && v !== name) done(v);
    back?.();
  };
  // the name it's over moved
  const leave = (e) => { if (e.target !== f) close(true); };
  f.onkeydown = (e) => {
    if (e.key === "Enter") { e.preventDefault(); close(true); }
    else if (e.key === "Escape") { e.preventDefault(); close(false); }
  };
  f.onblur = () => close(true);
  addEventListener("resize", leave);
  addEventListener("scroll", leave, true);
  document.body.append(f);
  f.focus();
  f.select();
}

function renameCharacter(c, name) {
  if (busy || !characters.includes(c)) return;
  edit(`renaming ${c.name}`, () => { c.name = name; });
  saveCharacter(c);
  renderTimeline();
  render();
}

// a track, or a photo's face
function renamePerson(p, name) {
  if (busy || !people().includes(p)) return;
  edit(`renaming ${p.label}`, () => { p.label = name; });
  renderTimeline();
  render();
}

// The names that a double-click (or F2 on them) opens to rename: a character card's name (which
// doesn't pick the character, so the double-click doesn't) and a track's header, whose
// double-click's second click doesn't select again. (A photo's faces show their picture, and a
// name would go nowhere: they aren't named.)
function initRenames() {
  const charOf = (el) => el?.closest?.(".char[data-key] .nm") && charByKey(el.closest(".char").dataset.key);
  const personOf = (el) => {
    const card = el?.closest?.(".track[data-id] .who");
    return card && scan.status === "done" && findPerson(card.closest("[data-id]").dataset.id);
  };
  const rename = (el, key) => {
    const c = charOf(el);
    if (c) {
      const back = key && (() => $("charGrid").querySelector(`.char[data-key="${c.key}"] .pick`)?.focus());
      renameField(el.closest(".char").querySelector(".nm"), c.name, (v) => renameCharacter(c, v), back);
      return true;
    }
    const p = personOf(el);
    if (!p) return false;
    const sel = `.track[data-id="${pkey(p)}"] .who`;
    // the double-click's first click took it out of the selection (a click on the only one
    // selected goes back to every face): it's the one being named, so it's selected again
    if (!key && !p.parts.some((q) => picked.has(q.id))) selectPerson(p.id);
    const back = key && (() => document.querySelector(sel)?.focus());
    renameField(document.querySelector(`${sel} .nm`), p.label, (v) => renamePerson(p, v), back);
    return true;
  };
  for (const el of [$("charGrid"), $("tlTracks")]) {
    el.addEventListener("click", (e) => {
      if (charOf(e.target) || (e.detail > 1 && personOf(e.target))) e.stopPropagation();
    }, true);
    el.addEventListener("dblclick", (e) => { if (rename(e.target)) e.preventDefault(); });
  }
  addEventListener("keydown", (e) => {
    if (e.key !== "F2" || e.ctrlKey || e.metaKey || e.altKey || typing(e.target)) return;
    const el = document.activeElement;
    const nm = el?.matches?.(".char[data-key] .pick") ? el.querySelector(".nm") : el;
    if (nm && rename(nm, true)) e.preventDefault();
  });
}

function showImage(bitmapOrData, w, h) {
  const cv = $("view");
  // a resize clears and reallocates the canvas: skip it for the stream of video previews
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  const ctx = cv.getContext("2d");
  if (bitmapOrData instanceof ImageBitmap) { ctx.clearRect(0, 0, w, h); ctx.drawImage(bitmapOrData, 0, 0); }
  else ctx.putImageData(new ImageData(bitmapOrData, w, h), 0, 0);
  cv.hidden = false;
  player.stop();
  $("player").hidden = true;
}

function setInfo(html, { save = false } = {}) {
  $("result").hidden = !html;
  $("infoMeta").innerHTML = html || "";
  $("saveBtn").hidden = !save;
}

const baseName = (name) => name.replace(/\.[^.]+$/, "");

const OUTPUT_BASE_MAX = 60; // characters of the original's name kept in a saved file's
const OUTPUT_CHARACTER_MAX = 40; // and of the character's

// The start of `text`, at most `max` characters, without what's left dangling where it was cut.
function shortened(text, max) {
  const chars = Array.from(text);
  return chars.length <= max ? text : chars.slice(0, max).join("").replace(/[\s._\-[(]+$/, "");
}

// What the open photo or video (file) is saved as: its name and who was swapped in, the first
// character given and how many others ("clip - Ana +1.mp4"). The server keeps it from replacing a
// file already there.
function outputName(file, ext) {
  const base = shortened(baseName(file.name), OUTPUT_BASE_MAX);
  const given = [...new Set(scan.people.flatMap(live).map((q) => charByKey(q.s.source)).filter(Boolean))];
  if (!given.length) return `${base}-swapped${ext}`;
  // (a slash would read as a folder)
  const name = shortened(given[0].name.replace(/[/\\]/g, "-"), OUTPUT_CHARACTER_MAX);
  return `${base} - ${name}${given.length > 1 ? ` +${given.length - 1}` : ""}${ext}`;
}

// ------------------------------------------------------------------ inputs

// a note under the characters about the images just read (warn: some were left out)
function charsNote(text, warn = false) {
  const help = $("charsHelp");
  help.hidden = !text;
  help.textContent = text;
  help.className = "help" + (warn ? " warn" : "");
}

// Makes one character of images of the same person, dropped or chosen together: the mean of their
// largest faces' embeddings. Images without a face are left out.
async function makeCharacter(files) {
  if (busy) return;
  files = files.filter((f) => f.type.startsWith("image/"));
  if (!files.length) { showError("Characters are made from images."); return; }
  if (!ready) { showError("Characters can be made once the models are loaded."); return; }
  making++;
  charsNote("");
  render();
  const faces = [], skipped = [], several = [];
  try {
    for (const file of files) {
      const name = baseName(file.name);
      let bitmap;
      try {
        bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
      } catch {
        skipped.push(`${name} couldn't be opened`);
        continue;
      }
      let r;
      try {
        r = await engine.embedImage(await createImageBitmap(bitmap));
      } catch (e) {
        bitmap.close();
        if (gpuGone(e)) { charsNote("The graphics card stopped. Drop the images again once the models are loaded again.", true); return; }
        console.error(e);
        skipped.push(`${name}: ${e.message}`);
        continue;
      }
      if (!r.face) { bitmap.close(); skipped.push(`no face in ${name}`); continue; }
      if (r.count > 1) several.push(name);
      faces.push({ bitmap, box: r.face.box, embed: r.face.embedding });
    }
    const notes = [...skipped];
    if (several.length) notes.push(`${several.length > 1 ? `${several.length} images have` : `${several[0]} has`} more than one face: the largest is used`);
    const note = notes.length ? `${notes.join("; ")}.` : "";
    if (!faces.length) { charsNote(`${note} Try a clearer, front-facing image.`, true); return; }
    charsNote(note, skipped.length > 0);
    await createCharacter(faces);
  } finally {
    for (const f of faces) f.bitmap.close();
    making--;
    render();
  }
}

// Makes a character of faces ([{ bitmap, box, embed }]): their mean embedding, pictured by the
// ones closest to it.
async function createCharacter(faces) {
  const sum = new Float32Array(faces[0].embed.length);
  for (const f of faces) for (let k = 0; k < sum.length; k++) sum[k] += f.embed[k];
  const embed = normalize(sum);
  const near = (f) => f.embed.reduce((a, v, k) => a + v * embed[k], 0);
  const ranked = faces.map((f) => [near(f), f]).sort((a, b) => b[0] - a[0]).map(([, f]) => f);
  const n = nextChar++;
  const c = {
    key: `c${n}`, name: `Character ${n}`, embed, url: facesUrl(ranked),
    faces: ranked.map((f) => ({ url: thumbUrl(f.bitmap, f.box, 96, 128), embed: f.embed })), loaded: false,
  };
  const step = edit("new character", () => characters.push(c));
  render();
  try {
    await loadCharacter(c);
  } catch (e) {
    console.error(e);
    showError(`${c.name} couldn't be made: ${e.message}`);
    edits.drop(step);
    removeCharacter(c, { undoable: false });
    renderHistory();
    return;
  }
  // undone meanwhile
  if (!characters.includes(c)) { dropCharacter(c); return; }
  saveCharacter(c);
  render();
}

const saveCharacter = (c) => saved.put("characters", { key: c.key, name: c.name, embed: c.embed, url: c.url, faces: c.faces });

async function loadCharacter(c) {
  await engine.setCharacter(c.key, c.embed);
  c.loaded = true;
  epoch++;
}

// The engine forgets a character the page no longer has (an undo took it out while it loaded).
function dropCharacter(c) {
  if (!charByKey(c.key) && ready) engine.removeCharacter(c.key);
  c.loaded = false;
  render();
}

// Removes a character: the faces it was given are kept.
function removeCharacter(c, { undoable = true } = {}) {
  if (busy) return;
  const i = characters.indexOf(c);
  if (i < 0) return;
  const remove = () => {
    characters.splice(i, 1);
    for (const p of [...scan.people, ...photos.flatMap((ph) => ph.people || [])]) for (const q of p.parts) if (q.s.source === c.key) q.s.source = null;
  };
  if (undoable) edit(`removing ${c.name}`, remove);
  else remove();
  saved.remove("characters", c.key);
  if (ready && c.loaded) engine.removeCharacter(c.key);
  epoch++;
  renderTimeline();
  render();
}

// ------------------------------------------------------------------ undo

// What an undo step brings back: the people with their segments (also of the other selected photos),
// the characters and the in and out points, of the open photo or video (doc: which one, and whether its faces are found). Each person's
// fields are copied, and their parts with their settings. What's only ever replaced, never changed
// in place, is shared: a person's segments, a character's embedding, picture and faces.
const copyPerson = (p) => ({ ...p, parts: p.parts.map((q) => ({ ...q, s: { ...q.s } })) });
const docOf = () => `${scan.id}:${scan.status}`;
function snapshot() {
  return {
    doc: docOf(),
    people: scan.people.map(copyPerson),
    others: otherPhotos().map((ph) => [ph, ph.people.map(copyPerson)]),
    gone: new Set(scan.gone),
    characters: characters.map(({ loaded, ...c }) => c),
    range: target.kind === "video" && player.range ? { ...player.range } : null,
  };
}
// a snapshot as text, to compare (without the embeddings, nor where the people are on screen:
// the scan finds more of that as it goes)
const noEmbeds = (k, v) => (ArrayBuffer.isView(v) ? undefined : v);
const snapKey = (s) => JSON.stringify([s.people.map(({ segments, frames, thumb, ...p }) => p), s.others.map(([ph, ps]) => [ph.serial, ps.map(({ segments, frames, thumb, ...p }) => p)]),
  [...s.gone], s.characters, s.range], noEmbeds);

let openStep = null; // a step still being made (a slider or marker drag): { label, before }
let editing = 0;
let quiet = 0; // changes that aren't edits (an undo putting the in and out points back)

// The edits go through here: edit(label, fn) runs fn, and when it changed what a snapshot holds,
// that's one undo step, named by label ("Undo split"). With coalesce, the edits with the same label
// that follow are one step until the pointer or the key is let go (or a slider's change); a
// different edit ends it first. Edits made inside fn are part of its step. -> the step (for
// edits.drop), or null
function edit(label, fn, { coalesce = false } = {}) {
  if (editing) { fn(); return null; }
  if (openStep && (!coalesce || openStep.label !== label)) endEdit();
  const before = openStep?.before ?? snapshot();
  editing++;
  try {
    fn();
  } finally {
    editing--;
  }
  if (coalesce) { openStep ??= { label, before }; return null; }
  return record(label, before);
}

// Ends a step of coalesced edits.
function endEdit() {
  if (!openStep) return;
  const { label, before } = openStep;
  openStep = null;
  record(label, before);
}

function record(label, before) {
  if (before.doc !== docOf() || snapKey(before) === snapKey(snapshot())) return null;
  if (edits.doc !== before.doc) { edits.clear(); edits.doc = before.doc; }
  const step = edits.push(label, before);
  renderHistory();
  return step;
}

// Whether the photo or video open (with the photos selected with it) has edits a reset would undo:
// a face given a character, a setting changed, a track renamed or deleted, the in and out points.
function hasEdits() {
  if (!target.file || scan.status !== "done") return false;
  const plain = (p) => p.label === `Face ${p.id + 1}` && p.parts.every((q) => !q.off && !charByKey(q.s.source) && SEGMENT.every((k) => same(q.s[k], SEGMENT_DEFAULTS[k])));
  const ranged = target.kind === "video" && !!player.range && (player.range.a > 0 || player.range.b < player.total - 1);
  return scan.gone.size > 0 || ranged || ![...scan.people, ...otherPhotos().flatMap((ph) => ph.people)].every(plain);
}

// Puts the photo or video open (and the photos selected with it) back as it was first opened: no
// characters given, every setting (and the next segments') as it starts, the tracks as found, the
// whole video in range. The faces found stay; one undo step brings the edits back.
function resetEdits() {
  if (busy || !target.file || scan.status !== "done") return;
  endEdit();
  Object.assign(ui, SEGMENT_DEFAULTS);
  saveUi();
  const fresh = (p) => person({ ...p, label: null, parts: [{ t0: 0, s: {} }] });
  edit("reset", () => {
    if (target.kind === "video") {
      scan.people = [];
      scan.gone = new Set();
      refreshTracks();
      if (player.total) player.setRange(0, player.total - 1);
    } else {
      scan.people = scan.people.map(fresh);
      for (const ph of otherPhotos()) ph.people = ph.people.map(fresh);
    }
  });
  picked = new Set();
  wholeTracks = new Set();
  lastTrack = lastPart = null;
  showResumed(false);
  renderTimeline();
  render();
}

// The note offering a reset, when a photo or video opens with the edits it had before: for a
// while, until closed, or until another one opens.
let resumedTimer = 0;
function showResumed(on) {
  clearTimeout(resumedTimer);
  $("resumed").hidden = !on;
  if (!on) return;
  $("resumedText").textContent = `Your edits to this ${target.kind === "video" ? "video" : "photo"} are back`;
  resumedTimer = setTimeout(() => showResumed(false), 12000);
}

// Runs fn without making its changes an undo step.
function untracked(fn) {
  quiet++;
  try {
    return fn();
  } finally {
    quiet--;
  }
}

// the history is of the photo or video open now, with its faces as they were found: another one,
// or the faces found again, start a new one
function currentEdits() {
  if (edits.doc !== docOf()) { edits.clear(); edits.doc = docOf(); }
  return edits;
}

function undo(back = true) {
  if (busy) return;
  endEdit();
  const h = currentEdits();
  const snap = back ? h.undo(snapshot()) : h.redo(snapshot());
  if (snap) restore(snap);
  renderHistory();
}

// Puts a snapshot's state back.
function restore(snap) {
  // the characters, by key: the ones it doesn't have go, the ones it has come back
  const keep = new Set(snap.characters.map((c) => c.key));
  let changed = false;
  for (const c of characters) {
    if (keep.has(c.key)) continue;
    saved.remove("characters", c.key);
    if (ready && c.loaded) engine.removeCharacter(c.key);
    changed = true;
  }
  const back = [];
  const list = snap.characters.map((s) => {
    const c = charByKey(s.key);
    if (!c) {
      changed = true;
      const n = { ...s, loaded: false };
      saved.put("characters", s);
      back.push(n);
      return n;
    }
    // renamed
    const { loaded, ...now } = c;
    if (JSON.stringify(now, noEmbeds) !== JSON.stringify(s, noEmbeds)) { Object.assign(c, s); saved.put("characters", s); }
    return c;
  });
  characters.splice(0, characters.length, ...list);
  for (const c of back) {
    // without the models, they go to the engine once it has them (loadModels)
    if (ready) loadCharacter(c).then(() => { if (!characters.includes(c)) dropCharacter(c); else render(); }, (e) => { console.error(e); showError(`${c.name} couldn't be brought back: ${e.message}`); });
  }
  if (changed) epoch++;
  // the people, with their ids: the selection and the timeline keep them
  scan.people = snap.people.map(copyPerson);
  for (const [ph, ps] of snap.others) ph.people = ps.map(copyPerson);
  scan.gone = new Set(snap.gone);
  refreshTracks();
  const parts = new Set(people().flatMap((p) => p.parts.map((q) => q.id)));
  picked = new Set([...picked].filter((id) => parts.has(id)));
  if (target.kind === "video" && player.total) untracked(() => (snap.range ? player.setRange(snap.range.a, snap.range.b) : player.setRange(0, player.total - 1)));
  renderTimeline();
  selectionChanged();
}

// the undo and redo buttons, with what they'd undo or redo
function renderHistory() {
  const h = currentEdits();
  const mac = /Mac|iP/.test(navigator.platform);
  const key = (k) => (mac ? `⌘${k}` : `Ctrl+${k}`);
  $("undoBtn").disabled = busy || !h.canUndo;
  $("undoBtn").title = h.canUndo ? `Undo ${h.undoLabel} (${key("Z")})` : `Undo (${key("Z")})`;
  $("redoBtn").disabled = busy || !h.canRedo;
  $("redoBtn").title = h.canRedo ? `Redo ${h.redoLabel} (${mac ? "⇧⌘Z" : "Ctrl+Y"})` : `Redo (${mac ? "⇧⌘Z" : "Ctrl+Y"})`;
}

// a field the browser's own undo works in (a slider isn't one)
const textField = (el) => el instanceof HTMLElement && (el.isContentEditable || el.tagName === "TEXTAREA"
  || (el.tagName === "INPUT" && !/^(range|checkbox|radio|button|submit|reset|file|color)$/.test(el.type)));

// ------------------------------------------------------------------ saved work

// a person found, from what analyze() or the saved work has of them; parts: [{ t0, s }]. ids: the
// people of the analysis their track covers (more than one once tracks are merged). The default
// name and the colour go by the analysis's numbering, so a merge doesn't change the others'.
function person({ id, ids, label, thumb, segments, frames, parts }) {
  const fix = (s) => ({ ...segmentDefaults(), ...s, enhance: "enhance" in s ? enhanceOf(s.enhance) : ui.enhance, source: charByKey(s.source) ? s.source : null });
  return { id, ids: ids || [id], label: label || `Face ${id + 1}`, color: COLORS[id % COLORS.length], thumb, segments, frames, parts: parts.map(({ t0, s, off, auto }) => Object.assign(newPart(t0, fix(s)), off && { off }, auto && { auto })) };
}

// the people as they're saved: what they are, their names and their segments, without the page's ids
const savedPeopleOf = (list) => list.map(({ id, ids, label, thumb, segments, frames, parts }) => ({ id, ids, label, thumb, segments, frames, parts: parts.map(({ t0, s, off, auto }) => ({ t0, s, ...(off && { off }), ...(auto && { auto }) })) }));
const savedPeople = () => savedPeopleOf(scan.people);
// the in and out points as they're saved: frames (the same for the same file), or null
const savedRange = () => (target.kind === "video" && player.range ? { ...player.range } : null);
// the edits as they're saved: the people, the analysis's people whose tracks were deleted, and the
// in and out points
const savedWork = () => ({ people: savedPeople(), gone: [...scan.gone], range: savedRange() });

const ANALYSIS_SAVE = 3000; // ms between saves of a video's faces while they're being found
let lastSaved = ""; // the target's work as last saved (JSON)
let saveTimer = 0;
let analysisTimer = 0;

// Saves the target's segments and settings once the edits pause (render() calls it after any), and
// its in and out points (the preview's onDraw notices those).
function saveSoon() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveNow, 400);
}

function saveNow() {
  clearTimeout(saveTimer);
  if (target.kind === "image") saveOtherPhotos();
  if (analysisTimer) { saveAnalysis(true); return; }
  if (!target.hash || scan.status !== "done") return;
  const work = savedWork();
  const json = JSON.stringify(work);
  if (json === lastSaved) return;
  lastSaved = json;
  saved.putTarget(target.hash, work);
}

// Saves the faces and segments of the photos that aren't open (edited with them selected), where
// they changed.
function saveOtherPhotos() {
  const cur = openPhotoOf();
  for (const ph of photos) {
    if (ph === cur || !ph.people || !ph.hash || !ph.analysis) continue;
    const work = { people: savedPeopleOf(ph.people), gone: [], range: null };
    const json = JSON.stringify(work);
    if (json === ph.saved) continue;
    ph.saved = json;
    saved.putTarget(ph.hash, { name: ph.file.name, kind: "image", analysis: ph.analysis, ...work });
  }
}

// Saves the faces found so far of the video (or, only now, the photo), with the work: every few
// seconds while they're being found, or now.
function saveAnalysis(now = false) {
  if (!now) {
    const id = scan.id;
    analysisTimer ||= setTimeout(() => { analysisTimer = 0; if (scan.id === id) saveAnalysis(true); }, ANALYSIS_SAVE);
    return;
  }
  clearTimeout(analysisTimer);
  analysisTimer = 0;
  const analysis = scan.analysis || scan.imageAnalysis;
  if (!target.hash || !analysis || scan.status !== "done") return;
  const work = savedWork();
  lastSaved = JSON.stringify(work);
  saved.putTarget(target.hash, { name: target.file.name, kind: target.kind, analysis, ...work });
}

// Takes where each person is from an analysis (session.js analyze): the preview's frames. (A
// video's frames not looked at yet have no one.)
function takeFaces(analysis) {
  const t0 = analysis.t0 || 0;
  scan.times = analysis.times.map((t) => t - t0);
  scan.faces = analysis.faces.map((fs) => (fs || []).filter((f) => f.person >= 0).map((f) => f.person));
  scan.boxes = analysis.faces.map((fs) => (fs || []).filter((f) => f.person >= 0).map((f) => f.box));
}

// Takes the faces found in the target before, with their segments, as analysis `id`. -> whether
// there were any
async function restoreFaces(id) {
  // (a photo already among the open ones has been on screen: its edits aren't from an earlier visit)
  const earlierVisit = !openPhotoOf()?.people;
  const s = await saved.getTarget(target.hash);
  // (a video's faces as earlier versions found them, all at once, are found again)
  if (scan.id !== id || !s?.analysis || (s.analysis.kind === "video" && !s.analysis.state)) return false;
  try {
    await engine.restoreAnalysis(id, s.analysis);
  } catch (e) {
    console.warn("the saved faces couldn't be used", e);
    return false;
  }
  if (scan.id !== id) return true;
  scan.people = s.people.map(person);
  scan.gone = new Set(s.gone);
  scan.frames = s.analysis.faces.length;
  takeFaces(s.analysis);
  if (s.analysis.kind === "video") scan.analysis = s.analysis;
  else scan.imageAnalysis = s.analysis;
  scan.status = "done";
  refreshTracks();
  // (the in and out points stay as they are: after a switch of Recognize, the next save gives
  // them to this record too)
  lastSaved = JSON.stringify({ ...savedWork(), range: s.range ?? null });
  if (!scan.people.length && target.kind === "image") showError("No faces found in this photo.");
  renderTimeline();
  render();
  keepScanning(true);
  // (an export opens photos itself, or may have started while this one opened: not the moment)
  if (earlierVisit && !busy && !saving && !working && hasEdits()) showResumed(true);
  return true;
}

// The characters saved on earlier visits; the engine gets them once the models are loaded. (Earlier
// versions kept the images characters were made from, and characters without their faces: both
// are forgotten.)
async function restoreLibrary() {
  const chars = await saved.all("characters");
  const num = (k) => +k.slice(1);
  for (const r of chars.sort((a, b) => num(a.key) - num(b.key))) {
    nextChar = Math.max(nextChar, num(r.key) + 1);
    if (!r.faces) { saved.remove("characters", r.key); continue; }
    characters.push({ ...r, loaded: false });
  }
  render();
}

// Finds the target's faces and tells its people apart, as soon as there's a photo or video: a
// video's around the playhead first (scanAround). No face is swapped until it's given a character.
async function findFaces() {
  showResumed(false);
  resetScan(scan.id + 1);
  const id = scan.id;
  resetPreview();
  picked = new Set();
  wholeTracks = new Set();
  lastTrack = lastPart = null;
  renderTimeline();
  render();
  // found before: no need for the models
  if (target.hash && engine && !busy && await restoreFaces(id)) return;
  if (scan.id !== id || !ready || !target.file || busy) return;
  if (target.kind === "video") { firstScan(); return; }
  scan.status = "running";
  render();
  const ac = new AbortController();
  scan.ac = ac;
  try {
    const r = await engine.analyze(id, await createImageBitmap(target.bitmap), { recognize: true }, { signal: ac.signal });
    if (scan.id !== id) { for (const p of r.people) p.thumb?.close(); return; }
    scan.people = r.people.map((p) => {
      const thumb = p.thumb ? thumbUrl(p.thumb) : "";
      p.thumb?.close();
      return person({ ...p, thumb, parts: [{ t0: 0, s: segmentDefaults() }] });
    });
    scan.frames = r.frames;
    takeFaces(r.analysis);
    scan.imageAnalysis = r.analysis;
    scan.status = "done";
    if (target.hash) {
      const work = savedWork();
      lastSaved = JSON.stringify(work);
      saved.putTarget(target.hash, { name: target.file.name, kind: target.kind, analysis: r.analysis, ...work });
    }
    if (!scan.people.length) showError("No faces found in this photo.");
  } catch (e) {
    if (scan.id !== id) return;
    scan.status = "";
    if (e.name !== "AbortError" && !gpuGone(e)) { console.error(e); showError("The faces couldn't be found: " + e.message); }
  } finally {
    if (scan.id === id) scan.ac = null;
  }
  if (scan.id !== id) return;
  renderTimeline();
  render();
}

// a fresh scan for another photo or video (or none), as scan `id`
function resetScan(id) {
  scan.ac?.abort();
  Object.assign(scan, { id, ...freshScan() });
}

// ------------------------------------------------------------------ finding a video's faces

const PLAY_LOOKAHEAD = 2; // seconds ahead of the playhead, while playing, that are to be looked at already
const SCAN_SETTLE = 300; // ms the playhead stays put on a frame not looked at before it's scanned there (at once while playing)
const SCAN_DRAW = 250; // ms between redraws of the faces found

// the video's frames between the in and out points, the only ones its faces are found in:
// [first, last] seconds from its first frame (the frames starting then), and the point in them a
// scan's window is around (the playhead, or the in or out point nearest it)
const scanRange = () => [player.time(player.inPoint), player.time(player.outPoint)];
const scanPoint = () => player.time(Math.min(Math.max(player.playhead, player.inPoint), player.outPoint));
// whether every frame between the in and out points has been looked at
const rangeScanned = () => scan.analysis?.state.subarray(frameIndex(player.inPoint), frameIndex(player.outPoint) + 1).every((x) => x === 2);

// Finds the faces of the video's window around t seconds, between the in and out points
// (scanner.js), taking them as they come. The scan going on stops (what it found stays).
async function scanAround(t) {
  if (!engine || !ready || busy || target.kind !== "video" || !target.file) return;
  scan.ac?.abort();
  const id = scan.id, ac = new AbortController(), range = scanRange();
  Object.assign(scan, { ac, scanning: true, center: t, span: range.join(":"), coverage: 0, finished: false });
  renderScanStatus();
  let finished = false;
  try {
    await engine.analyze(id, target.file, { recognize: true, around: t, range, whole: ui.auto }, {
      signal: ac.signal,
      // (a scan cut short sends what it found too)
      onProgress: (p) => {
        if (scan.id !== id) { for (const [, b] of p.scan?.thumbs || []) b.close(); return; }
        if (p.scan) takeScan(p.scan);
        if (scan.ac === ac) { scan.coverage = p.coverage; renderScanStatus(); }
      },
    });
    finished = true;
  } catch (e) {
    if (scan.id !== id) return;
    if (e.name !== "AbortError" && !gpuGone(e)) { console.error(e); showError("The faces couldn't be found: " + e.message); }
  }
  if (scan.id !== id || scan.ac !== ac) return;
  Object.assign(scan, { ac: null, scanning: false, finished });
  if (scan.status === "running") scan.status = ""; // (nothing came: found again once it can be)
  refreshTracks();
  saveAnalysis(true);
  if (finished && !scan.people.length && rangeScanned()) showError(`No faces found ${player.range ? "between the in and out points" : "in this video"}.`);
  renderTimeline();
  render();
}

// A video's first scan, around the playhead (its faces weren't found before).
function firstScan() {
  if (!ready || busy || target.kind !== "video" || !target.file || scan.status === "done" || scan.ac) return;
  scan.status = "running";
  render();
  scanAround(scanPoint());
}

// Stops the scan going on (before an export, or while the models load).
function stopScan() {
  scan.ac?.abort();
  Object.assign(scan, { ac: null, scanning: false });
  renderScanStatus();
}

let scanTimer = 0;
// Scans around the playhead once it stays put on a frame not looked at yet (at once while
// playing: playback has gone past what's found), when the in or out point moved, or when the last
// scan was cut short: at once with now. A scan finding the faces near the playhead already
// (NEAR_SECONDS from where it started) goes on.
function keepScanning(now = false) {
  clearTimeout(scanTimer);
  scanTimer = setTimeout(() => {
    const a = scan.analysis;
    if (!a || scan.status !== "done" || !ready || busy || target.kind !== "video" || !player.media) return;
    const t = scanPoint(), moved = scan.span !== scanRange().join(":");
    // (while playing, the frames a little ahead too: the scan gets there before playback)
    const soon = player.playing ? Math.min(scanRange()[1], t + PLAY_LOOKAHEAD) : t;
    const there = a.state[nearest(scan.times, t)] === 2 && a.state[nearest(scan.times, soon)] === 2;
    const near = scan.scanning && t >= scan.center - NEAR_SECONDS.behind && t < scan.center + NEAR_SECONDS.ahead;
    if (scan.scanning ? moved || (!there && !near) : moved || !there || !scan.finished) scanAround(t);
  }, now || player.playing ? 0 : SCAN_SETTLE);
}

// Turns Auto on or off: the scan going on starts again the new way (what it found stays), and the
// background swap starts or stops.
function setAuto(on) {
  ui.auto = on;
  saveUi();
  renderAuto();
  if (target.kind === "video" && scan.status === "done" && (scan.scanning || on)) scanAround(scanPoint());
  preview?.kick();
}

function renderAuto() {
  const b = $("tlAuto");
  b.setAttribute("aria-pressed", String(ui.auto));
  b.title = ui.auto
    ? "Auto is on (W): every face near the playhead is found and the video is swapped ahead in the background, so the export has less left to do. Click to only work around the playhead."
    : "Auto is off (W): only the frames around the playhead are found and swapped, with a quick look over the whole video, to keep the load low. Click to work ahead in the background.";
  b.setAttribute("aria-label", "Auto: work ahead in the background");
}

// Takes an update of the video's faces (scanner.js update()) into the analysis mirrored here.
function takeScan(u) {
  if (u.open) {
    const n = u.open.times.length;
    scan.analysis = { ...u.open, state: new Uint8Array(n), faces: new Array(n).fill(null), people: [] };
    takeFaces(scan.analysis);
    scan.frames = n;
    scan.status = "done";
  }
  const a = scan.analysis;
  if (!a) { for (const [, b] of u.thumbs) b.close(); return; }
  for (const [i, st, fs] of u.frames) {
    a.state[i] = st;
    a.faces[i] = fs;
    const on = (fs || []).filter((f) => f.person >= 0);
    scan.faces[i] = on.map((f) => f.person);
    scan.boxes[i] = on.map((f) => f.box);
  }
  for (const [p, sum, n, into] of u.people) a.people[p] = into == null ? { sum, n } : { sum, n, into };
  for (const [p, b] of u.thumbs) { scan.thumbs.set(p, thumbUrl(b)); b.close(); }
  saveAnalysis(!!u.open);
  if (u.open) { refreshTracks(); renderTimeline(); render(); return; }
  scanChanged();
}

// The tracks again, once in a while, as the faces found change: where the people are on screen,
// and a track for each new one.
let scanDraw = 0;
function scanChanged() {
  if (scanDraw) return;
  const id = scan.id;
  scanDraw = setTimeout(() => {
    scanDraw = 0;
    if (scan.id !== id) return;
    const shape = () => JSON.stringify(scan.people.map((p) => p.ids));
    const before = shape();
    refreshTracks();
    renderTimeline();
    preview?.invalidate();
    drawMarks();
    // (a new track: what's swapped and the settings' scope change)
    if (shape() !== before) render();
    else renderBadge();
  }, SCAN_DRAW);
}

// seconds the tracks bridge between the keyframes the loose pass looks at
const spans = new WeakMap();
function looseSpan(a) {
  if (!spans.has(a)) {
    const d = a.keys.slice(1).map((k, j) => a.times[k] - a.times[a.keys[j]]).sort((x, y) => x - y);
    spans.set(a, Math.min(15, Math.max(2, 1.5 * (d[d.length >> 1] ?? 1))));
  }
  return spans.get(a);
}

// A video's tracks from the faces found so far: tracks whose people the scan found to be one
// (scanner.js regroup) as one, each one's stretches on screen (its analysis people's), a new track
// for a person found (seen in enough frames, or by the loose pass) on none and not deleted, and no
// track for one the scan took for someone else since (a track the user did something with stays).
function refreshTracks() {
  const a = scan.analysis;
  if (!a) return;
  const root = (id) => { while (a.people[id]?.into != null) id = a.people[id].into; return id; };
  for (let k = 0; k < scan.people.length; k++) {
    const top = scan.people[k], roots = new Set(top.ids.map(root));
    const same = scan.people.filter((p, j) => j > k && p.ids.some((id) => roots.has(root(id))));
    if (!same.length) continue;
    const who = [top, ...same];
    const merged = mergedTrack(who, top, { coalesce: true }) || { ...top, ids: who.flatMap((p) => p.ids) };
    scan.people = scan.people.filter((p) => !same.includes(p)).map((p) => (p === top ? merged : p));
    k--; // (its people may be one with another track's too)
  }
  const st = stretches(a, { maxGap: GAP, span: looseSpan(a) });
  const track = new Map(); // who an analysis person is (root) -> their track
  for (const p of scan.people) for (const id of p.ids) track.set(root(id), p);
  // the people found since: on the track of who they are, else on a new one
  const fresh = new Map(); // root -> [ids]
  for (const id of st.keys()) {
    const r = root(id), t = track.get(r);
    if (scan.gone.has(id) || scan.gone.has(r) || t?.ids.includes(id)) continue;
    if (t) t.ids = [...t.ids, id];
    else fresh.set(r, [...(fresh.get(r) || []), id]);
  }
  const at = (i) => Math.max(0, (i < a.times.length ? a.times[i] : a.duration) - a.t0);
  const shown = (ids) => ids.reduce((n, id) => n + (st.get(id)?.frames || 0), 0) >= MIN_FRAMES || ids.some((id) => st.get(id)?.loose);
  const segmentsOf = (ids) => {
    const runs = ids.flatMap((id) => st.get(id)?.runs || []).sort((x, y) => x[0] - y[0]);
    const out = [];
    for (const [i, j] of runs) {
      // (a person's runs bridge GAP frames already: so do the runs of the people one track has)
      const last = out[out.length - 1];
      if (last && i <= last[1] + GAP) last[1] = Math.max(last[1], j);
      else out.push([i, j]);
    }
    return out.map(([i, j]) => [at(i), at(j)]);
  };
  const touched = (p) => p.label !== `Face ${p.id + 1}` || p.parts.filter((q) => !q.auto).length > 1 || p.parts.some((q) => q.s.source || q.off);
  const thumbOf = (ids) => ids.map((id) => scan.thumbs.get(id)).find(Boolean);
  const people = [];
  for (const p of scan.people) {
    if (!shown(p.ids) && !touched(p)) continue;
    p.segments = segmentsOf(p.ids);
    cutAtGaps(p);
    p.frames = p.ids.reduce((n, id) => n + (st.get(id)?.frames || 0), 0);
    p.thumb = thumbOf(p.ids) || p.thumb;
    people.push(p);
  }
  for (const [r, ids] of fresh) {
    if (!shown(ids)) continue;
    const np = person({ id: r, ids: [...new Set([r, ...ids])], thumb: thumbOf(ids) || "", segments: segmentsOf(ids), frames: ids.reduce((n, id) => n + st.get(id).frames, 0), parts: [{ t0: 0, s: segmentDefaults() }] });
    cutAtGaps(np);
    people.push(np);
  }
  scan.people = people;
}

// whether t (seconds) is where a person comes back on screen: in a gap between two of their
// stretches, or at the start of the later one
const atGap = (p, t) => p.segments.some(([a], k) => k > 0 && t > p.segments[k - 1][1] - EPS && t <= a + EPS);

// Cuts a track where its person comes back after being off screen, so each stretch on screen is a
// segment of its own (a click selects it alone), with the settings it had. Such a cut (auto) goes
// again once the stretches meet (the scan found the person in between), when its settings are
// still the ones before it.
function cutAtGaps(p) {
  const same = (x, y) => JSON.stringify([x.s, !!x.off]) === JSON.stringify([y.s, !!y.off]);
  const parts = [];
  for (const q of p.parts) {
    const prev = parts[parts.length - 1];
    if (q.auto && prev && !atGap(p, q.t0) && same(q, prev)) continue;
    parts.push(q);
  }
  for (const [k, [a]] of p.segments.entries()) {
    if (!k || parts.some((q, i) => i > 0 && q.t0 > p.segments[k - 1][1] - EPS && q.t0 <= a + EPS)) continue;
    let at = 0;
    for (const [i, q] of parts.entries()) if (q.t0 <= a + EPS) at = i;
    const q = parts[at];
    parts.splice(at + 1, 0, Object.assign(newPart(a, q.s), { auto: true }, q.off && { off: true }));
  }
  p.parts = parts;
}

// how far the scan around the playhead is: under the clip's name, and in Export's tooltip
function renderScanStatus() {
  const on = !busy && scan.scanning && target.kind === "video";
  $("scanNote").hidden = !on;
  if (on) $("scanText").textContent = `Scouting faces · ${Math.round(100 * scan.coverage)}%`;
  if (!on || !$("goBtn").disabled) return;
  if (scan.status !== "done") $("goBtn").title = `Scouting faces… ${Math.round(100 * scan.coverage)}%`;
}

// The Export button's progress: frac 0–1 (null: not known), its line, and the details in its tooltip
function setProgress(frac, text, details = "") {
  const run = $("goRun");
  run.style.setProperty("--p", frac == null ? "0%" : `${(100 * frac).toFixed(1)}%`);
  if (frac == null) run.removeAttribute("aria-valuenow");
  else run.setAttribute("aria-valuenow", String(Math.round(100 * frac)));
  $("progressText").textContent = text;
  run.title = details || text;
}

async function loadTarget(file) {
  if (!file || busy) return;
  const kind = file.type.startsWith("video/") ? "video" : file.type.startsWith("image/") ? "image" : "";
  if (!kind) { showError("Choose a photo or a video."); return; }
  saveNow();
  const hashing = saved.fileHash(file).catch((e) => { console.warn("the file couldn't be hashed", e); return null; });
  target.bitmap?.close();
  Object.assign(target, { file, kind, hash: null, bitmap: null, width: 0, height: 0, duration: 0 });
  resetScan(scan.id + 1);
  resetPreview();
  clearResult();
  try {
    if (kind === "image") {
      target.bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
      target.width = target.bitmap.width;
      target.height = target.bitmap.height;
      player.close();
      showImage(target.bitmap, target.width, target.height);
      drawPhoto();
    } else {
      const v = await player.open(file);
      Object.assign(target, { width: v.width, height: v.height, duration: v.duration });
      showPlayer();
    }
  } catch (e) {
    if (e.name === "AbortError") return; // another file was opened meanwhile
    player.close();
    showError("That file couldn't be opened: " + e.message);
    target.file = null;
  }
  renderTargetFile();
  viewport.fit();
  const hash = await hashing;
  if (target.file !== file) return; // another file was opened meanwhile
  target.hash = hash;
  // its in and out points as saved: they don't need the models
  if (hash && kind === "video" && player.media) {
    const s = await saved.getEdits(hash);
    if (target.file !== file) return;
    // (put back, not an edit: nothing to undo)
    if (s?.range) untracked(() => player.setRange(s.range.a, s.range.b));
  }
  await findFaces();
}

// the video's name on its clip, above the face tracks (photos: renderPhotos)
function renderTargetFile() {
  if (target.file) $("tgtName").textContent = target.file.name;
}

// Closes the video, or every photo.
function clearTarget() {
  saveNow();
  photos.length = 0;
  target.bitmap?.close();
  Object.assign(target, { file: null, kind: "", hash: null, bitmap: null });
  resetScan(scan.id + 1);
  resetPreview();
  picked = new Set();
  wholeTracks = new Set();
  lastTrack = lastPart = null;
  clearResult();
  $("view").hidden = true;
  $("player").hidden = true;
  player.close();
  renderTargetFile();
  renderTimeline();
  render();
}

// ------------------------------------------------------------------ photos

const samePhoto = (a, b) => a.name === b.name && a.size === b.size && a.lastModified === b.lastModified;
const openPhotoOf = () => photos.find((p) => p.file === target.file) || null;
// whether Save saves every photo
const allPhotos = () => target.kind === "image" && photos.length > 1 && ui.photoScope === "all";

// Opens files dropped or chosen together: one video (in place of what's open), or photos (added to
// the photos open, else in place of the video). The first new photo is opened.
async function openFiles(list) {
  if (busy || saving) return;
  list = [...list];
  const videos = list.filter((f) => f.type.startsWith("video/")), images = list.filter((f) => f.type.startsWith("image/"));
  if (!videos.length && !images.length) { showError("Choose a video, or photos."); return; }
  if (videos.length && list.length > 1) { showError("Drop one video at a time, or one or more photos."); return; }
  if (videos.length) { photos.length = 0; await loadTarget(videos[0]); return; }
  if (images.length < list.length) showError(`${list.length - images.length} of those files aren't photos, so they were left out.`);
  if (target.kind !== "image") photos.length = 0;
  const added = [];
  for (const file of images) {
    let p = photos.find((x) => samePhoto(x.file, file));
    if (!p) { p = { file, serial: photoSerial++, sel: false, url: "", faces: null, swapped: null }; photos.push(p); thumbOfPhoto(p); }
    added.push(p);
  }
  await pickPhoto(added[0]);
}

let photoSerial = 1;
let photoAnchor = null; // the photo last clicked, where a Shift-click's range starts
// A photo clicked to open, until it's open. Opening takes a moment (the image is decoded), and the
// page shows the click at once: this one as the open photo, the one being left as not selected.
// The open photo's data (scan) stays the old one's until the new one is loaded.
let opening = null;
const selectedPhotos = () => photos.filter((p) => p.sel);
// the selected photos besides the open one, with their faces found
const otherPhotos = () => (target.kind === "image" ? photos.filter((p) => p.sel && p !== openPhotoOf() && p.people) : []);

// A click on a photo; how: as clickMode. Plain: only it selected, and open. Ctrl: added to the
// selection and opened, or taken out of it. Shift: the photos from the one last clicked to it
// selected, and it opened. The selected photos' faces show together, and the settings change them
// all.
async function pickPhoto(p, how = "") {
  if (busy || saving || !p) return;
  const cur = openPhotoOf();
  if (how === "toggle" && p.sel && selectedPhotos().length > 1) {
    p.sel = false;
    photoAnchor = p;
    const next = p === cur ? selectedPhotos().at(-1) : null;
    if (next) opening = next;
    photosChanged();
    if (next) await openWhile(next);
    return;
  }
  if (how === "range" && photos.includes(photoAnchor ?? cur)) {
    const a = photos.indexOf(photoAnchor ?? cur), b = photos.indexOf(p);
    for (const [i, x] of photos.entries()) x.sel = i >= Math.min(a, b) && i <= Math.max(a, b);
  } else {
    if (how !== "toggle") for (const x of photos) x.sel = false;
    p.sel = true;
    photoAnchor = p;
  }
  if (p !== cur) opening = p;
  photosChanged();
  await openWhile(p);
  loadSelectedPeople();
}

// opens p, showing it as the open photo meanwhile
async function openWhile(p) {
  try { await openPhoto(p); } finally { if (opening === p) { opening = null; render(); } }
}

async function openPhoto(p) {
  if (busy || saving || !p) return;
  if (target.file === p.file) { render(); return; }
  await loadTarget(p.file);
}

// The photos selected changed: the faces selected stay only when they're still shown.
function photosChanged() {
  const parts = new Set(people().flatMap((x) => x.parts.map((q) => q.id)));
  picked = new Set([...picked].filter((id) => parts.has(id)));
  render();
}

// Removes photos from the ones open (their saved work stays); when the open one goes, the next one
// opens.
function removePhotos(list) {
  if (busy || saving || !list.length) return;
  saveNow();
  const cur = openPhotoOf(), at = photos.indexOf(cur);
  for (const p of list) if (photos.includes(p)) photos.splice(photos.indexOf(p), 1);
  if (!photos.length) { clearTarget(); return; }
  if (photos.includes(cur)) { photosChanged(); return; }
  const next = photos[Math.min(at, photos.length - 1)];
  for (const x of photos) x.sel = x === next;
  photoAnchor = next;
  loadTarget(next.file);
}
const removePhoto = (p) => removePhotos([p]);

// Gets the faces and segments of the selected photos that aren't open: as saved, else found now.
let loadingPeople = false, loadAgain = false;
async function loadSelectedPeople() {
  if (loadingPeople) { loadAgain = true; return; }
  loadingPeople = true;
  try {
    do {
      loadAgain = false;
      const todo = photos.filter((p) => p.sel && p !== openPhotoOf() && !p.people && !p.failed);
      const fresh = [];
      for (const p of todo) {
        if (p.hash === undefined) p.hash = await saved.fileHash(p.file).catch(() => null);
        const s = p.hash ? await saved.getTarget(p.hash) : null;
        if (p.people || p.file === target.file) continue; // (opened meanwhile)
        if (s?.analysis?.kind === "image") takePhotoWork(p, s.analysis, s.people.map(person), true);
        else fresh.push(p);
      }
      if (fresh.length) await findPhotoFaces(fresh);
      photosChanged();
    } while (loadAgain);
  } finally {
    loadingPeople = false;
  }
}

// a photo's faces and segments, as saved (stored: no need to save them again) or just found
function takePhotoWork(p, analysis, list, stored = false) {
  p.analysis = analysis;
  p.people = list;
  p.saved = stored ? JSON.stringify({ people: savedPeopleOf(list), gone: [], range: null }) : null;
}

// Finds the faces of photos that aren't open. The engine holds one photo's faces at a time, so the
// preview waits meanwhile and the open photo's are given back after.
let pausePreview = false;
async function findPhotoFaces(list) {
  if (!ready || !engine) return;
  pausePreview = true;
  preview?.stop();
  try {
    for (let i = 0; i < 60 && (preview?.run || preview?.bg); i++) await new Promise((r) => setTimeout(r, 50));
    for (const p of list) {
      if (!photos.includes(p) || !p.sel || p.people || p.file === target.file) continue;
      try {
        const bm = await createImageBitmap(p.file, { imageOrientation: "from-image" });
        const r = await engine.analyze(`photo-${p.serial}`, bm, { recognize: true });
        takePhotoWork(p, r.analysis, r.people.map((x) => {
          const thumb = x.thumb ? thumbUrl(x.thumb) : "";
          x.thumb?.close();
          return person({ ...x, thumb, parts: [{ t0: 0, s: segmentDefaults() }] });
        }));
      } catch (e) {
        if (gpuGone(e)) return;
        console.error(e);
        p.failed = true;
      }
    }
  } finally {
    if (target.kind === "image" && scan.status === "done" && scan.imageAnalysis && engine && !engine.lost) {
      await engine.restoreAnalysis(scan.id, scan.imageAnalysis).catch((e) => console.warn(e));
    }
    pausePreview = false;
    saveSoon();
    render();
  }
}

// a photo's thumbnail, one at a time (a set can be large)
let thumbs = Promise.resolve();
function thumbOfPhoto(p) {
  thumbs = thumbs.then(async () => {
    if (!photos.includes(p)) return;
    try {
      const full = await createImageBitmap(p.file, { imageOrientation: "from-image" });
      p.width = full.width;
      p.height = full.height;
      const b = await createImageBitmap(full, { resizeHeight: Math.min(240, full.height), resizeQuality: "high" });
      full.close();
      const c = canvas2d(b.width, b.height);
      c.getContext("2d").drawImage(b, 0, 0);
      b.close();
      p.url = c.toDataURL("image/jpeg", 0.8);
    } catch { p.url = ""; }
    renderPhotos();
  });
}

let photosKey = "";
let photosOpen = null; // the photo the filmstrip last showed open (to scroll it into view when that changes)
// The photos on the timeline: in the top row, which ones are selected and the open one's details;
// in the media row, how many with add, output and remove all, and a filmstrip of the photos at
// the lane's height, each in its own shape, with the characters its faces become in its corner (a
// click opens it, Ctrl or Shift selects several, its x takes it out), then Add.
function renderPhotos() {
  const set = $("photoSet");
  set.hidden = target.kind !== "image" || !target.file;
  if (set.hidden) return;
  const cur = openPhotoOf();
  if (cur) {
    // (the open photo is one of the selected; while another one is being opened, it may not be)
    if (!photos.some((ph) => ph.sel)) cur.sel = true;
    if (scan.status === "done") Object.assign(cur, { people: scan.people, analysis: scan.imageAnalysis, hash: target.hash, saved: null });
  }
  // how many faces each photo has, and how many are swapped (with which characters)
  for (const ph of photos) {
    if (!ph.people) continue;
    ph.faces = ph.people.length;
    ph.swapped = ph.people.filter((x) => personSource(x).src).length;
    ph.chars = [...new Set(ph.people.map((x) => personSource(x).src?.key).filter(Boolean))];
  }
  // (the photo clicked shows as the open one at once, before it's loaded)
  const shown = opening || cur;
  const n = photos.length, at = photos.indexOf(shown), sel = selectedPhotos();
  const nums = sel.map((ph) => photos.indexOf(ph) + 1);
  $("psWhich").innerHTML = n < 2 ? "<b>Photo</b>" : sel.length > 1 ? `<b>${sel.length > 3 ? `${sel.length} photos` : `Photos ${nums.join(", ")}`}</b> of ${n}` : `<b>Photo ${at + 1}</b> of ${n}`;
  const nf = shown?.faces;
  const size = shown === cur || !shown?.width ? `${target.width}×${target.height}` : `${shown.width}×${shown.height}`;
  $("psInfo").textContent = `${(shown?.file || target.file).name} · ${size}${nf == null ? "" : ` · ${nf ? `${nf} face${nf > 1 ? "s" : ""}` : "no faces"}`}${sel.length > 1 ? " · Ctrl-click adds a photo, Shift-click a range" : ""}`;
  $("psCount").textContent = n > 1 ? `${n} photos` : target.file.name;
  $("psSel").textContent = n > 1 ? (sel.length > 1 ? `${sel.length} selected` : size) : size;
  $("psPrev").disabled = !!saving || at <= 0;
  $("psNext").disabled = !!saving || at < 0 || at >= n - 1;
  const shownFaces = sel.filter((ph) => ph.people);
  const faces = shownFaces.reduce((k, ph) => k + ph.faces, 0), swapped = shownFaces.reduce((k, ph) => k + ph.swapped, 0);
  const facesNote = faces ? `${sel.length > 1 ? `in ${sel.length} photos · ` : ""}${swapped} of ${faces} swapped` : "";
  if ($("psFacesNote").textContent !== facesNote) $("psFacesNote").textContent = facesNote;
  for (const b of set.querySelectorAll(".tgt-clear")) {
    b.title = n > 1 ? "Remove all photos" : "Remove the photo";
    b.setAttribute("aria-label", b.title);
  }
  const key = JSON.stringify([photos.map((p) => [p.url, p.width, p.faces, p.swapped, p.sel, (p.chars || []).map((k) => charByKey(k)?.url)]), at, busy || !!saving]);
  if (key === photosKey) return;
  photosKey = key;
  const row = $("psStrip");
  row.innerHTML = "";
  for (const [i, p] of photos.entries()) {
    const t = document.createElement("div");
    t.className = "ps-card";
    t.dataset.i = i;
    const open = p === shown;
    const chars = (p.chars || []).map(charByKey).filter(Boolean);
    const what = p.faces == null ? "" : !p.faces ? " · no faces found" : ` · ${p.swapped} of ${p.faces} face${p.faces > 1 ? "s" : ""} swapped`;
    t.innerHTML = `<button type="button" class="ps-pick" role="option">${p.url ? `<img alt="">` : ""}${chars.length ? `<span class="tos">${chars.slice(0, 3).map(() => `<img class="to" alt="">`).join("")}</span>` : ""}</button><button type="button" class="x">${X_ICON}</button>`;
    const pick = t.querySelector(".ps-pick");
    // (its shape before the thumbnail comes, so the strip doesn't move)
    pick.style.aspectRatio = p.width ? `${p.width} / ${p.height}` : "4 / 3";
    pick.setAttribute("aria-selected", String(!!p.sel));
    pick.classList.toggle("open", open);
    pick.title = `${p.file.name}${p.width ? ` · ${p.width}×${p.height}` : ""}${what}${open ? "" : ": click to open"} (Ctrl adds it to the selection, Shift selects a range)`;
    pick.disabled = !!saving;
    if (p.url) t.querySelector(".ps-pick > img").src = p.url;
    for (const [k, im] of [...t.querySelectorAll(".to")].entries()) im.src = chars[k].url;
    const x = t.querySelector(".x");
    x.title = `Remove ${p.file.name}`;
    x.setAttribute("aria-label", `Remove ${p.file.name}`);
    x.hidden = busy || !!saving;
    row.append(t);
  }
  const add = document.createElement("label");
  add.className = "ps-new";
  add.htmlFor = "tgtInput";
  add.title = "Add photos (or drop them here)";
  add.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 6v12M6 12h12"/></svg><span>Add</span>`;
  row.append(add);
  if (cur !== photosOpen) {
    photosOpen = cur;
    row.querySelector(`.ps-card[data-i="${at}"]`)?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }
}

const X_ICON = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 7l10 10M17 7 7 17"/></svg>`;

function initPhotos() {
  const strip = $("psStrip");
  const step = (d) => { const i = photos.indexOf(openPhotoOf()); pickPhoto(photos[Math.max(0, Math.min(photos.length - 1, i + d))]); };
  $("psDelete").onclick = () => removePhotos(selectedPhotos());
  strip.addEventListener("scroll", () => renderSelBar(), { passive: true });
  $("psPrev").onclick = () => step(-1);
  $("psNext").onclick = () => step(1);
  // the lane resized (the layout or a split moved): the open photo stays in view
  new ResizeObserver(() => strip.querySelector(".ps-pick[aria-selected=true]")?.scrollIntoView({ block: "nearest", inline: "nearest" })).observe(strip);
  strip.addEventListener("click", (e) => {
    const card = e.target.closest(".ps-card");
    const p = card && photos[+card.dataset.i];
    if (!p) return;
    if (e.target.closest(".x")) removePhoto(p);
    else if (e.target.closest(".ps-pick")) pickPhoto(p, clickMode(e));
  });
  // a wheel scrolls the rows sideways
  for (const row of [strip, $("people")]) {
    row.addEventListener("wheel", (e) => {
      if (row.scrollWidth <= row.clientWidth || Math.abs(e.deltaX) > Math.abs(e.deltaY)) return;
      e.preventDefault();
      row.scrollLeft += e.deltaY;
    }, { passive: false });
  }
  // the arrow keys (and Page Up / Down) open the photo before or after; Delete removes the
  // selected ones from the set
  addEventListener("keydown", (e) => {
    if (target.kind !== "image" || e.ctrlKey || e.metaKey || e.altKey || typing(e.target) || e.target.closest?.("dialog, [popover]")) return;
    if (e.key === "Delete" || e.key === "Backspace") { e.preventDefault(); removePhotos(selectedPhotos()); return; }
    const d = { ArrowLeft: -1, ArrowRight: 1, PageUp: -1, PageDown: 1 }[e.key];
    if (!d || photos.length < 2) return;
    e.preventDefault();
    step(d);
  });
}

// Waits until the models the open photo's segments want are loaded (they load by themselves).
async function modelsSettled(signal) {
  while (modelLoad || modelsMissing()) {
    if (signal.aborted) throw new DOMException("Stopped", "AbortError");
    const failed = Object.values(modelState).find((x) => x.failed);
    if (failed && !modelLoad) throw new Error(failed.failed);
    await new Promise((r) => setTimeout(r, 100));
  }
}

// Saves each photo with a face given a character to the outputs, in its own format: opened in turn
// and made by the preview. The photo open before comes back after.
async function saveAllPhotos() {
  const start = openPhotoOf(), list = [...photos], sels = new Map(photos.map((p) => [p, p.sel]));
  const ac = new AbortController();
  saving = { ac };
  abort = ac;
  showResumed(false);
  setProgress(0, "Saving…");
  render();
  const t0 = performance.now();
  let done = 0, left = 0, bytes = 0, finished = false;
  let together = null;
  const stopped = () => { if (ac.signal.aborted) throw new DOMException("Stopped", "AbortError"); };
  const set = workspace.outputSet(ac.signal);
  try {
    for (const [i, p] of list.entries()) {
      stopped();
      if (!photos.includes(p)) continue;
      setProgress(i / list.length, `Photo ${i + 1} of ${list.length}`, `Photo ${i + 1} of ${list.length} · ${done} saved`);
      // (a photo known to have no face given a character is passed by without opening it)
      if (p.swapped === 0) { left++; continue; }
      if (target.file !== p.file) await loadTarget(p.file);
      stopped();
      if (target.file !== p.file || scan.status !== "done") throw new Error(`${p.file.name} couldn't be opened.`);
      await modelsSettled(ac.signal);
      if (!swappedFaces(0).length) { left++; continue; }
      const f = photoFormat(p.file);
      const blob = await photoBlob(f);
      stopped();
      await set.add(blob, outputName(p.file, f.ext));
      bytes += blob.size;
      done++;
    }
    together = await set.finish();
    finished = true;
  } finally {
    abort = null;
    try {
      if (start && photos.includes(start) && target.file !== start.file) await loadTarget(start.file);
    } finally {
      saving = null;
    }
    for (const p of photos) p.sel = !!sels.get(p);
    if (finished && (done || left)) {
      const sep = `<span class="sep">·</span>`;
      setInfo(`<b>Jutsu complete</b><span class="sep">·</span>${done} photo${done === 1 ? "" : "s"} saved${together ? ` in ${together.name}` : ""}${left ? `${sep}${left} without a face given a character left out` : ""}${sep}${((performance.now() - t0) / 1000).toFixed(1)} s${sep}${fmtSize(bytes)}`, { save: done > 0 });
    }
    render();
  }
}

function clearResult() {
  setInfo("");
}

// ------------------------------------------------------------------ exporting

// What the engine swaps with: each person's segments, each with its face to use and settings.
function swapOpts() {
  const enh = enhancerIds(have.enhancer), occ = !!have.occluder;
  // a merged track's segments are every one of its analysis people's
  const plan = Object.fromEntries(scan.people.flatMap((p) => p.ids.map((id) => [id, p.parts.map(({ t0, s, off }) => ({
    t0,
    source: !off && have.swapper && charByKey(s.source)?.loaded ? s.source : null,
    pixelshuffle: s.pixelshuffle,
    blur: s.blur,
    enhance: enh.includes(s.enhance) ? { model: s.enhance, blend: s.enhanceBlend, detail: s.restore === "detail" } : null,
    occlusion: occ && s.occlusion,
    maskSize: s.maskSize,
  }))])));
  return { analysis: scan.id, plan };
}

// ------------------------------------------------------------------ browser flags

// The browser's own flags page for `flag` (brave://, edge://, opera:// or chrome://flags), or null
// where there's none: not Chromium, or a phone (whose flags pages don't list these)
function flagsUrl(flag) {
  const ua = navigator.userAgent;
  if (!/Chrome\//.test(ua) || /Android|Mobile/.test(ua)) return null;
  const scheme = navigator.brave ? "brave" : /Edg\//.test(ua) ? "edge" : /OPR\//.test(ua) ? "opera" : "chrome";
  return `${scheme}://flags/#${flag}`;
}
const WEBNN_FLAG_URL = flagsUrl(WEBNN_FLAG.split("#")[1]);
const webnnHelp = () => (WEBNN_FLAG_URL
  ? `WebNN is off in this browser. To turn it on, open ${WEBNN_FLAG_URL}, set WebNN API to Enabled and relaunch.`
  : "This browser doesn't have WebNN. Chrome and Edge on a computer do, behind a flag.");

// An option this browser can't do: it looks off, but its tooltip still says how to turn it on and,
// where a flag does that, a click copies the flag's address (a page can't open a flags page)
function unavailable(btn, help, flag) {
  btn.setAttribute("aria-disabled", "true");
  btn.classList.toggle("copies", !!flag);
  btn.title = flag ? `${help}

Click to copy the address.` : help;
  btn.onclick = flag ? () => copyFlag(btn, flag) : null;
}

async function copyFlag(btn, flag) {
  let ok = false;
  try {
    await navigator.clipboard.writeText(flag);
    ok = true;
  } catch {
    const t = Object.assign(document.createElement("textarea"), { value: flag });
    document.body.append(t);
    t.select();
    try { ok = document.execCommand("copy"); } catch { /* no clipboard */ }
    t.remove();
  }
  if (!ok) return showError(`Couldn't copy it. Open ${flag} in the address bar.`);
  btn.dataset.label ||= btn.textContent;
  btn.textContent = "Copied";
  btn.classList.add("copied");
  clearTimeout(btn.copiedTimer);
  btn.copiedTimer = setTimeout(() => { btn.textContent = btn.dataset.label; btn.classList.remove("copied"); }, 1600);
}

// ------------------------------------------------------------------ output

const encodable = new Map(); // "codec WxH" -> whether this browser can encode it

// the export's size: the video's
const outDims = () => ({ w: target.width || 1920, h: target.height || 1080 });

// whether this browser can encode `codec` at the export's size: false when it can't, undefined
// while that's being checked (render() runs again when it's known)
function canEncode(codec) {
  const { w, h } = outDims(), k = `${codec} ${w}x${h}`;
  if (!encodable.has(k)) {
    encodable.set(k, undefined);
    canEncodeVideo(codec, { width: w, height: h }).catch(() => false).then((v) => { encodable.set(k, v); render(); });
  }
  return encodable.get(k);
}

// the format and codec the export uses: the ones chosen, else the first this browser can encode
const outFormat = () => OUT_FORMATS[0];
// H.264, else the first codec MP4 takes that this browser can encode at the video's size
function outCodec() {
  const f = outFormat();
  return f.codecs.find((c) => canEncode(c) !== false) || f.codecs[0];
}
// a photo's format: its own (PNG for any other)
const photoFormat = (file = target.file) => PHOTO_FORMATS.find((x) => x.mime === file?.type || (x.id === "jpeg" && file?.type === "image/jpg")) || PHOTO_FORMATS[0];
// whether the export keeps to the in and out points
const inOut = () => target.kind === "video" && !!player.range && ui.outRange !== "all";

// What Export makes, on its badge: the format ("MP4", "PNG")
const goBadge = () => (!target.file ? "" : target.kind === "image" ? photoFormat().label : outFormat().label);

// "MP4 · 1080p": in Export's tooltip and the output section's head
function outSummary() {
  if (target.kind === "image") return photoFormat().label;
  return [outFormat().label, target.width && `${Math.min(target.width, target.height)}p`].filter(Boolean).join(" · ");
}

function renderOutput() {
  const photo = target.kind === "image";
  $("outVideo").hidden = photo;
  $("outPhoto").hidden = !photo;
  $("saveToSeg").sync?.();
  for (const id of ["outQualitySeg", "outRangeSeg", "photoScopeSeg"]) {
    $(id).sync();
    for (const b of $(id).children) b.disabled = busy;
  }
  $("outRangeRow").hidden = !(target.kind === "video" && player.range);
  $("photoScopeRow").hidden = !(photo && photos.length > 1);
  for (const b of $("photoScopeSeg").children) if (b.dataset.id === "all") b.textContent = `All ${photos.length}`;
  $("outPopSum").textContent = target.file ? outSummary() : "";
  // nothing to make: no output section
  const sec = $("outTitle").closest(".pop-sec");
  sec.hidden = !target.file;
  sec.nextElementSibling.hidden = !target.file;
}

const outOpen = () => $("setPop").matches(":popover-open");

// The settings panel: the header's gear (its popovertarget) opens it under the gear, kept inside
// the window; a click outside, Escape or the gear again closes it. Its own buttons close it too.
function initOutput() {
  const pop = $("setPop"), btn = $("settingsBtn");
  const place = () => {
    const r = btn.getBoundingClientRect(), m = 8, w = Math.min(340, innerWidth - 2 * m);
    pop.style.left = `${Math.max(m, Math.min(innerWidth - w - m, r.right - w))}px`;
    pop.style.top = `${r.bottom + 6}px`;
    pop.style.maxHeight = `${innerHeight - r.bottom - 6 - m}px`;
  };
  pop.addEventListener("beforetoggle", (e) => {
    const open = e.newState === "open";
    if (open) { renderOutput(); syncPanel(); renderBackend(); place(); }
    btn.setAttribute("aria-expanded", String(open));
  });
  addEventListener("resize", () => { if (outOpen()) place(); });
  $("keysBtn").onclick = () => { pop.hidePopover(); openKeys(); };
  $("resetBtn").onclick = () => { pop.hidePopover(); resetEdits(); };
  $("storageBtn").onclick = () => { pop.hidePopover(); openStorage(); };
  initStorage();
}

// the panel's layout and theme switches
function syncPanel() {
  $("layoutSeg").sync();
  $("themeSeg").sync();
}

// The theme: the system's (the CSS follows it), or light or dark whatever the system's is
const THEMES = [
  { id: "system", label: "System", tip: "Follow the system's light or dark mode" },
  { id: "light", label: "Light", tip: "Light" },
  { id: "dark", label: "Dark", tip: "Dark" },
];
function applyTheme() {
  const root = document.documentElement;
  if (ui.theme === "light" || ui.theme === "dark") root.dataset.theme = ui.theme;
  else delete root.dataset.theme;
  player?.invalidate();
}

// The keyboard shortcuts' sheet (the panel's button, or ?)
const KEYS = [
  ["Playback", [["Play or pause", "Space"], ["Previous / next frame", "← →"], ["10 frames back / on", "Shift ← →"], ["To the in / out point", "Home End"], ["Mute", "M"], ["Auto (work ahead)", "W"]]],
  ["In and out points", [["Set the in point", "I"], ["Set the out point", "O"], ["Clear the in / out point", "Alt I  Alt O"], ["Clear both", "Alt X"]]],
  ["Editing", [["Undo", "Ctrl Z"], ["Redo", "Ctrl Y"], ["Split at the playhead", "S"], ["Join segments", "J"], ["Merge tracks", "Shift J"], ["Delete", "Delete"], ["Select every track", "Ctrl A"], ["Rename", "F2"], ["Back to every face", "Esc"]]],
  ["View", [["Zoom in / out", "+ −"], ["Fit", "0"], ["Fit the timeline", "F"], ["Landscape / vertical layout", "L"], ["Previous / next photo", "← →"], ["Keyboard shortcuts", "?"]]],
];
function openKeys() {
  const list = $("keysList");
  if (!list.childElementCount) {
    for (const [title, rows] of KEYS) {
      const h = document.createElement("h3");
      h.textContent = title;
      list.append(h);
      for (const [what, key] of rows) {
        const row = document.createElement("div");
        row.className = "k";
        row.append(what);
        const k = document.createElement("kbd");
        k.textContent = key;
        row.append(k);
        list.append(row);
      }
    }
  }
  $("keys").showModal();
}

async function go() {
  if (busy || !ready) return;
  showError("");
  try {
    if (target.kind === "image") await (allPhotos() ? saveAllPhotos() : savePhoto());
    else await exportVideo();
  } catch (e) {
    if (gpuGone(e)) showError("The export stopped because the graphics card stopped. Export again once the models are loaded again.");
    else if (e.name !== "AbortError") { console.error(e); showError(e.message); }
  }
}

// Asks where to save `name`, when the output step says so and the browser can. It must be the
// The info area's line on what was made: `html`, and the name it was saved as
function showResult(html, saved) {
  setInfo(`${html}<span class="sep">·</span>Saved as <b class="saved-as"></b>`, { save: !!saved.path });
  $("infoMeta").querySelector(".saved-as").textContent = saved.name;
}

// The photo as the preview shows it, in the output step's format: made by the preview, if it
// isn't yet.
async function savePhoto() {
  const file = target.file, f = photoFormat();
  const name = outputName(file, f.ext);
  const ac = new AbortController();
  abort = ac; // (Stop, should the server have stopped and the saving wait for it)
  working = true;
  showResumed(false);
  setProgress(null, "Saving…");
  render();
  const t0 = performance.now();
  let blob, kept;
  try {
    blob = await photoBlob(f);
    if (target.file !== file) return;
    kept = await workspace.saveOutput(blob, name, ac.signal);
  } finally {
    working = false;
    abort = null;
    render();
  }
  const n = swappedFaces(0).length;
  showResult(`${n ? `<b>Jutsu complete</b><span class="sep">·</span>${n} face${n > 1 ? "s" : ""} swapped` : "<b>No face swapped</b>"}<span class="sep">·</span>${target.width}×${target.height}<span class="sep">·</span>${((performance.now() - t0) / 1000).toFixed(2)} s<span class="sep">·</span>${fmtSize(blob.size)}`, kept);
}

// The open photo as the preview shows it, in format f (made by the preview, if it isn't yet).
async function photoBlob(f) {
  const file = target.file;
  await preview.ensure(0);
  if (target.file !== file) throw new DOMException("Another photo was opened", "AbortError");
  const cv = canvas2d(target.width, target.height);
  const ctx = cv.getContext("2d");
  ctx.drawImage(target.bitmap, 0, 0);
  preview.draw(ctx, 0);
  return new Promise((res) => cv.toBlob(res, f.mime, PHOTO_QUALITY));
}

// Exports the video as the output step says to the outputs, written as it's made: between the in
// and out points (or all of it), what the preview made, and the frames it hasn't swapped yet.
async function exportVideo() {
  const fmt = outFormat(), codec = outCodec(), quality = OUT_QUALITY.find((q) => q.id === ui.outQuality) || OUT_QUALITY[2];
  // (the first thing after the click: where the browser asks where to save, it needs one)
  const name = await workspace.chooseOutput(outputName(target.file, fmt.ext), fmt, ui.saveMode);
  // media times: from the in point's start to the end of the out point
  const trim = inOut() && { start: player.mediaTime(player.inPoint), end: player.outPoint + 1 < player.total ? player.mediaTime(player.outPoint + 1) : player.media.demux.index.end };
  clearResult();
  busy = true;
  exporting = true;
  showResumed(false);
  abort = new AbortController();
  preview.stop();
  player.stop();
  picked = new Set();
  wholeTracks = new Set();
  setProgress(0, "Starting…");
  render();
  const t0 = performance.now();
  let done = false;
  stopScan();
  try {
    if (!(await openPreviewTarget())) return;
    const r = await engine.swapVideo(target.file, { ...(plan || swapOpts()), trim }, {
      name,
      out: { format: fmt.id, codec, quality: quality.id, size: 0, audio: true },
      signal: abort.signal,
      onProgress: (p) => {
        if (p.scan) takeScan(p.scan);
        if (p.frac == null) return;
        // the playhead marks the frame being made
        player.track((trim ? player.inPoint : 0) + Math.max(0, p.frames - 1));
        const elapsed = (performance.now() - t0) / 1000;
        const eta = p.frac > 0.02 ? elapsed / p.frac - elapsed : NaN;
        setProgress(p.frac, `Swapping ${Math.round(100 * p.frac)}%`, `${Math.round(100 * p.frac)}% · frame ${p.frames} · ${p.fps.toFixed(1)} fps${Number.isFinite(eta) ? ` · about ${fmtTime(eta)} left` : ""}`);
      },
      onPreview: (bm) => { showImage(bm, bm.width, bm.height); bm.close(); },
    });
    const c = OUT_CODECS.find((x) => x.id === r.codec)?.label || r.codec;
    await workspace.handOver(r.saved);
    showResult(`<b>Jutsu complete</b><span class="sep">·</span>${r.frames} frames<span class="sep">·</span>${r.width}×${r.height} ${c}<span class="sep">·</span>${fmtTime(r.ms / 1000)} (${(r.frames / (r.ms / 1000)).toFixed(1)} fps)<span class="sep">·</span>${fmtSize(r.size)}`, r.saved);
    done = true;
  } finally {
    busy = false;
    exporting = false;
    abort = null;
    player.untrack();
    if (player.media) showPlayer(); // back from the export's frames
    player.invalidate();
    render();
    saveAnalysis(true);
    keepScanning(true);
  }
}

// hold to compare: the photo or video as it is, while pressed
function compare(on) {
  if (!preview) return;
  $("compareBtn").setAttribute("aria-pressed", String(on));
  preview.original = on;
  if (target.kind === "image") drawPhoto();
  else player.invalidate();
  renderBadge();
}

// ------------------------------------------------------------------ models

// Loads the detector and the recognizer, in place of the whole layout: what finding faces and
// making characters take. The other models come once the segments want them (syncModels).
async function loadModels() {
  modelLoad?.ac.abort();
  $("loadingTitle").textContent = "Loading the face detector…";
  $("loadingText").textContent = "";
  showPanel("loading");
  $("loadingBar").style.width = "0%";
  const ac = new AbortController();
  ready = false;
  stopScan();
  render();
  try {
    if (engine.lost) await restartEngine();
    await engine.load(BASE, manifest, { ...selection(), backend: ui.backend, winograd: params.get("winograd") !== "0" }, {
      signal: ac.signal,
      onStatus: (s) => {
        if (s.phase === "load") {
          $("loadingTitle").textContent = `Loading the ${s.what}…`;
          $("loadingBar").style.width = `${(100 * s.frac).toFixed(0)}%`;
          $("loadingText").textContent = "Moving the weights onto your graphics card.";
        }
      },
    });
    ready = true;
    showLost("");
    epoch++;
    have = {};
    modelSkip = "";
    modelState = {};
    if (ui.backend === "webnn" && engine.gpu.backend !== "webnn") {
      showError(`WebNN couldn't start, so the WebGPU engine is used instead: ${engine.gpu.nnError || "unknown error"}`);
    }
    // characters saved on earlier visits, or made before the models were removed, go to the engine now
    for (const c of characters.filter((x) => !x.loaded)) await loadCharacter(c);
    if (target.file && scan.status !== "done") await findFaces();
    else keepScanning(true);
  } catch (e) {
    console.error(e);
    if (gpuGone(e)) { render(); return; } // gpuLost loads them again, or says why it doesn't
    await welcome();
    if (e.name !== "AbortError") {
      showError(/out of memory|allocation/i.test(e.message) ? `The jutsu fizzled: your graphics card ran out of memory. ${MEMORY_ADVICE}` : e.message);
    }
  }
  render();
}

const MODEL_PARTS = ["swapper", "enhancer", "occluder"];
// what the models' buttons call them
const PART_NAMES = { swapper: "face swapper", enhancer: "enhancer", occluder: "occlusion mask" };
// whether a and b ({ swapper, enhancer, occluder }) have model k the same
const sameModel = (a, b, k) => (k === "enhancer" ? (a[k] || "") === (b[k] || "") : !!a[k] === !!b[k]);

// Loads the models to have that the engine doesn't have yet and unloads the others, each in its
// card while the rest of the app keeps working.
function syncModels() {
  if (!ready || busy || modelLoad || !engine || engine.lost) return;
  // (one unloaded by hand and wanted again stays once it's no longer wanted)
  const w = wanted();
  for (const k of MODEL_PARTS) if (w[k] && keep[k] === false) delete keep[k];
  const t = modelTarget();
  if (MODEL_PARTS.every((k) => sameModel(t, have, k))) return;
  const key = JSON.stringify([t, ui.backend]);
  if (key === modelSkip) return;
  loadInline(t, key);
}

async function loadInline(parts, key) {
  const ac = new AbortController();
  modelLoad = { ac };
  const coming = MODEL_PARTS.filter((k) => parts[k] && !sameModel(parts, have, k));
  modelState = Object.fromEntries(coming.map((k) => [k, { phase: "wait", frac: 0 }]));
  renderModels();
  renderSteps();
  try {
    await engine.load(BASE, manifest, { ...selection(parts), backend: ui.backend, winograd: params.get("winograd") !== "0" }, {
      signal: ac.signal,
      onStatus: (s) => {
        const k = s.part?.split(":")[0];
        if (!modelState[k]) return;
        if (s.phase === "load") modelState[k] = { phase: "load", frac: s.frac };
        renderModels();
        renderSteps();
      },
    });
    // (the models were loaded again from the start meanwhile: only those count)
    if (!ready) return;
    have = parts;
    modelState = {};
    epoch++;
  } catch (e) {
    modelState = {};
    if (gpuGone(e) || !ready) return; // gpuLost loads them again, or says why it doesn't
    // those it had and was to keep are still there; the ones it was to drop went first
    have = Object.fromEntries(MODEL_PARTS.map((k) => [k, sameModel(parts, have, k) ? have[k] : false]));
    if (e.name === "AbortError") return; // unloaded while it was read
    console.error(e);
    modelSkip = key;
    const why = /out of memory|allocation/i.test(e.message) ? `The jutsu fizzled: your graphics card ran out of memory. ${MEMORY_ADVICE}` : e.message;
    for (const k of coming) modelState[k] = { failed: why };
  } finally {
    if (modelLoad?.ac === ac) modelLoad = null;
    render();
  }
}

// what a model's card shows: { name, size } of the one it would load, or null when there's none
function modelInfo(k) {
  const info = k === "swapper" ? swapperInfo() : k === "enhancer" ? enhancerInfo() : occluderInfo();
  if (!manifest || !info) return null;
  let size = 0;
  try { size = resolveFiles(manifest, selection({ [k]: k === "enhancer" ? info.id : true }))[k === "enhancer" ? `enhancer:${info.id}` : k].size; } catch { /* not built */ }
  return { name: info.label, size };
}

const ICON_LOAD = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4v11M7 10l5 5 5-5M5 20h14"/></svg>`;
const ICON_UNLOAD = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5l7 8H5zM5 19h14"/></svg>`;

// The models' buttons in their cards' heads: load, or unload while the model is loaded or loading
// (not while a segment uses it: it would only be loaded again).
function renderModels() {
  const w = wanted();
  for (const sw of document.querySelectorAll(".step-load")) {
    const k = sw.dataset.part;
    const info = ready ? modelInfo(k) : null;
    sw.hidden = !info;
    if (!info) continue;
    const st = modelState[k];
    const loading = !!st && !st.failed;
    const on = keep[k] !== false && (modelLoaded(k) || loading);
    const inUse = on && (k === "enhancer" ? enhancerIds(w.enhancer).includes(enhancerInfo().id) : !!w[k]);
    if (sw.dataset.on !== String(on)) sw.innerHTML = on ? ICON_UNLOAD : ICON_LOAD;
    sw.dataset.on = String(on);
    sw.title = inUse ? "In use" : on ? "Unload to free memory" : `Load${info.size ? ` · ${fmtSize(info.size)}` : ""}`;
    sw.setAttribute("aria-label", on ? `Unload the ${PART_NAMES[k]}` : `Load the ${PART_NAMES[k]}`);
    sw.disabled = busy || inUse;
    sw.onclick = () => {
      if (busy) return;
      keep[k] = !on;
      if (on && loading) modelLoad?.ac.abort(); // (reading the file stops; loading onto the graphics card finishes first)
      modelSkip = "";
      delete modelState[k];
      render();
    };
  }
}

const modelLoaded = (k) => !!have[k] && (k !== "enhancer" || enhancerIds(have[k]).includes(enhancerInfo()?.id));

// ------------------------------------------------------------------ a lost graphics card

const MEMORY_ADVICE = "Close other tabs and apps.";
const RETRY_AFTER = 30000; // ms: lost again this soon after loading the models again, it stops
let restarting = false; // loading the models again after a loss
let restartedAt = -Infinity;

// whether an engine call failed because the GPU was lost (the call's own error can come back
// before the engine hears of the loss)
const gpuGone = (e) => e?.name === "GPULostError" || !!engine?.lost || /device (is |was )?lost|context (is |was )?lost/i.test(e?.message || "");

// the note on the canvas while the models are loaded again, or why they aren't (with Reload)
function showLost(text, stuck = false) {
  $("lostCard").hidden = !text;
  $("lostText").textContent = text;
  $("reloadBtn").hidden = !stuck;
}

// The GPU was lost: a driver reset, the GPU process crashing, or running out of memory (the app
// never lets go of its device itself). Nothing the engine had on it works any more, so a fresh
// one loads the models again (from disk), the characters and the faces found. The edits live
// on the page and stay; the frames already swapped come back from the patches kept on disk.
// Lost again while doing that or soon after, it stops trying.
async function gpuLost(info) {
  console.warn("the graphics card stopped:", info.message);
  const reload = ready || !$("loading").hidden; // (before the models are loaded, nothing to load again)
  const again = restarting || performance.now() - restartedAt < RETRY_AFTER;
  ready = false;
  for (const c of characters) c.loaded = false;
  resetPreview();
  render();
  if (again) {
    showLost(`The graphics card stopped again, so the models aren't loaded again by themselves. ${MEMORY_ADVICE} Then start again below, or reload the page.`, true);
    await welcome();
    render();
    return;
  }
  if (!reload) return; // a fresh engine when they are
  restarting = true;
  showLost("The graphics card stopped, so the models are being loaded again.");
  try {
    await loadModels();
  } finally {
    restarting = false;
    restartedAt = performance.now();
  }
  // (lost again meanwhile: that says so already)
  if (!ready && !engine.lost) {
    showLost(`The graphics card stopped, and the models couldn't be loaded again. ${MEMORY_ADVICE} Then start again below, or reload the page.`, true);
    await welcome();
  }
}

// A fresh engine in place of the one whose GPU was lost. The faces found in the photo or video
// go to it from the saved work (without them, they're found again once the models are loaded).
async function restartEngine() {
  // the browser brings its GPU back a moment after it crashed: until then there's no adapter
  for (let i = 0; i < 20 && !(await navigator.gpu.requestAdapter({ powerPreference: "high-performance" }).catch(() => null)); i++) {
    await new Promise((r) => setTimeout(r, 500));
  }
  await engine.restart();
  for (const c of characters) c.loaded = false;
  have = {};
  modelState = {};
  resetPreview();
  if (scan.status !== "done") return;
  const id = scan.id;
  if (scan.analysis) { await engine.restoreAnalysis(id, scan.analysis); return; }
  const s = target.hash ? await saved.getTarget(target.hash) : null;
  if (scan.id !== id) return;
  if (s?.analysis) await engine.restoreAnalysis(id, s.analysis);
  else scan.status = "";
}

// The start of the app when the models aren't loaded: what the models folder lacks (`missing`:
// files with their addresses, and Download all), or the button to load them again (after a lost
// GPU, or a failure).
async function welcome(missing = [], extras = []) {
  showPanel("welcome");
  const go = $("welcomeGo");
  go.textContent = missing.length ? "Check again" : "Start";
  // (checking again is for files put in the models folder by hand)
  go.hidden = !!missing.length && !workspace.modelsByHand;
  // (a reload: the settings are built from what's in the folder)
  go.onclick = missing.length ? () => location.reload() : loadModels;
  $("welcomeNote").textContent = "Works best in Chrome or Edge on a laptop or desktop.";
  $("modelsNeeded").hidden = !missing.length;
  $("modelsOpen").hidden = !missing.length;
  $("modelsOpen").onclick = () => workspace.openFolder("models");
  $("modelsAll").hidden = true;
  go.classList.replace("ghost", "primary");
  if (!missing.length) return;
  const [where, { urls, problem }] = await Promise.all([workspace.status(), workspace.modelUrls()]);
  $("modelsPath").textContent = where.models;
  $("modelsPath").parentElement.title = where.models;
  const files = [...missing, ...extras];
  const reachable = files.filter((f) => urls[f.name]);
  if (!reachable.length) go.hidden = false;
  const left = new Set(reachable); // the files still to get
  const rows = new Map();
  const get = (list) => getModels(list, { rows, left, missing, all: reachable });
  for (const f of files) rows.set(f.name, modelRow(f, urls[f.name], () => get([f])));
  $("modelsMissing").replaceChildren(...[...rows.values()].map((r) => r.li));
  if (reachable.length) {
    $("modelsAll").hidden = false;
    go.classList.replace("primary", "ghost");
    $("modelsAll").textContent = "Download all";
    $("modelsAll").onclick = () => get([...left]);
  }
  if (problem) $("welcomeNote").textContent = problem;
  else if (reachable.length < files.length) $("welcomeNote").textContent = "The files without a Download have no address in config.toml, next to the app.";
}

// A file's row on the start page: what it does, and its Download while it isn't downloading. Without
// an address, its name in its place, to put it in the folder by hand.
function modelRow(f, url, onGet) {
  const li = document.createElement("li");
  const span = (className, textContent) => Object.assign(document.createElement("span"), { className, textContent });
  const what = span("need-what", f.what);
  what.title = f.name;
  const state = span("need-state", url ? "" : f.name);
  const button = (label) => {
    const b = Object.assign(document.createElement("button"), { type: "button", className: "need-get", textContent: label, title: `${f.name}\nFrom ${url}` });
    b.onclick = onGet;
    return b;
  };
  if (url) state.append(button("Download"));
  else { state.classList.add("need-file"); state.title = f.name; }
  li.append(what, span("need-tag", f.optional ? "Optional" : ""), span("need-size", fmtSize(f.size)), state);
  return {
    li,
    show(text) {
      state.textContent = text;
      li.classList.remove("failed");
    },
    failed() {
      state.replaceChildren(button("Retry"));
      li.classList.add("failed");
    },
  };
}

// Into the models folder, one at a time. All of them there: on into the app. Only those it can't do
// without: it's for the user to go on or get the others.
async function getModels(list, { rows, left, missing, all }) {
  const btn = $("modelsAll");
  const go = $("welcomeGo");
  const buttons = () => document.querySelectorAll(".need-get");
  btn.disabled = go.disabled = true;
  for (const b of buttons()) b.disabled = true;
  const total = list.reduce((a, f) => a + f.size, 0);
  const why = [];
  let finished = 0;
  $("welcomeNote").textContent = "";
  for (const f of list) rows.get(f.name).show("Waiting…");
  for (const f of list) {
    const row = rows.get(f.name);
    row.show("0%");
    try {
      await workspace.downloadModel(f.name, (done, size) => {
        row.show(`${Math.floor(100 * done / (size || f.size))}%`);
        btn.textContent = `Downloading… ${Math.floor(100 * (finished + done) / total)}%`;
      });
      row.show("Done");
      left.delete(f);
    } catch (e) {
      row.failed();
      why.push(`${f.name}: ${e.message}`);
      $("welcomeNote").textContent = why.join(" · ");
    }
    finished += f.size;
  }
  // (a reload: the settings are built from what's in the folder)
  if (!left.size && !why.length) { location.reload(); return; }
  btn.disabled = go.disabled = false;
  for (const b of buttons()) b.disabled = false;
  btn.textContent = document.querySelector("#modelsMissing li.failed") ? "Try again" : "Download all";
  btn.hidden = !left.size;
  if (!missing.some((f) => left.has(f) || !all.includes(f))) {
    go.textContent = "Start";
    go.hidden = false;
    go.classList.replace("ghost", "primary");
    btn.classList.replace("primary", "ghost");
  }
}

// ------------------------------------------------------------------ settings

// The dialog opens at once; what's there is filled in as it's known
async function openStorage() {
  const dlg = $("storageDlg");
  if (!dlg.open) {
    if (!$("storageText").textContent) $("storageText").textContent = "Counting…";
    dlg.showModal();
  }
  renderStorage();
}

// The settings panel's engine row; its tooltip says what's running
function renderBackend() {
  const el = $("backendSeg");
  const nn = hasWebNN();
  el.innerHTML = "";
  for (const b of BACKENDS) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.setAttribute("role", "radio");
    btn.setAttribute("aria-checked", String(ui.backend === b.id));
    btn.textContent = b.label;
    btn.title = b.tip;
    btn.disabled = busy;
    btn.onclick = async () => {
      if (busy || ui.backend === b.id) return;
      ui.backend = b.id;
      saveUi();
      renderBackend();
      if (ready) { $("setPop").hidePopover(); await loadModels(); }
    };
    if (b.id === "webnn" && !nn) unavailable(btn, webnnHelp(), WEBNN_FLAG_URL);
    el.append(btn);
  }
  placeThumb(el);
  segThumbs.observe(el);
  const g = engine?.gpu || {};
  const running = !ready ? "" : g.backend !== "webnn" ? "WebGPU" : `WebNN ${g.nnParts.join(", ").replace(/, ([^,]*)$/, " and $1")}, WebGPU for the rest`;
  const help = nn ? "WebNN runs the face swapper (most of the work) on your graphics card's fp16 tensor cores: several times faster. It is experimental: if it fails to start, the WebGPU engine takes over. Switching loads the models again."
    : `${webnnHelp()} It is several times faster on most graphics cards.`;
  const info = running && `Running: ${running}${engine?.inWorker ? " (in a worker)" : ""}${g.name ? ` · ${g.name}` : ""}${g.peak ? ` · peak working memory ${fmtSize(g.peak)}` : ""}`;
  $("backendRow").title = [help, info].filter(Boolean).join("\n\n");
}

let storageRun = 0; // the latest renderStorage: an older one's answers are out of date
async function renderStorage() {
  const run = ++storageRun;
  renderKept(run);
  saved.summary().then((n) => {
    if (run !== storageRun) return;
    const what = [[n.characters, "character"], [n.targets, "photo or video", "photos and videos"]]
      .filter(([k]) => k).map(([k, one, many]) => `${k} ${k > 1 ? many || one + "s" : one}`);
    $("savedText").textContent = what.length ? `Your work: ${what.join(", ")}, with the faces found and the settings given them.` : "";
    $("clearSavedBtn").hidden = !what.length;
  });
  const [entries, where] = await Promise.all([workspace.modelFiles().catch(() => []), workspace.status().catch(() => null)]);
  if (run !== storageRun) return;
  const total = entries.reduce((a, e) => a + e.size, 0);
  $("storageText").textContent = entries.length ? `${fmtSize(total)} in ${entries.length} file${entries.length > 1 ? "s" : ""}` : "None yet.";
  $("modelsDir").textContent = where?.models ?? "";
  $("outputsDir").textContent = where?.outputs ?? "";
  $("workDir").textContent = where?.work ?? "";
  $("cacheDir").textContent = where?.cache ?? "";
}

// The swapped frames' line, and its button when there are any.
async function renderKept(run) {
  const kept = await workspace.patchUsage().catch(() => null);
  if (run !== storageRun) return;
  $("clearPatchesBtn").hidden = !kept?.count;
  $("patchText").textContent = kept?.count ? `Swapped frames: ${fmtSize(kept.bytes)}, kept so they're shown and exported again without swapping them again.` : "";
}

// The dialog's buttons answer at once, whatever is still being counted. btn stays disabled while
// work() runs, and the dialog is filled in again after it.
function initStorage() {
  const clearing = async (btn, work) => {
    if (busy || btn.disabled) return;
    btn.disabled = true;
    storageRun++;
    try {
      await work();
    } finally {
      btn.disabled = false;
      renderStorage();
    }
  };
  const frames = $("clearPatchesBtn");
  frames.onclick = () => clearing(frames, async () => {
    $("patchText").textContent = "Swapped frames: removing…";
    preview.stop();
    await engine.clearPatches();
    $("patchText").textContent = "";
    frames.hidden = true;
    preview.reset();
    player.invalidate();
    drawPhoto();
    render();
  });
  const work = $("clearSavedBtn");
  work.onclick = () => clearing(work, async () => {
    await saved.clearAll();
    lastSaved = "";
  });
  $("openOutputsBtn").onclick = () => workspace.openFolder("outputs");
  $("openModelsBtn").onclick = () => workspace.openFolder("models");
}

// ------------------------------------------------------------------ wiring

// a drop target for files (other drags, as a face to use, pass by); multiple: onFiles gets them all
function dropZone(el, onFile, multiple = false) {
  const files = (e) => e.dataTransfer?.types.includes("Files");
  el.addEventListener("dragover", (e) => { if (!files(e)) return; e.preventDefault(); el.classList.add("over"); });
  el.addEventListener("dragleave", (e) => { if (!el.contains(e.relatedTarget)) el.classList.remove("over"); });
  el.addEventListener("drop", (e) => {
    el.classList.remove("over");
    if (!files(e)) return;
    e.preventDefault();
    const list = [...(e.dataTransfer.files || [])];
    if (multiple) { if (list.length) onFile(list); }
    else if (list.length > 1) showError("Drop one file at a time.");
    else if (list[0]) onFile(list[0]);
  });
}

// a character dragged onto a face in the settings or a timeline track
const CHAR_TYPE = "application/x-fs-character";
function charDrops(el) {
  const row = (e) => e.dataTransfer?.types.includes(CHAR_TYPE) && e.target.closest?.("[data-id]");
  el.addEventListener("dragover", (e) => {
    const r = row(e);
    for (const x of el.querySelectorAll(".drop-face")) if (x !== r) x.classList.remove("drop-face");
    if (!r) return;
    e.preventDefault();
    e.stopPropagation();
    e.dataTransfer.dropEffect = "copy";
    r.classList.add("drop-face");
  });
  el.addEventListener("dragleave", (e) => { if (!el.contains(e.relatedTarget)) for (const x of el.querySelectorAll(".drop-face")) x.classList.remove("drop-face"); });
  el.addEventListener("drop", (e) => {
    const r = row(e);
    for (const x of el.querySelectorAll(".drop-face")) x.classList.remove("drop-face");
    if (!r) return;
    e.preventDefault();
    e.stopPropagation();
    assign(r.dataset.id, e.dataTransfer.getData(CHAR_TYPE));
    selectPerson(findPerson(r.dataset.id) ? r.dataset.id : null);
  });
}

function initChars() {
  const grid = $("charGrid");
  // a wheel scrolls the row sideways
  grid.addEventListener("wheel", (e) => {
    if (grid.scrollWidth <= grid.clientWidth || Math.abs(e.deltaX) > Math.abs(e.deltaY)) return;
    e.preventDefault();
    grid.scrollLeft += e.deltaY;
  }, { passive: false });
  $("srcInput").onchange = (e) => { makeCharacter([...e.target.files]); e.target.value = ""; };
  // images dropped on the characters, or in their place before there's one
  dropZone($("chars"), makeCharacter, true);
  dropZone($("srcStart"), makeCharacter, true);
}

function wire() {
  // while a video exports, or every photo is saved, the editor's keys do nothing (they go to the
  // page's listeners, on window, after this one)
  document.addEventListener("keydown", (e) => {
    if ((exporting || saving) && !e.target.closest?.("dialog")) e.stopPropagation();
  }, true);
  // the settings
  multiSeg($("detailSeg"), DETAIL, () => (one("pixelshuffle") === undefined ? [] : pixelshuffleSet(one("pixelshuffle"))), (v) => setSegment("pixelshuffle", v.length > 1 ? v : v[0]));
  $("blurRange").oninput = (e) => setSegment("blur", +e.target.value, { coalesce: true });
  $("blurRange").onchange = endEdit;
  $("maskSizeRange").oninput = (e) => setSegment("maskSize", +e.target.value, { coalesce: true });
  $("maskSizeRange").onchange = endEdit;
  seg($("enhanceModeSeg"), RESTORE, () => one("restore"), (v) => setSegment("restore", v));
  $("enhanceBlendRange").oninput = (e) => setSegment("enhanceBlend", +e.target.value, { coalesce: true });
  $("enhanceBlendRange").onchange = endEdit;
  // the output, the whole project's
  const out = (k) => (v) => { ui[k] = v; saveUi(); };
  seg($("outQualitySeg"), OUT_QUALITY, () => ui.outQuality, out("outQuality"));
  seg($("outRangeSeg"), OUT_RANGE, () => ui.outRange, out("outRange"));
  seg($("photoScopeSeg"), PHOTO_SCOPE, () => ui.photoScope, out("photoScope"));
  const choices = workspace.saveChoices;
  if (choices) {
    $("saveRow").hidden = false;
    seg($("saveToSeg"), choices.options, () => (choices.help ? choices.options[0].id : ui.saveMode), out("saveMode"));
    if (choices.help) unavailable($("saveToSeg").lastElementChild, choices.help, choices.flag && flagsUrl(choices.flag));
  }
  seg($("layoutSeg"), [{ id: "land", label: "Landscape", tip: "Settings in a full-height column on the right, for landscape videos (L)" }, { id: "port", label: "Vertical", tip: "The canvas the full height, for vertical videos (L)" }], () => layout.mode(), (v) => layout.setMode(v));
  seg($("themeSeg"), THEMES, () => ui.theme, (v) => { ui.theme = v; saveUi(); applyTheme(); syncPanel(); });
  layout.onChange(() => { if (outOpen()) syncPanel(); player?.invalidate(); });
  matchMedia("(prefers-color-scheme: light)").addEventListener("change", () => player?.invalidate());
  $("resultClose").onclick = clearResult;
  $("saveBtn").onclick = () => workspace.openFolder("outputs");
  $("resumedReset").onclick = resetEdits;
  $("resumedClose").onclick = () => showResumed(false);
  // ? opens the keyboard shortcuts
  addEventListener("keydown", (e) => {
    if (e.key !== "?" || e.ctrlKey || e.metaKey || e.altKey || typing(e.target) || e.target.closest?.("dialog")) return;
    e.preventDefault();
    openKeys();
  });
  $("splitBtn").onclick = split;
  $("joinBtn").onclick = join;
  $("mergeBtn").onclick = merge;
  $("deleteBtn").onclick = remove;
  $("undoBtn").onclick = () => undo();
  $("redoBtn").onclick = () => undo(false);
  // a drag's edits (a slider, an in or out marker) are one step until it's let go; a key's, until
  // the key is
  for (const ev of ["pointerup", "pointercancel", "keyup"]) addEventListener(ev, endEdit, true);
  player.beforeRange = () => { if (!quiet) edit("in and out points", () => {}, { coalesce: true }); };
  // Ctrl/Cmd+Z undoes, Ctrl/Cmd+Shift+Z and Ctrl+Y redo; in a text field, the field's own undo does
  addEventListener("keydown", (e) => {
    if (!(e.ctrlKey || e.metaKey) || e.altKey || textField(e.target)) return;
    const k = e.key.toLowerCase();
    if (k !== "z" && k !== "y") return;
    e.preventDefault();
    undo(k === "z" && !e.shiftKey);
  });
  // Ctrl/Cmd+A selects every track
  addEventListener("keydown", (e) => {
    if (!(e.ctrlKey || e.metaKey) || e.altKey || e.shiftKey || e.key.toLowerCase() !== "a" || typing(e.target) || e.target.closest?.("dialog")) return;
    if (!people().length || scan.status !== "done") return;
    e.preventDefault();
    selectAll();
  });
  initChars();
  initRenames();
  initOutput();
  $("tgtInput").onchange = (e) => { openFiles(e.target.files); e.target.value = ""; };
  dropZone($("tgtDrop"), openFiles, true);
  dropZone(document.querySelector(".work"), openFiles, true);
  initPhotos();
  // drag a character onto a face
  $("charGrid").addEventListener("dragstart", (e) => {
    const t = e.target.closest?.(".char[data-key]");
    if (!t || busy || !charByKey(t.dataset.key)?.loaded) { e.preventDefault(); return; }
    e.dataTransfer.setData(CHAR_TYPE, t.dataset.key);
    e.dataTransfer.effectAllowed = "copy";
    e.dataTransfer.setDragImage(t.querySelector("img"), 20, 20);
  });
  charDrops($("people"));
  charDrops($("tlTracks"));
  // or onto one of a track's segments
  const canvas = $("tlCanvas");
  canvas.addEventListener("dragover", (e) => {
    if (!e.dataTransfer?.types.includes(CHAR_TYPE) || !timeline.partAtClient(e.clientX, e.clientY)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  });
  canvas.addEventListener("drop", (e) => {
    const hit = e.dataTransfer?.types.includes(CHAR_TYPE) && timeline.partAtClient(e.clientX, e.clientY);
    if (!hit) return;
    e.preventDefault();
    selectPart(hit.part);
    assign(null, e.dataTransfer.getData(CHAR_TYPE));
  });
  timeline.onSelect = clickPerson;
  timeline.onPart = (_, partId, how) => selectPart(partId, how);
  timeline.onBox = selectBox;
  timeline.onSplit = split;
  timeline.onJoin = join;
  timeline.onMerge = merge;
  timeline.onDelete = remove;
  timeline.onRemove = removeTrack;
  timeline.onMergeInto = mergeInto;
  timeline.canMerge = canMergeInto;
  timeline.onMove = moveTrack;
  // the scope card's ×, and Escape, go back to every face (in a text field, a name being changed,
  // Escape is the field's)
  $("scopeAll").onclick = () => selectPerson(null);
  addEventListener("keydown", (e) => {
    if (e.key !== "Escape" || textField(e.target)) return;
    if (picked.size && !typing(e.target) && !$("storageDlg").open && !outOpen()) selectPerson(null);
  });
  // the Remove buttons on the timeline's clip and the photos' header
  document.addEventListener("click", (e) => { if (e.target.closest(".tgt-clear") && !busy && !saving) { e.preventDefault(); clearTarget(); } });
  $("goBtn").onclick = go;
  $("stopBtn").onclick = () => abort?.abort();
  const cmp = $("compareBtn");
  cmp.addEventListener("pointerdown", () => compare(true));
  for (const ev of ["pointerup", "pointerleave", "pointercancel"]) cmp.addEventListener(ev, () => compare(false));
  $("reloadBtn").onclick = () => location.reload();
  $("serverReload").onclick = () => location.reload();
  addEventListener("pagehide", saveNow);
  workspace.status().then((s) => { pageBuild = s.build; }).catch(() => {});
  workspace.watchServer(showServer);
}

function fatal(title, text) {
  $("fatal").innerHTML = `<div><h2></h2><p></p></div>`;
  $("fatal").querySelector("h2").textContent = title;
  $("fatal").querySelector("p").textContent = text;
  $("fatal").hidden = false;
}

async function boot() {
  applyTheme();
  layout = initLayout($("app"));
  document.fonts?.ready.then(() => {
    layout.relayout();
    player?.invalidate();
    for (const el of document.querySelectorAll(".seg")) placeThumb(el);
  });
  initSteps($("steps"));
  player = new Player($("player"));
  timeline = initTimeline(player);
  $("tlAuto").onclick = () => setAuto(!ui.auto);
  renderAuto();
  viewport = initViewport($("canvasBox"), [$("view"), $("player")], () => drawMarks());
  renderTimeline();
  wire();
  addEventListener("input", (e) => { if (e.target.type === "range") fillRange(e.target); });
  render();
  await restoreLibrary();
  if (!navigator.gpu) {
    fatal("WebGPU isn't available", "This app needs WebGPU: a recent Chrome or Edge on a computer with a graphics card.");
    return;
  }
  let models;
  try {
    manifest = await fetchManifest(BASE);
    models = modelsThere(manifest, await workspace.modelFiles());
  } catch (e) {
    fatal("The local server isn't answering", `Start FaceJutsu again, then reload this page. (${e.message})`);
    return;
  }
  manifest = models.manifest;
  const occ = occluderInfo();
  if (occ) {
    toggle($("occlusionSeg"), [{ id: false, label: "Off" }, { id: true, label: "On" }], () => one("occlusion"), (v) => setSegment("occlusion", v), "Keep hands and glasses in front");
  }
  if (ui.enhance && !enhancerById(ui.enhance)) ui.enhance = "";
  seg($("enhanceSeg"), ["", ...enhancers().map((e) => e.id)].map((id) => ({ id, label: enhancerLabel(id) })), () => one("enhance"), (v) => setSegment("enhance", v));
  engine = await createEngine({ inPage: params.get("engine") === "page" });
  engine.onLost = gpuLost;
  initPreview();
  if (models.missing.length) await welcome(models.missing, models.extras);
  else await loadModels();
  render();
}

boot();
