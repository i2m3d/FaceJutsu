// What the app does with the pipeline: find the faces of a photo or video and tell the people
// apart, then swap the source face the user picked for each person into that person's faces. Runs
// in the engine worker (worker.js), or on the page as a fallback.
//
// Video goes through Mediabunny: it demuxes and decodes the input with WebCodecs, hands each
// frame to `process`, encodes the result and muxes it with the original audio into an MP4,
// written to the file the user picked as it's made (or built in memory without one).
//
// The preview (render) and the export (swapVideo) make a frame the same way (paint): the faces
// found there, each swapped with its segment's settings, as patches (patches.js) made once and
// kept, so the export only swaps what the preview hasn't shown yet.

import { SwapPipeline } from "./pipeline.js";
import { area, RECOGNIZE_112 } from "./face.js";
import { normalize, nearest } from "./tracks.js";
import { VideoScan, GAP, thumbnail } from "./scanner.js";
import { PatchCache, hash } from "./patches.js";
import { GpuQueue, PRIO } from "./queue.js";
import { openOutput } from "./workspace.js";
import { Input, Output, Conversion, ALL_FORMATS, BlobSource, StreamTarget, Mp4OutputFormat, WebMOutputFormat, QUALITY_MEDIUM, QUALITY_HIGH, QUALITY_VERY_HIGH, getFirstEncodableVideoCodec, VideoSampleSink, CanvasSink } from "./vendor/mediabunny.min.mjs";

const cancelled = () => new DOMException("Cancelled", "AbortError");

// each format's video codecs, the most widely supported first
const VIDEO_CODECS = { mp4: ["avc", "hevc", "av1"], webm: ["vp9", "av1"] };
const QUALITY = { medium: QUALITY_MEDIUM, high: QUALITY_HIGH, "very-high": QUALITY_VERY_HIGH };

const READ_PAD = 2; // px around a face's box that its swap may read
const DETAIL_SPAN = 8; // frames either side of one whose faces' own detail is their median (detailAt)

const brief = (f) => ({ box: f.box.map((v) => Math.round(v * 10) / 10), score: f.score });
const SCAN_UPDATE = 150; // ms between the export's updates of the faces it looked at
const WARM_W = 1280, WARM_H = 720; // px of the blank frame the models look at once loaded

// The patch of box [x1, y1, x2, y2) from its pixels before (a) and after (b) the swap (RGBA rows):
// the box around the pixels that changed. In each row, the pixels from the first that changed to
// the last are opaque; those either side are all zeros, so they take next to nothing on disk
// (qoi.js) and aren't written back (pipeline writeRect).
// -> { box, data } (an empty box when nothing changed)
function trimPatch([x1, y1, x2, y2], a, b) {
  const w = x2 - x1, h = y2 - y1;
  const same = (p) => a[p] === b[p] && a[p + 1] === b[p + 1] && a[p + 2] === b[p + 2];
  const starts = new Int32Array(h), ends = new Int32Array(h); // each row's [first, last + 1): empty when none changed
  let lx = w, ly = h, hx = -1, hy = -1;
  for (let y = 0; y < h; y++) {
    const row = y * w * 4;
    let s = 0, e = w - 1;
    while (s < w && same(row + s * 4)) s++;
    if (s === w) continue;
    while (same(row + e * 4)) e--;
    starts[y] = s;
    ends[y] = e + 1;
    if (s < lx) lx = s;
    if (e > hx) hx = e;
    if (y < ly) ly = y;
    hy = y;
  }
  if (hx < 0) return { box: [x1, y1, x1, y1], data: new Uint8ClampedArray(0) };
  const tw = hx - lx + 1, th = hy - ly + 1;
  const data = new Uint8ClampedArray(tw * th * 4);
  for (let y = 0; y < th; y++) {
    const s = starts[ly + y], e = ends[ly + y], row = (ly + y) * w * 4, to = y * tw * 4;
    if (s === e) continue; // (a row with nothing changed stays all zeros)
    data.set(b.subarray(row + s * 4, row + e * 4), to + (s - lx) * 4);
    for (let p = to + (s - lx) * 4 + 3; p < to + (e - lx) * 4; p += 4) data[p] = 255;
  }
  return { box: [x1 + lx, y1 + ly, x1 + hx + 1, y1 + hy + 1], data };
}

export class Session {
  // onLost({ reason, message }): the GPU device (or a WebNN context) was lost; nothing the session
  // has on the GPU works after that, so the page starts a new one
  constructor({ onLost } = {}) {
    this.pipe = new SwapPipeline({ onLost });
    this.sources = new Map(); // key -> pipeline source
    // the faces found in the photo or video to change (analyze): { id, kind, times, faces, ... }
    // (a video's as scanner.js has it); faces[i]: frame i's [{ box, kps, person }] (null: not
    // looked at yet), at times[i] seconds
    this.analysis = null;
    this.scan = null; // the VideoScan looking at the video's frames (a video's analysis)
    this.scanning = Promise.resolve(); // the scan's run going on
    // the photo or video the preview renders (openTarget): { key, kind, w, h, bitmap | input, sink, upright }
    this.target = null;
    this.patches = new PatchCache();
    this.queue = new GpuQueue();
  }

  // Runs fn with the GPU's frame work to itself: the preview's and the export's swaps share the
  // pipeline's buffers (the frame before the swap, for occlusion). prio: how wanted it is (PRIO).
  exclusive(fn, prio = PRIO.NOW) {
    return this.queue.run(fn, prio);
  }

  get gpuInfo() {
    const i = this.pipe.gpu?.info || {};
    return {
      name: [i.vendor, i.architecture, i.description].filter(Boolean).join(" "),
      peak: this.pipe.gpu?.pool.peak || 0,
      backend: this.pipe.selection?.backend || "webgpu",
      swapperBackend: this.pipe.selection?.swapperBackend || "webgpu",
      nnParts: this.pipe.selection?.nnParts || [],
      nnError: this.pipe.nnError || null,
    };
  }

  // (after any preview or export still running: they use the models) Once they're loaded, the
  // detector and the recognizer look at a blank frame, so their shaders are made before the first
  // faces are wanted (that takes seconds).
  async load(base, manifest, selection, opts) {
    await this.exclusive(() => this.pipe.load(base, manifest, selection, opts));
    this.exclusive(() => this.warm()).catch(() => {});
  }

  async warm() {
    const bitmap = await createImageBitmap(new ImageData(WARM_W, WARM_H));
    const frame = this.pipe.upload(bitmap, WARM_W, WARM_H);
    bitmap.close();
    try {
      await this.pipe.detect(frame);
      await this.pipe.embed(frame, RECOGNIZE_112.map(([x, y]) => [WARM_H * x + (WARM_W - WARM_H) / 2, WARM_H * y]));
    } finally {
      this.pipe.destroyFrame(frame);
    }
  }

  // ------------------------------------------------------------------ the faces to use

  // The largest face of an image a character is made from, with its normalized ArcFace embedding.
  // -> { face: { box, score, embedding } | null, count (faces found) }
  async embedImage(bitmap) {
    const frame = this.pipe.upload(bitmap, bitmap.width, bitmap.height);
    bitmap.close?.();
    try {
      const found = (await this.pipe.detect(frame)).sort((a, b) => area(b) - area(a));
      if (!found.length) return { face: null, count: 0 };
      const embedding = normalize(await this.pipe.embed(frame, found[0].kps));
      return { face: { ...brief(found[0]), embedding }, count: found.length };
    } finally {
      this.pipe.destroyFrame(frame);
    }
  }

  // Makes a character's identity (an embedding, the mean of its images') the source `key`.
  async setCharacter(key, embedding) {
    this.removeCharacter(key);
    const s = await this.pipe.createSource(Float32Array.from(embedding));
    s.id = await hash(JSON.stringify(Array.from(s.embedding))); // what the patches swapped with it are kept under
    this.sources.set(key, s);
  }

  removeCharacter(key) {
    const s = this.sources.get(key);
    if (!s) return;
    this.pipe.freeSource(s);
    this.sources.delete(key);
  }

  // ------------------------------------------------------------------ finding faces

  // Finds the faces of the photo (ImageBitmap) or video (File) to change and tells the people
  // apart. In a photo each face is a person. A video's are found a window at a time, around
  // opts.around seconds from its first frame shown (scanner.js), kept in opts.range ([first, last]
  // seconds) when given: a call goes on with analysis `id` (the faces found so far stay), and
  // ends once the window has been looked at. Faces are
  // followed from frame to frame and, with `recognize`, told apart by their ArcFace embeddings;
  // without it a face that leaves the picture comes back as a new person.
  // A photo's -> { people: [{ id, segments: [], frames, thumb (ImageBitmap) }], frames, boxes,
  //   times, ms, analysis (what restoreAnalysis takes to skip finding them again) }
  // A video's -> { coverage (of the window) }, the faces sent as they're found:
  //   onProgress({ scan (scanner.js update()), coverage })
  async analyze(id, input, { recognize = true, detectSize, around = 0, range, whole = true } = {}, { onProgress, signal } = {}) {
    if (input instanceof Blob) return this.scanVideo(id, input, { recognize, detectSize, around, range, whole }, { onProgress, signal });
    await this.dropScan();
    this.analysis = null;
    const t0 = performance.now();
    const r = await this.analyzeImage(input, { detectSize });
    const analysis = { recognize, t0: 0, ...r.analysis };
    this.analysis = { id, ...analysis };
    const boxes = r.analysis.faces.map((fs) => fs.map((f) => [...brief(f).box, f.person]));
    return { people: r.people, frames: 1, boxes, times: [0], ms: performance.now() - t0, analysis };
  }

  async scanVideo(id, file, { recognize, detectSize, around, range, whole }, { onProgress, signal }) {
    // one run at a time: the one before stops at its next frame
    const before = this.scanning;
    let done;
    this.scanning = new Promise((r) => { done = r; });
    try {
      await before;
      if (signal?.aborted) throw cancelled();
      await this.openScan(id, file, { recognize, detectSize });
      return await this.scan.run(around, { range, whole, signal, onUpdate: (scan, coverage) => onProgress?.({ scan, coverage }) });
    } finally {
      done();
    }
  }

  // the scan of analysis `id` of the video, opened (a new analysis when it's another)
  async openScan(id, file, { recognize = true, detectSize } = {}) {
    if (this.scan && this.analysis?.id === id && this.scan.a === this.analysis) return;
    const kept = this.analysis?.id === id && this.analysis.kind === "video" ? this.analysis : null;
    this.scan?.close();
    this.scan = null;
    this.analysis = null;
    const scan = await VideoScan.open(this.pipe, this.queue, file, kept, { recognize: kept ? kept.recognize : recognize, detectSize });
    scan.a.id = id;
    scan.fresh = !kept; // (the page has a restored one)
    this.scan = scan;
    this.analysis = scan.a;
  }

  // (after the run going on, which stops once the page cancels it)
  async dropScan() {
    await this.scanning;
    this.scan?.close();
    this.scan = null;
  }

  // Takes the faces analyze() found in this photo or video before (its result's analysis, or a
  // video's analysis as the page mirrors it) as analysis `id`.
  async restoreAnalysis(id, analysis) {
    await this.dropScan();
    this.analysis = { ...analysis, id };
  }

  async analyzeImage(bitmap, { detectSize }) {
    const frame = this.pipe.upload(bitmap, bitmap.width, bitmap.height);
    try {
      // people numbered left to right
      const found = (await this.pipe.detect(frame, { size: detectSize })).sort((a, b) => a.box[0] - b.box[0]);
      const people = [];
      for (const [i, f] of found.entries()) people.push({ id: i, segments: [], frames: 1, thumb: await thumbnail(bitmap, f) });
      const details = await this.pipe.faceDetails(frame, found.map((f) => f.kps));
      const faces = [found.map((f, i) => ({ box: f.box, kps: f.kps, person: i, detail: details[i] }))];
      return { people, analysis: { kind: "image", times: [0], faces } };
    } finally {
      bitmap.close?.();
      this.pipe.destroyFrame(frame);
    }
  }

  // The settings a person's face is swapped with at t seconds of media time: from the segment of
  // opts.plan[person] that covers it, the one starting last at or before t (the first one before
  // them all). -> { source, opts } or null when that face is kept
  settingsAt(person, t, opts) {
    const parts = opts.plan[person];
    if (!parts?.length) return null;
    const at = t - (this.analysis.t0 || 0) + 1e-3; // a segment starts on a frame's own time
    let part = parts[0];
    for (const p of parts) if (p.t0 <= at) part = p;
    const source = this.sources.get(part.source);
    if (!source?.styles) return null; // (its identity waits for the swapper)
    const { pixelshuffle, blur, enhance, occlusion, maskSize } = part;
    // (only when it changes the result: these options are part of what names a cached frame)
    return { source, opts: { pixelshuffle, blur, enhance, occlusion, ...(occlusion && maskSize ? { maskSize } : {}) } };
  }

  // Swaps the faces, each with its own settings: faces with the same settings go through the
  // pipeline together. faces: [{ box, kps, source, opts }]
  async swapPlanned(frame, faces) {
    const groups = new Map();
    for (const f of faces) {
      const k = JSON.stringify(f.opts);
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k).push(f);
    }
    for (const fs of groups.values()) await this.pipe.swapFaces(frame, fs, fs[0].opts);
  }

  // A face's own detail (the scan's, pipeline faceDetails) at analysed frame i: the median over
  // the frames within DETAIL_SPAN where its person has one, so the enhancer's restore doesn't
  // flicker with noise in the measure. rec: the face's record; null when none is known.
  detailAt(i, rec) {
    const a = this.analysis, v = [];
    for (let j = Math.max(0, i - DETAIL_SPAN); j <= Math.min(a.faces.length - 1, i + DETAIL_SPAN); j++)
      for (const r of a.faces[j] || []) if (r.person === rec.person && r.detail != null) v.push(r.detail);
    if (!v.length) return null;
    v.sort((x, y) => x - y);
    return v[v.length >> 1];
  }

  checkAnalysis(opts) {
    if (!this.analysis || this.analysis.id !== opts.analysis) throw new Error("The faces haven't been found yet.");
  }

  // ------------------------------------------------------------------ patches

  // The faces of analysed frame i that get a source, in the patches they make: faces whose boxes
  // (pipeline faceBox, with what they read around them) touch are swapped together, into one.
  // -> [{ box, faces: [{ kps, source, opts }], key }] for a w x h frame
  async patchesOf(i, opts, w, h) {
    const a = this.analysis, t = a.times[i];
    const faces = [];
    for (const f of a.faces[i] || []) {
      if (f.person < 0) continue;
      const s = this.settingsAt(f.person, t, opts);
      if (!s) continue;
      // several pixelshuffles (or "auto", older saves: all four): each face's own, from its size in this frame
      const b = s.opts.pixelshuffle;
      const o = typeof b === "number" ? s.opts : { ...s.opts, pixelshuffle: this.pipe.autoPixelshuffle(f.kps, b === "auto" ? undefined : b) };
      faces.push({ kps: f.kps, source: s.source, opts: o, detail: o.enhance?.detail ? this.detailAt(i, f) : null, area: this.pipe.faceBox(f.kps, o, w, h) });
    }
    const near = (p, q) => p[0] < q[2] + 2 * READ_PAD && q[0] < p[2] + 2 * READ_PAD && p[1] < q[3] + 2 * READ_PAD && q[1] < p[3] + 2 * READ_PAD;
    let groups = [];
    for (const f of faces) {
      // this face and every group its box touches become one
      const touching = groups.filter((g) => near(g.box, f.area));
      groups = groups.filter((g) => !touching.includes(g));
      const parts = [...touching, { box: f.area, faces: [f] }];
      groups.push({
        box: [0, 1, 2, 3].map((k) => (k < 2 ? Math.min : Math.max)(...parts.map((g) => g.box[k]))),
        faces: parts.flatMap((g) => g.faces).sort((p, q) => faces.indexOf(p) - faces.indexOf(q)),
      });
    }
    return Promise.all(groups.filter((g) => g.box[2] > g.box[0] && g.box[3] > g.box[1]).map(async (g) => {
      const key = await hash(JSON.stringify([this.target.key, t, w, h, g.faces.map((f) => [this.pipe.configFor(f.opts), f.kps, f.source.id, f.opts, f.detail])]));
      return { ...g, key: `${this.target.folder}/${key}` };
    }));
  }

  // Puts the patches of `units` (patchesOf) onto frame, in place: swaps the ones not made yet
  // and keeps them; with apply, the ones made before are copied onto it too (else it's left
  // without them). -> a promise of every unit's patch { box, data } (trimPatch)
  async paint(frame, units, { apply = true } = {}) {
    const have = await Promise.all(units.map((u) => this.patches.get(u.key)));
    const todo = units.filter((_, k) => !have[k]);
    const before = todo.map((u) => this.pipe.readRect(frame, u.box));
    for (const u of todo) await this.swapPlanned(frame, u.faces);
    const after = todo.map((u) => this.pipe.readRect(frame, u.box));
    if (apply) units.forEach((u, k) => { if (have[k]) this.pipe.writeRect(frame, have[k].box, have[k].data); });
    const made = await Promise.all(todo.map(async (u, k) => {
      const [a, b] = await Promise.all([before[k], after[k]]);
      const patch = trimPatch(u.box, a, b);
      this.patches.put(u.key, patch);
      return patch;
    }));
    return units.map((u, k) => have[k] || made[todo.indexOf(u)]);
  }

  // ------------------------------------------------------------------ preview

  // The photo (ImageBitmap) or video (File) the preview renders, kept under `key` (its content's
  // hash: what its patches are kept under).
  openTarget(key, input) {
    return this.exclusive(() => this.open(key, input));
  }

  async open(key, input) {
    this.closeTarget();
    // (its folder in the cache: the key may not be a hash, for a file that couldn't be hashed)
    const folder = (await hash(key)).slice(0, 32);
    if (!(input instanceof Blob)) {
      this.target = { key, folder, kind: "image", w: input.width, h: input.height, bitmap: input };
      return;
    }
    const src = new Input({ source: new BlobSource(input), formats: ALL_FORMATS });
    const track = await src.getPrimaryVideoTrack();
    if (!track) { src.dispose(); throw new Error("This file has no video track."); }
    // upright, as analyze and the export see it
    const upright = !!((await track.getRotation()) || (await track.getFlip?.()));
    this.target = { key, folder, kind: "video", w: track.displayWidth, h: track.displayHeight, input: src, upright, track };
  }

  // Tells the cache what the target's frames are being swapped with now, when that changed: who is
  // swapped and how (not who else has been found, which doesn't change a frame).
  notePlan(plan) {
    const swapped = Object.entries(plan).filter(([, parts]) => parts.some((q) => q.source));
    return swapped.length ? this.patches.setPlan(this.target.folder, JSON.stringify(swapped)) : null;
  }

  closeTarget() {
    const t = this.target;
    this.target = null;
    t?.bitmap?.close();
    t?.input?.dispose();
  }

  // the target's frames from `start` up to `end` seconds: { src, timestamp, close() }
  async *targetFrames(start, end) {
    const t = this.target;
    if (t.kind === "image") { yield { src: t.bitmap, timestamp: 0, close() {} }; return; }
    if (t.upright) {
      // (a decoder and canvases of its own: two calls may decode side by side)
      for await (const c of new CanvasSink(t.track, { poolSize: 2 }).canvases(start, end)) yield { src: c.canvas, timestamp: c.timestamp, close() {} };
      return;
    }
    for await (const s of new VideoSampleSink(t.track).samples(start, end)) {
      const vf = s.toVideoFrame();
      yield { src: vf, timestamp: s.timestamp, close() { vf.close(); s.close(); } };
    }
  }

  // Makes the patches of analysed frames `frames` (indices, ascending) of the target: the ones
  // kept, and the rest swapped. onFrame({ i, patches: [{ box, bitmap }] }) for each frame as it's
  // done, the frames with every patch kept first; with opts.send false, only made and kept
  // (onFrame({ i, patches: null })); with opts.swap false, only the kept ones are sent, the rest
  // reported (onFrame({ i, missing: true })). The kept ones don't wait for the GPU: playing what's
  // made goes on while another call swaps.
  async render(frames, opts, { onFrame, signal } = {}) {
    this.checkAnalysis(opts);
    const tg = this.target, a = this.analysis;
    if (!tg) throw new Error("No photo or video to preview.");
    await this.notePlan(opts.plan);
    const send = async (i, patches) => {
      if (opts.send === false) { onFrame?.({ i, patches: null }); return; }
      const out = await Promise.all(patches.filter((p) => p.box[2] > p.box[0] && p.box[3] > p.box[1]).map(async (p) => ({
        box: p.box,
        bitmap: await createImageBitmap(new ImageData(p.data, p.box[2] - p.box[0], p.box[3] - p.box[1]), { premultiplyAlpha: "none", colorSpaceConversion: "none" }),
      })));
      onFrame?.({ i, patches: out });
    };
    const todo = new Map(); // analysed frame -> its units, when some aren't made yet
    // every frame's patches looked up at once (reading from disk side by side), sent in order
    const looks = frames.map(async (i) => {
      const units = await this.patchesOf(i, opts, tg.w, tg.h);
      const have = opts.send === false
        ? await Promise.all(units.map(async (u) => (await this.patches.has(u.key)) && u))
        : await Promise.all(units.map((u) => this.patches.get(u.key)));
      return { i, units, have };
    });
    looks.forEach((l) => l.catch(() => {})); // (the ones not awaited after a cancel)
    for (const look of looks) {
      if (signal?.aborted) throw cancelled();
      const { i, units, have } = await look;
      if (have.every(Boolean)) await send(i, have);
      else todo.set(i, units);
    }
    if (opts.swap === false) {
      for (const i of todo.keys()) onFrame?.({ i, missing: true });
      return;
    }
    if (!todo.size) return;
    // the frames to show are swapped with the faces around the playhead being found, taking
    // turns (claiming the GPU from the rest, between frames too); the background's after the rest
    const prio = opts.send === false ? PRIO.BACKGROUND : PRIO.NEAR;
    const release = prio === PRIO.NEAR ? this.queue.claim(prio) : null;
    const list = [...todo.keys()];
    let frame = null;
    try {
      for await (const item of this.targetFrames(a.times[list[0]], a.times[list[list.length - 1]] + 1e-4)) {
        try {
          if (signal?.aborted || this.target !== tg) throw cancelled();
          const i = nearest(a.times, item.timestamp);
          const units = todo.get(i);
          if (!units) continue;
          todo.delete(i);
          const patches = await this.exclusive(() => {
            if (signal?.aborted || this.target !== tg) throw cancelled();
            frame = this.pipe.upload(item.src, tg.w, tg.h, frame);
            return this.paint(frame, units, { apply: false });
          }, prio);
          await send(i, patches);
        } finally {
          item.close();
        }
        if (!todo.size) break;
      }
    } finally {
      release?.();
      await this.exclusive(() => {
        if (frame) this.pipe.destroyFrame(frame);
        this.pipe.gpu.pool.trim();
      });
    }
  }

  // -> { bytes, count } of the patches kept on disk
  clearPatches() { return this.patches.clear(); }

  // ------------------------------------------------------------------ export

  // Swaps every frame of the video file (the target, openTarget) into a new video, or the frames
  // from opts.trim.start up to opts.trim.end (media seconds): the patches the preview made, and
  // the rest swapped now. Written to the outputs folder as it's made, as `name`.
  // this browser can't encode it), quality (QUALITY's), size (px on the short side to downscale
  // to, 0 for the original), audio (keep it) }. onProgress({ frac, frames, fps });
  // onPreview(ImageBitmap) for each swapped frame, skipping frames while the last one is still
  // being copied out.
  // -> { saved: { name, path }, size (bytes), frames, ms, width, height, codec }
  // The frames no scan has looked at are looked at as the export reaches them:
  // onProgress({ scan }) with what's found (scanner.js update()). (After the scan's run going on:
  // the page cancels it first.)
  async swapVideo(file, opts, { name, out, onProgress, onPreview, signal } = {}) {
    this.checkAnalysis(opts);
    if (this.target?.kind !== "video") throw new Error("No video to export.");
    await this.scanning;
    await this.openScan(this.analysis.id, file);
    return this.exclusive(() => this.exportVideo(file, opts, { name, out, onProgress, onPreview, signal }));
  }

  // The output is a part file until it's closed, so a failed or stopped export leaves nothing.
  async exportVideo(file, opts, { name, ...hooks }) {
    await this.notePlan(opts.plan);
    const writable = await openOutput(name, hooks.signal);
    try {
      const r = await this.encodeVideo(file, opts, writable, hooks);
      return { ...r, saved: await writable.close() };
    } catch (e) {
      await writable.abort().catch(() => {});
      throw e;
    }
  }

  async encodeVideo(file, opts, writable, { out = {}, onProgress, onPreview, signal }) {
    const { format = "mp4", quality = "very-high", size: short = 0, audio = true } = out;
    const pipe = this.pipe, a = this.analysis;
    const { device } = pipe.gpu;
    const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS });
    const track = await input.getPrimaryVideoTrack();
    if (!track) throw new Error("This file has no video track.");
    if (!(await track.canDecode())) throw new Error("This browser can't decode this video's format.");
    const w = track.displayWidth;
    const h = track.displayHeight;
    // downscaled after the swap (the faces were found at full size), to even sizes for the codecs
    const k = short && short < Math.min(w, h) ? short / Math.min(w, h) : 1;
    const ow = k < 1 ? 2 * Math.round((w * k) / 2) : w, oh = k < 1 ? 2 * Math.round((h * k) / 2) : h;
    const codecs = [...new Set([out.codec, ...VIDEO_CODECS[format]].filter(Boolean))];
    const codec = await getFirstEncodableVideoCodec(codecs, { width: ow, height: oh });
    if (!codec) throw new Error(`This browser can't encode ${ow}x${oh} video as ${format.toUpperCase()}.`);
    // written as it's made: an MP4's index goes at the end, so nothing waits in memory for it; the
    // stream given to Mediabunny only writes, and never closes the file
    let size = 0;
    const output = new Output({
      format: format === "webm" ? new WebMOutputFormat() : new Mp4OutputFormat({ fastStart: false }),
      target: new StreamTarget(new WritableStream({
        write: (chunk) => {
          size = Math.max(size, chunk.position + chunk.data.byteLength);
          return writable.write(chunk);
        },
      }), { chunked: true }),
    });
    const canvas = new OffscreenCanvas(w, h);
    const ctx = canvas.getContext("webgpu");
    ctx.configure({ device, format: "rgba8unorm", usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT, alphaMode: "opaque" });
    const small = k < 1 ? new OffscreenCanvas(ow, oh) : null;
    const sctx = small?.getContext("2d");
    if (sctx) sctx.imageSmoothingQuality = "high";

    let frame = null;
    let frames = 0;
    let previewing = false;
    const t0 = performance.now();
    // the frames not looked at yet are, in order, going on from the faces before them
    const scan = this.scan, pass = scan.pass();
    let seeded = false, sent = t0;
    const sendScan = () => {
      const u = scan.update();
      if (u) onProgress?.({ scan: u });
      sent = performance.now();
    };
    const conversion = await Conversion.init({
      input,
      output,
      tracks: "primary",
      trim: opts.trim || undefined,
      video: {
        codec,
        quality: QUALITY[quality],
        forceTranscode: true,
        allowTransformationMetadata: false, // rotate phone videos upright, so faces are upright
        // (not width/height: those would resize the frame before the swap sees it)
        processedWidth: ow,
        processedHeight: oh,
        process: async (sample) => {
          if (signal?.aborted) throw cancelled();
          const vf = sample.toVideoFrame();
          frame = pipe.upload(vf, vf.displayWidth, vf.displayHeight, frame);
          // (a trimmed export's timestamps start at 0)
          const i = nearest(a.times, sample.timestamp + (opts.trim?.start ?? 0));
          try {
            if (!seeded) {
              for (let j = Math.max(0, i - GAP); j < i; j++) if (a.state[j] === 2) pass.known(j);
              seeded = true;
            }
            if (a.state[i] === 2) pass.known(i);
            else {
              await pass.look(i, frame, vf);
              await scan.measured(i); // (its faces' detail, for their swap below)
            }
          } finally {
            vf.close();
          }
          if (performance.now() - sent > SCAN_UPDATE) sendScan();
          const units = await this.patchesOf(i, opts, frame.w, frame.h);
          if (units.length) await this.paint(frame, units);
          pipe.present(frame, ctx);
          sctx?.drawImage(canvas, 0, 0, ow, oh);
          const done = small || canvas;
          frames++;
          if (onPreview && !previewing) {
            previewing = true;
            const snap = new VideoFrame(done, { timestamp: 0 });
            createImageBitmap(snap).then(onPreview).finally(() => { snap.close(); previewing = false; });
          }
          return done;
        },
      },
      // kept audio is copied where the format takes it; WebM's is Opus
      audio: !audio ? { discard: true } : format === "webm" ? { codec: "opus" } : undefined,
    });
    const lost = conversion.discardedTracks.filter((d) => d.reason !== "discarded_by_user");
    if (!conversion.isValid || lost.length) {
      throw new Error("This video can't be converted: " + lost.map((d) => `${d.track.type} track (${d.reason})`).join(", ") + (lost.some((d) => d.track.type === "audio") ? ". Output → Audio: Remove exports it without its audio." : ""));
    }
    conversion.onProgress = (p) => onProgress?.({ frac: p, frames, fps: frames / ((performance.now() - t0) / 1000) });
    const stop = () => conversion.cancel();
    signal?.addEventListener("abort", stop, { once: true });
    try {
      await conversion.execute();
      if (signal?.aborted) throw cancelled();
    } catch (e) {
      if (signal?.aborted) throw cancelled();
      throw e;
    } finally {
      signal?.removeEventListener("abort", stop);
      sendScan();
      if (frame) pipe.destroyFrame(frame);
      ctx.unconfigure();
      pipe.gpu.pool.trim();
    }
    return { size, frames, ms: performance.now() - t0, width: ow, height: oh, codec };
  }
}
