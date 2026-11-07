// Face swap pipeline: the models folder's files -> WebGPU models -> detect, align, embed, swap and
// paste back (optionally keeping what covers the face, and restoring the face with an enhancer),
// entirely on the GPU. Frames stay on the GPU between steps; only the detector's head outputs (to
// decode boxes) and the identity embedding come back to the CPU.

import { GPU } from "./gpu/device.js";
import * as ops from "./gpu/ops.js";
import { SafeTensors, destroyWeights } from "./weights.js";
import { SCRFD } from "./models/scrfd.js";
import { ArcFace } from "./models/arcface.js";
import { Inswapper } from "./models/inswapper.js";
import { FaceEnhancer } from "./models/enhancer.js";
import { XSeg } from "./models/xseg.js";
import { WebNNInswapper } from "./webnn/inswapper.js";
import { WebNNSCRFD } from "./webnn/scrfd.js";
import { WebNNXSeg } from "./webnn/xseg.js";
import * as face from "./face.js";
import { readModel, untilBack } from "./workspace.js";

export async function fetchManifest(baseUrl) {
  const res = await fetch(new URL("manifest.json", baseUrl), { cache: "no-cache" });
  if (!res.ok) throw new Error(`could not load ${new URL("manifest.json", baseUrl)} (${res.status}).`);
  const m = await res.json();
  if (m.version !== 1) throw new Error("models/manifest.json is from a different build.");
  return m;
}

// The face swappers a manifest offers: [{ id (component), model, label, size (crop px) }].
// "swapper" is inswapper_128 (older manifests don't describe it).
export function listSwappers(manifest) {
  return Object.entries(manifest.components)
    .filter(([id, c]) => id === "swapper" || c.swapper)
    .map(([id, c]) => ({ id, model: "inswapper", label: "inswapper_128", size: 128, ...c.swapper }));
}

// The face enhancers a manifest offers: [{ id (component), model, label, size (crop px), template }]
export function listEnhancers(manifest) {
  return Object.entries(manifest.components).filter(([, c]) => c.enhancer).map(([id, c]) => ({ id, ...c.enhancer }));
}

// The occlusion masks a manifest offers: [{ id (component), model, label, size (crop px), template }]
export function listOccluders(manifest) {
  return Object.entries(manifest.components).filter(([, c]) => c.occluder).map(([id, c]) => ({ id, ...c.occluder }));
}

// selection: { detector, recognizer, swapper, enhancer, occluder } precisions (or lists of
// precisions, the first one built is taken; no swapper: not used), model: the swapper component
// (default "swapper"),
// enhancerModels / occluderModel: the enhancer components (any number, each its own part,
// "enhancer:<component>"; enhancerModel: one) / the occlusion mask component (none: not used)
// -> manifest file entries, by part
export function resolveFiles(manifest, selection) {
  const files = {};
  const optional = { swapper: selection.swapper && (selection.model || "swapper"), occluder: selection.occluderModel };
  for (const id of [].concat(selection.enhancerModels || selection.enhancerModel || [])) optional[`enhancer:${id}`] = id;
  for (const k of ["detector", "recognizer", ...Object.keys(optional).filter((k) => optional[k])]) {
    const id = optional[k] || k;
    const c = manifest.components[id];
    const want = [].concat(selection[partKind(k)]);
    const p = want.find((q) => c?.builds[q]);
    if (!p) throw new Error(`no ${want.join(" / ")} build of the ${id}`);
    files[k] = { ...c.builds[p], precision: p };
  }
  return files;
}

const SWAPPERS = { inswapper: Inswapper };
const NN_SWAPPERS = { inswapper: WebNNInswapper };
// a part's kind: "enhancer" for each enhancer's part ("enhancer:<component>")
const partKind = (k) => k.split(":")[0];
const LOADERS = { detector: SCRFD, recognizer: ArcFace, swapper: Inswapper, enhancer: FaceEnhancer, occluder: XSeg };
const LABELS = { detector: "face detector", recognizer: "face recognizer", swapper: "face swapper", enhancer: "face enhancer", occluder: "occlusion mask" };
// faces per occlusion mask pass (the largest map is 16 MiB per face)
const OCCLUDER_BATCH = 4;

// What this.loaded records for a component: the file path, tagged with how the detector, the
// swapper, the enhancer and the occlusion mask run
const loadedKey = (files, k, backend, winograd) => (k === "swapper" || k === "occluder" ? `${files[k].path}#${backend}${winograd ? "+winograd" : ""}`
  : k === "detector" ? `${files[k].path}#${backend}` : partKind(k) === "enhancer" ? `${files[k].path}${winograd ? "#winograd" : ""}` : files[k].path);

export class SwapPipeline {
  // gpuOptions: passed to GPU.create (e.g. { profile: true }); its onLost
  // also hears of a lost WebNN context
  constructor(gpuOptions = {}) {
    this.gpuOptions = gpuOptions;
    this.gpu = null;
    this.loaded = {}; // component -> path of the file on the GPU
    this.fileDates = {}; // component -> when that file was last changed
    this.masks = new Map(); // crop size + blur -> mask profile tensor
    this.occBlurs = new Map(); // edge softness -> the occlusion mask's Gaussian taps { buf, r }
    this.orig = null; // the frame before the swap (occlusion), reused while frames keep their size
    this.source = null; // the default identity (setSource)
    this.sources = new Set(); // every identity made by createSource: { embedding, latent, styles, stylesData }
    this.contexts = new Set(); // the WebNN contexts in use, whose loss is reported
  }

  // Reads the selected files from the models folder and loads them; components already on the GPU
  // with the same file are kept, and those left out of it go (the detector and the recognizer
  // are always in it). selection.backend "webnn" also runs the swapper, the detector and
  // the occlusion mask through WebNN (each falling back to WebGPU, with this.nnError set, when that fails;
  // selection.webnnInterop: true shares the WebGPU device with the WebNN swapper, experimental);
  // selection.winograd: false runs the swapper's, the enhancer's and the occlusion mask's 3x3
  // convs directly on WebGPU (less memory, slower). The enhancer (selection.enhancerModel)
  // always runs on WebGPU.
  async load(baseUrl, manifest, selection, { onStatus = () => {}, signal } = {}) {
    this.gpu = this.gpu || (await GPU.create(this.gpuOptions));
    const want = resolveFiles(manifest, selection);
    const drop = Object.keys(this.loaded).filter((k) => !want[k] && this[k]);
    for (const k of drop) this.unloadPart(k);
    // the working buffers the dropped models used go too (they come back as they're needed)
    if (drop.length) this.gpu.pool.trim();
    // each enhancer's { id, model, label, size, template }, by component
    this.enhancerInfos = Object.fromEntries(listEnhancers(manifest).map((e) => [e.id, e]));
    this.occluderInfo = selection.occluderModel ? listOccluders(manifest).find((e) => e.id === selection.occluderModel) : null;
    const model = listSwappers(manifest).find((s) => s.id === (selection.model || "swapper"))?.model || "inswapper";
    const loaders = { ...LOADERS, swapper: SWAPPERS[model] };
    const backend = selection.backend === "webnn" ? "webnn" : "webgpu";
    // a swapper without a WebNN graph runs on WebGPU with either backend, the enhancer always
    const runsOn = (k) => ((k === "swapper" && !NN_SWAPPERS[model]) || partKind(k) === "enhancer" ? "webgpu" : backend);
    const winograd = selection.winograd !== false;
    const parts = Object.keys(want).filter((k) => this.loaded[k] !== loadedKey(want, k, runsOn(k), winograd));
    const files = {};
    for (const k of parts) {
      // (the file as it is now: without this the browser's own cache can go on giving the one it
      // replaced, for days)
      const url = new URL(want[k].path, baseUrl);
      const read = async () => {
        const res = await readModel(url, { signal, cache: "no-store" });
        if (!res.ok) throw new Error(`The model file ${want[k].path} couldn't be read (${res.status}).`);
        files[k] = await res.arrayBuffer();
        this.fileDates[k] = res.headers.get("last-modified");
      };
      // (from the local server, it waits if that has stopped; elsewhere, not reaching it is an error)
      await (url.origin === self.location.origin ? untilBack(read, signal) : read());
    }
    if (parts.includes("swapper") || parts.includes("detector") || parts.includes("occluder")) this.nnError = null;
    for (const k of parts) {
      this.unloadPart(k);
      onStatus({ phase: "load", part: k, what: LABELS[partKind(k)], frac: 0 });
      const st = new SafeTensors(files[k]);
      delete files[k]; // (it goes once this model is loaded, not after all of them)
      const progress = (f) => onStatus({ phase: "load", part: k, what: LABELS[partKind(k)], frac: f });
      const nn = k === "swapper" && runsOn(k) === "webnn";
      // with WebNN, the WebGPU swapper keeps the direct weights: the WebNN graph's filters are
      // prepared from them, and they are the fallback
      this[k] = await loaders[partKind(k)].load(this.gpu, st, progress, { winograd: winograd && !nn });
      if (nn) {
        const what = "face swapper (WebNN)";
        onStatus({ phase: "load", part: k, what, frac: 0 });
        try {
          this.nn = await NN_SWAPPERS[model].load({ gpu: this.gpu, weights: this.swapper.w, st }, { interop: selection.webnnInterop === true });
          // build the single-face graph now, so a WebNN that can't run this model fails here
          await this.nn.ensureGraph(1, (f) => onStatus({ phase: "load", part: k, what, frac: f }));
          this.watch(this.nn.context);
        } catch (e) {
          console.warn("WebNN swapper unavailable, using WebGPU:", e);
          this.nnError = e.message || String(e);
          this.nn?.dispose();
          this.nn = null;
          if (winograd) {
            destroyWeights(this.swapper.w);
            this.swapper = await loaders.swapper.load(this.gpu, st, progress, { winograd });
          }
        }
      }
      if (k === "detector" && backend === "webnn") {
        // the WebGPU detector stays loaded as the fallback (8 MB)
        try {
          this.nnDetector = await WebNNSCRFD.load(st);
          await this.nnDetector.ensureGraph(face.DET_SIZE);
          this.watch(this.nnDetector.context);
        } catch (e) {
          console.warn("WebNN detector unavailable, using WebGPU:", e);
          this.nnError = e.message || String(e);
          this.nnDetector?.dispose();
          this.nnDetector = null;
        }
      }
      if (k === "occluder" && backend === "webnn") {
        // the WebGPU occlusion mask stays loaded as the fallback
        const what = "occlusion mask (WebNN)";
        onStatus({ phase: "load", part: k, what, frac: 0 });
        try {
          this.nnOccluder = await WebNNXSeg.load(this.gpu, st);
          await this.nnOccluder.ensureGraph((f) => onStatus({ phase: "load", part: k, what, frac: f }));
          this.watch(this.nnOccluder.context);
        } catch (e) {
          console.warn("WebNN occlusion mask unavailable, using WebGPU:", e);
          this.nnError = e.message || String(e);
          this.nnOccluder?.dispose();
          this.nnOccluder = null;
        }
      }
      const runs = (k === "swapper" && !this.nn) || (k === "detector" && !this.nnDetector) || (k === "occluder" && !this.nnOccluder) ? "webgpu" : backend;
      this.loaded[k] = loadedKey(want, k, runs, winograd);
    }
    // the swapper's styles depend on its weights
    if (parts.includes("swapper")) for (const s of this.sources) await this.buildSource(s);
    await this.gpu.sync();
    // nnParts: what runs through WebNN
    const nnParts = [this.nn && "face swapper", this.nnDetector && "face detector", this.nnOccluder && "occlusion mask"].filter(Boolean);
    this.selection = { ...selection, backend: nnParts.length ? "webnn" : "webgpu", swapperBackend: this.nn ? "webnn" : "webgpu", nnParts };
    onStatus({ phase: "ready" });
  }

  // A WebNN context is lost as a device is (the GPU process crashing, a driver reset): told to
  // gpuOptions.onLost, unless the pipeline let go of it (destroying a context loses it too).
  watch(context) {
    this.contexts.add(context);
    context.lost?.then((e) => {
      if (this.contexts.delete(context)) this.gpuOptions.onLost?.({ reason: "webnn", message: e?.message || "WebNN context lost" });
    });
  }

  // disposes of a WebNN model (nn, nnDetector, nnOccluder)
  drop(m) {
    if (!m) return;
    this.contexts.delete(m.context);
    m.dispose();
  }

  unloadPart(k) {
    // dispatches recorded but not yet submitted (a new source's style rows) may still read the
    // weights: submit them first, destroy waits for submitted work
    this.gpu?.flush();
    if (this[k]) destroyWeights(this[k].w);
    if (k === "swapper") {
      for (const s of this.sources) this.freeSource(s, true);
      this.drop(this.nn);
      this.nn = null;
    }
    if (k === "detector") {
      this.drop(this.nnDetector);
      this.nnDetector = null;
    }
    if (k === "occluder") {
      this.drop(this.nnOccluder);
      this.nnOccluder = null;
    }
    this[k] = null;
    delete this.loaded[k];
  }

  // ------------------------------------------------------------------ frames

  // Uploads an image source (ImageBitmap, VideoFrame, canvas, ...) as a GPU frame. Pass `reuse`
  // (a frame of the same size) to overwrite it instead of allocating.
  upload(source, w, h, reuse = null) {
    const { device } = this.gpu;
    const frame = reuse && reuse.w === w && reuse.h === h ? reuse : this.createFrame(w, h);
    device.queue.copyExternalImageToTexture({ source }, { texture: frame.tex }, [w, h]);
    this.gpu.encode((enc) => enc.copyTextureToBuffer({ texture: frame.tex }, { buffer: frame.buf, bytesPerRow: frame.stride * 4 }, [w, h]));
    return frame;
  }

  createFrame(w, h) {
    const { device } = this.gpu;
    const stride = Math.ceil(w / 64) * 64; // rows 256-byte aligned for buffer <-> texture copies
    return {
      w, h, stride,
      buf: device.createBuffer({ size: stride * h * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }),
      tex: device.createTexture({ size: [w, h], format: "rgba8unorm", usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC | GPUTextureUsage.RENDER_ATTACHMENT }),
    };
  }

  destroyFrame(frame) {
    frame.buf.destroy();
    frame.tex.destroy();
  }

  // RGBA bytes [w*h*4] of a frame
  async read(frame) {
    this.gpu.flush();
    const { device } = this.gpu;
    const bytes = frame.stride * frame.h * 4;
    const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.gpu.encode((enc) => enc.copyBufferToBuffer(frame.buf, 0, staging, 0, bytes));
    await staging.mapAsync(GPUMapMode.READ);
    this.gpu.checkErrors();
    const src = new Uint8Array(staging.getMappedRange());
    const out = new Uint8ClampedArray(frame.w * frame.h * 4);
    for (let y = 0; y < frame.h; y++) out.set(src.subarray(y * frame.stride * 4, (y * frame.stride + frame.w) * 4), y * frame.w * 4);
    staging.unmap();
    staging.destroy();
    return out;
  }

  // RGBA bytes [w*h*4] of the box [x1, y1, x2, y2) of a frame. The copy is recorded now, so later
  // work on the frame doesn't change what comes back.
  async readRect(frame, [x1, y1, x2, y2]) {
    const { device } = this.gpu;
    const row = frame.stride * 4, w = x2 - x1, h = y2 - y1;
    const staging = device.createBuffer({ size: row * h, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.gpu.encode((enc) => enc.copyBufferToBuffer(frame.buf, y1 * row, staging, 0, row * h));
    try {
      await staging.mapAsync(GPUMapMode.READ);
      const src = new Uint8Array(staging.getMappedRange());
      const out = new Uint8ClampedArray(w * h * 4);
      for (let y = 0; y < h; y++) out.set(src.subarray(y * row + x1 * 4, y * row + x2 * 4), y * w * 4);
      staging.unmap();
      return out;
    } finally {
      staging.destroy();
    }
  }

  // Writes a patch's RGBA bytes [w*h*4] (session.js trimPatch) into the box [x1, y1, x2, y2) of a
  // frame: of each row, from its first opaque pixel to its last.
  writeRect(frame, [x1, y1, x2, y2], data) {
    // queue writes run before work still being recorded: submit that first
    this.gpu.flush();
    const { queue } = this.gpu.device;
    const w = x2 - x1;
    for (let y = y1; y < y2; y++) {
      const row = (y - y1) * w * 4;
      let s = 0, e = w;
      while (s < w && !data[row + s * 4 + 3]) s++;
      if (s === w) continue;
      while (!data[row + (e - 1) * 4 + 3]) e--;
      queue.writeBuffer(frame.buf, (y * frame.stride + x1 + s) * 4, data, row + s * 4, (e - s) * 4);
    }
  }

  // Copies a frame into a canvas context configured with format "rgba8unorm" and COPY_DST usage.
  present(frame, ctx) {
    this.gpu.encode((enc) => enc.copyBufferToTexture({ buffer: frame.buf, bytesPerRow: frame.stride * 4 }, { texture: ctx.getCurrentTexture() }, [frame.w, frame.h]));
  }

  // ------------------------------------------------------------------ steps

  detectInput(frame, size = face.DET_SIZE) {
    if (size % 32) throw new Error(`detector size ${size} is not a multiple of 32`);
    const lb = face.letterbox(frame.w, frame.h, size);
    const x = ops.resample(this.gpu, frame, { ow: size, oh: size, m: lb.m, vw: lb.vw, vh: lb.vh, mul: 1 / 128, add: -127.5 / 128, fill: -127.5 / 128 });
    return { x, scale: lb.scale };
  }

  // opts: { size (detector input, default 640), threshold, nms }
  async detect(frame, opts = {}) {
    const size = opts.size || face.DET_SIZE;
    const { x, scale } = this.detectInput(frame, size);
    let data;
    if (this.nnDetector) {
      // WebNN: the letterboxed input goes through the CPU, in f16 when the graph runs f16
      const input = this.nnDetector.ct === "float16" ? await ops.readHalf(this.gpu, x) : await this.gpu.read(x);
      data = await this.nnDetector.forward(input, size);
      x.release();
    } else {
      const outs = this.detector.forward(x);
      x.release();
      data = await Promise.all(outs.map((o) => this.gpu.read(o)));
      for (const o of outs) o.release();
    }
    return face.decodeDetections(data, scale, { ...opts, size });
  }

  embedInput(frame, kps) {
    const m = face.alignment(kps, face.RECOGNIZE_112, 112);
    return ops.resample(this.gpu, frame, { ow: 112, oh: 112, m: face.invert(m), mul: 1 / 127.5, add: -1 });
  }

  // Each face's own detail in the frame (face.faceDetail: px across the swapper's framing up to
  // which it has detail), for the faces with keypoints kpsList: their spectra on the GPU, one read.
  async faceDetails(frame, kpsList) {
    if (!kpsList.length) return [];
    const N = face.DETAIL_CROP, n = N / 2 + 1;
    const x = this.gpu.empty([kpsList.length, N, N, 3]);
    kpsList.forEach((kps, i) => ops.resample(this.gpu, frame, { ow: N, oh: N, m: face.invert(face.alignment(kps, face.SWAP_128, N)), mul: 1 / 255, out: x, offset: i * N * N * 3 }));
    const r = ops.faceSpectrum(this.gpu, x);
    x.release();
    const s = await this.gpu.read(r);
    r.release();
    return kpsList.map((_, i) => face.faceDetail(s.subarray(i * n, (i + 1) * n)));
  }

  async embed(frame, kps) {
    const x = this.embedInput(frame, kps);
    const e = this.recognizer.forward(x);
    x.release();
    const out = await this.gpu.read(e);
    e.release();
    return out;
  }

  // An identity to swap in (an ArcFace embedding of a source face). A face of swapFaces takes
  // it as f.source; its swapper inputs are rebuilt when the swapper's weights change.
  async createSource(embedding) {
    const s = { embedding };
    this.sources.add(s);
    await this.buildSource(s);
    return s;
  }

  // (without the swapper, once it's loaded)
  async buildSource(s) {
    if (!this.swapper) return;
    s.latent = this.swapper.latent(s.embedding);
    s.styles = this.swapper.styles(s.latent);
    // the WebNN graph's per-source input (inswapper: the style rows)
    s.stylesData = this.nn ? await this.nn.sourceData(s.latent, s.styles) : null;
  }

  // keepEmbedding: only the swapper's inputs go (its weights are being replaced)
  freeSource(s, keepEmbedding = false) {
    for (const t of s.styles || []) t.release();
    s.latent = s.styles = s.stylesData = null;
    if (!keepEmbedding) this.sources.delete(s);
  }

  // The identity for faces without their own source.
  async setSource(embedding) {
    this.releaseSource();
    this.source = await this.createSource(embedding);
  }

  releaseSource() {
    if (this.source) this.freeSource(this.source);
    this.source = null;
  }

  maskProfile(size, blur) {
    const key = `${size}:${blur}`;
    let m = this.masks.get(key);
    if (!m) {
      m = this.gpu.fromArray(face.edgeMaskProfile(size, blur), [size]);
      this.masks.set(key, m);
    }
    return m;
  }

  // Aligned crop of a face for the swapper: [k*k, S, S, 3] (pixelshuffle layout; S and the value
  // range are the swapper's: 128 px in [0, 1] for inswapper) and the frame->crop map.
  // out: a GPUBuffer to write the crop into.
  swapInput(frame, kps, pixelshuffle, out = null) {
    const { size: S, mul, add } = this.swapper.crop;
    const size = S * pixelshuffle;
    const m = face.alignment(kps, face.SWAP_128, size);
    const x = ops.resample(this.gpu, frame, { ow: size, oh: size, m: face.invert(m), mul, add, k: pixelshuffle, out });
    return { x, m, size };
  }

  // The occlusion mask's blur for edge softness blur 0..1 (the edge mask's Edges): a
  // Gaussian of 5 px on the 256 px crop at its default 0.3, none at 0.
  occlusionBlur(blur) {
    let b = this.occBlurs.get(blur);
    if (!b) {
      const taps = face.gaussianKernel(face.occlusionSigma(blur));
      b = { buf: this.gpu.upload(Float32Array.from(taps)), r: taps.length >> 1 };
      this.occBlurs.set(blur, b);
    }
    return b;
  }

  // XSeg's output y [B, N, N, 1] -> the occlusion mask: grown or shrunk (maskSize, face.occlusionGrow),
  // blurred (occlusionBlur), (clamp(m, 0.5, 1) - 0.5) * 2
  occlusionPost(y, blur, maskSize = 0) {
    const b = this.occlusionBlur(blur);
    return ops.occlusionMask(this.gpu, y, b.buf, b.r, face.occlusionGrow(maskSize, y.shape[1]));
  }

  // the occlusion masks of the faces (what of each face is face, not a hand or glasses in
  // front of it) from frame (the frame before any swap): the loaded occluder on each face's crop
  // of template's framing, batched (through WebNN when loaded so), grown or shrunk by maskSize -1..1
  // and blurred for edge softness blur.
  // -> { face(i): unocclude's occ for face i ({ buf, off, m, size }), release() }
  async occlusionMasks(frame, faces, { template = this.occluderInfo.template, blur = 0.3, maskSize = 0 } = {}) {
    const gpu = this.gpu;
    const { size: N, mul, add } = this.occluder.crop;
    const maps = faces.map((f) => face.alignment(f.kps, face.TEMPLATES[template], N));
    const parts = [];
    for (let i = 0; i < faces.length; i += OCCLUDER_BATCH) {
      const n = Math.min(OCCLUDER_BATCH, faces.length - i);
      const x = gpu.empty([n, N, N, 3]);
      for (let j = 0; j < n; j++) ops.resample(gpu, frame, { ow: N, oh: N, m: face.invert(maps[i + j]), mul, add, out: x, offset: j * N * N * 3 });
      const y = this.nnOccluder ? await this.nnOccluder.forward(x) : this.occluder.forward(x);
      x.release();
      parts.push(this.occlusionPost(y, blur, maskSize));
      y.release();
    }
    return {
      face: (i) => ({ buf: parts[Math.floor(i / OCCLUDER_BATCH)].buf, off: (i % OCCLUDER_BATCH) * N * N, m: maps[i], size: N }),
      release: () => parts.forEach((p) => p.release()),
    };
  }

  // A copy of the frame as it is (this.orig, reused for frames of the same size).
  snapshot(frame) {
    if (this.orig && (this.orig.w !== frame.w || this.orig.h !== frame.h)) {
      this.orig.buf.destroy();
      this.orig = null;
    }
    const bytes = frame.stride * frame.h * 4;
    this.orig = this.orig || { w: frame.w, h: frame.h, stride: frame.stride, buf: this.gpu.device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }) };
    this.gpu.encode((enc) => enc.copyBufferToBuffer(frame.buf, 0, this.orig.buf, 0, bytes));
    return this.orig;
  }

  // The last step of a frame with Occlusion on: each face's occlusion mask from the frame before
  // the swap (orig), on the widest crop that changed (the enhancer's, when it ran), and orig put
  // back where it says something covers the face. With the edge masks of the pastes before it,
  // a face's mask is box * occlusion: one mask, its edges softened by blur.
  async unoccludeFaces(frame, orig, faces, { template, blur, maskSize }) {
    const occ = await this.occlusionMasks(orig, faces, { template, blur, maskSize });
    for (let i = 0; i < faces.length; i++) {
      const o = occ.face(i);
      ops.unocclude(this.gpu, frame, orig, o, face.pasteBox(o.m, o.size, frame.w, frame.h));
    }
    occ.release();
  }

  // Swaps a source identity into each face of the frame, in place: the face's own (f.source,
  // from createSource) or the default one (setSource).
  // pixelshuffle: 1 | 2 | 3 | 4 (crops of 1 to 4 times the swapper's size, k*k swapper passes batched);
  // blur: mask edge 0..1 (the edge masks', and the occlusion mask's); enhance: { blend, detail }
  // to restore the faces afterwards with the loaded enhancer (enhanceFaces; f.detail: the face's
  // own detail, faceDetails, when known), or null; occlusion: keep what
  // covers a face (hands, glasses) with the loaded occluder, as the last step (unoccludeFaces);
  // maskSize: -1..1, shrinks or grows that mask (more of the face is swapped when it's positive)
  async swapFaces(frame, faces, { pixelshuffle = 1, blur = 0.3, enhance = null, occlusion = false, maskSize = 0 } = {}) {
    const sourceOf = (f) => {
      const s = f.source || this.source;
      if (!s?.styles) throw new Error("no source face set");
      return s;
    };
    const orig = occlusion && this.occluder && faces.length ? this.snapshot(frame) : null;
    const swapSize = this.swapper.crop.size * pixelshuffle;
    const enh = this.enhancerFor(enhance);
    const template = enh ? enh.info.template : this.occluderInfo?.template;
    for (const f of faces) {
      const src = sourceOf(f);
      if (this.nn?.interop && pixelshuffle * pixelshuffle <= this.nn.maxBatch) {
        // WebNN sharing the device: the crop goes straight into the graph's input and the result
        // is pasted straight from its output (larger batches run in parts through forward below)
        let m, size;
        const y = await this.nn.forwardGPU((buf) => ({ m, size } = this.swapInput(frame, f.kps, pixelshuffle, buf)), pixelshuffle * pixelshuffle, src.stylesData);
        ops.paste(this.gpu, frame, y, this.maskProfile(size, blur), m, face.pasteBox(m, size, frame.w, frame.h), size, pixelshuffle);
        this.gpu.flush();
        y.destroy(); // back to WebNN, after the paste
        continue;
      }
      const { x, m, size } = this.swapInput(frame, f.kps, pixelshuffle);
      let y;
      if (this.nn) {
        // WebNN: the crops go through the CPU (192 KB per 128 px crop, each way)
        const out = await this.nn.forward(await this.gpu.read(x), x.shape[0], src.stylesData);
        y = this.gpu.fromArray(out, x.shape);
      } else y = this.swapper.forward(x, src.styles);
      x.release();
      ops.paste(this.gpu, frame, y, this.maskProfile(size, blur), m, face.pasteBox(m, size, frame.w, frame.h), size, pixelshuffle);
      y.release();
    }
    if (enh) this.enhanceFaces(frame, faces, { ...enhance, model: enh.part.slice(9), blur, pixelshuffle });
    if (orig) await this.unoccludeFaces(frame, orig, faces, { template, blur, maskSize });
  }

  // The pixelshuffle for a face with keypoints kps, of those in `choices` (sorted): the smallest
  // crop (1 to 4 times the swapper's size) with the face's own resolution in the frame (to within
  // 10%), so a small face isn't swapped at more detail than the frame has, and a large one isn't
  // blurred by a small crop; with none of the choices that large, the largest of them.
  autoPixelshuffle(kps, choices = [1, 2, 3, 4]) {
    const S = this.swapper.crop.size;
    const px = S / face.scaleOf(face.alignment(kps, face.SWAP_128, S)); // the crop, in frame px
    const need = Math.min(4, Math.max(1, Math.ceil(px / S - 0.1)));
    return choices.find((b) => b >= need) ?? choices[choices.length - 1];
  }

  // The part of a w x h frame swapFaces changes for a face with keypoints kps and these options:
  // the box [x1, y1, x2, y2) around the swap's paste and, when they run, the enhancer's and the
  // occlusion mask's. What it reads of the frame lies within a pixel of it.
  faceBox(kps, { pixelshuffle = 1, enhance = null, occlusion = false } = {}, w, h) {
    const swapSize = this.swapper.crop.size * pixelshuffle;
    const boxes = [face.pasteBox(face.alignment(kps, face.SWAP_128, swapSize), swapSize, w, h)];
    const enh = this.enhancerFor(enhance);
    if (enh) {
      const { size: N, template } = enh.info;
      boxes.push(face.pasteBox(face.alignment(kps, face.TEMPLATES[template], N), N, w, h));
    }
    if (occlusion && this.occluder) {
      const N = this.occluder.crop.size;
      const template = enh ? enh.info.template : this.occluderInfo.template;
      boxes.push(face.pasteBox(face.alignment(kps, face.TEMPLATES[template], N), N, w, h));
    }
    return [Math.min(...boxes.map((b) => b[0])), Math.min(...boxes.map((b) => b[1])), Math.max(...boxes.map((b) => b[2])), Math.max(...boxes.map((b) => b[3]))];
  }

  // A short string naming the loaded models a swap with these options (swapFaces') goes through:
  // it changes when they do, a file replaced by another of the same name too, and only then (a
  // model loaded for other faces doesn't change it).
  configFor({ enhance = null, occlusion = false } = {}) {
    const parts = ["swapper"];
    const enh = this.enhancerFor(enhance);
    if (enh) parts.push(enh.part);
    if (occlusion && this.occluder) parts.push("occluder");
    return JSON.stringify(parts.map((k) => [k, this.loaded[k], this.fileDates[k]]));
  }

  // The detail a swap at pixelshuffle `pixelshuffle` has, px across the swapper's framing: the most a
  // swap there holds (swapper.detail), less for a face with less detail of its own (faceDetail,
  // px, when known): softmin(a * faceDetail, most), softmin(x, y) = (x^-p + y^-p)^(-1/p)
  // (swapper.detailFit).
  swapDetail(pixelshuffle, faceDetail = null) {
    const most = this.swapper.detail[pixelshuffle - 1];
    if (faceDetail == null) return most;
    const { a, p } = this.swapper.detailFit;
    return Math.round(((a * faceDetail) ** -p + most ** -p) ** (-1 / p));
  }

  // The size of the band below which a swap with swapDetail px of detail (across the swapper's
  // framing, swapDetail) has all of it, in px of the enhancer's crop: Enhancer-256 shares the
  // swapper's framing, so 128 px stay 128; Enhancer-512's wider framing holds them in 143 px.
  enhancerBand(swapDetail, info) {
    return Math.round(swapDetail * face.scaleOf(face.similarity(face.TEMPLATES[info.template], face.SWAP_128)));
  }

  // The loaded enhancer a swap's enhance options ({ model: component }) name: { part, net, info },
  // or null (none named, or not loaded). Without a model, the enhancer loaded, while there's one.
  enhancerFor(enhance) {
    if (!enhance) return null;
    const ks = Object.keys(this.loaded).filter((k) => partKind(k) === "enhancer" && this[k]);
    const part = enhance.model ? `enhancer:${enhance.model}` : ks.length === 1 ? ks[0] : null;
    return part && this[part] ? { part, net: this[part], info: this.enhancerInfos[part.slice(9)] } : null;
  }

  // the enhancer loaded, while there's just one
  get enhancer() { return this.enhancerFor({})?.net || null; }

  // Restores each (swapped) face of the frame with the enhancer, in place: the face
  // enhancer (the enhancer's own crop, edge mask with edge blur, blended by blend 0..1).
  // detail: the restorer may only add what is finer than the swap (the detail the swap really
  // has, swapDetail: at its pixelshuffle, and the face's own detail f.detail when known):
  // out = G + lp(C - G), C the crop, G the
  // enhancer's output, lp the part below the swap's detail (down to it and back), so the result
  // downsamples back to the swap exactly. It is pasted as a change, out - C, so what the frame
  // has above the enhancer's resolution (a face larger than its crop) stays.
  enhanceFaces(frame, faces, { model, blend = 0.8, blur = 0.3, detail = true, pixelshuffle = 1 } = {}) {
    const gpu = this.gpu;
    const { net, info } = this.enhancerFor({ model });
    const { size: N, template } = info;
    const { mul, add } = net.crop;
    for (const f of faces) {
      const lo = detail ? this.enhancerBand(this.swapDetail(pixelshuffle, f.detail), info) : 0;
      const m = face.alignment(f.kps, face.TEMPLATES[template], N);
      const x = ops.resample(gpu, frame, { ow: N, oh: N, m: face.invert(m), mul, add });
      const g = net.forward(x);
      const band = detail ? ops.bandDown(gpu, x, g, lo) : null;
      const y = ops.enhanceOut(gpu, g, band, x);
      x.release();
      g.release();
      band?.release();
      ops.paste(gpu, frame, y, this.maskProfile(N, blur), m, face.pasteBox(m, N, frame.w, frame.h), N, 1, blend, detail);
      y.release();
    }
  }
}
