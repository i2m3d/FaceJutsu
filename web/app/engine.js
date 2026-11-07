// Page-side handle on the engine. Runs the session in a Web Worker (worker.js) so the UI stays
// responsive; falls back to running it on the page when WebGPU isn't exposed to workers.
//
// When the GPU is lost (a driver reset, the GPU process crashing, running out of memory), every
// call waiting fails with a GPULostError and onLost({ reason, message }) is called once; later
// calls fail too until restart() starts a fresh session, with nothing loaded.

import { Session } from "./session.js";

function lostError(info) {
  const e = new Error(`WebGPU device lost: ${info.message || info.reason || "unknown reason"}`);
  e.name = "GPULostError";
  return e;
}

const newWorker = () => new Worker(new URL("./worker.js", import.meta.url), { type: "module" });

class WorkerEngine {
  constructor(worker) {
    this.next = 1;
    this.calls = new Map();
    this.gpu = { name: "", peak: 0 };
    this.lost = null; // { reason, message } once the GPU was lost
    this.onLost = null;
    this.attach(worker);
  }

  attach(worker) {
    this.worker = worker;
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === "lost") { this.lose(m); return; }
      const call = this.calls.get(m.id);
      if (!call) return;
      if (m.type === "status") call.onStatus?.(m.status);
      else if (m.type === "progress") call.onProgress?.(m.progress);
      else if (m.type === "preview") call.onPreview?.(m.bitmap);
      else if (m.type === "frame") call.onFrame?.(m.frame);
      else {
        this.calls.delete(m.id);
        if (m.gpu) this.gpu = m.gpu;
        if (m.type === "error") {
          const err = new Error(m.message);
          err.name = m.name;
          err.stack = m.stack;
          call.reject(err);
        } else call.resolve(m);
      }
    };
  }

  // nothing waits for a worker whose GPU is gone
  lose(info) {
    if (this.lost) return;
    this.lost = { reason: info.reason, message: info.message };
    const err = lostError(this.lost);
    for (const call of this.calls.values()) call.reject(err);
    this.calls.clear();
    this.onLost?.(this.lost);
  }

  // whether the worker started, with WebGPU
  started() {
    return Promise.race([
      this.call({ type: "init" }),
      new Promise((_, rej) => setTimeout(() => rej(new Error("worker did not start")), 10000)),
    ]).then((r) => r.ok);
  }

  // A fresh worker in place of the one whose GPU was lost: whatever was left on the GPU goes with
  // the old one.
  async restart() {
    this.worker.terminate();
    for (const call of this.calls.values()) call.reject(lostError(this.lost || {}));
    this.calls.clear();
    this.lost = null;
    this.gpu = { name: "", peak: 0 };
    this.attach(newWorker());
    if (!(await this.started())) throw new Error("WebGPU isn't available to the engine any more.");
  }

  call(msg, hooks = {}, transfer = []) {
    if (this.lost) return Promise.reject(lostError(this.lost));
    const id = this.next++;
    return new Promise((resolve, reject) => {
      this.calls.set(id, { resolve, reject, ...hooks });
      if (hooks.signal) {
        if (hooks.signal.aborted) this.worker.postMessage({ type: "cancel", target: id });
        hooks.signal.addEventListener("abort", () => this.worker.postMessage({ type: "cancel", target: id }), { once: true });
      }
      this.worker.postMessage({ id, ...msg }, transfer);
    });
  }

  load(base, manifest, selection, { onStatus, signal } = {}) {
    return this.call({ type: "load", base: base.href, manifest, selection }, { onStatus, signal });
  }

  embedImage(bitmap) { return this.call({ type: "embedImage", bitmap }, {}, [bitmap]); }
  setCharacter(key, embedding) { return this.call({ type: "setCharacter", key, embedding }); }
  removeCharacter(key) { return this.call({ type: "removeCharacter", key }); }
  analyze(key, input, opts, { onProgress, signal } = {}) {
    return this.call({ type: "analyze", key, input, opts }, { onProgress, signal }, input instanceof Blob ? [] : [input]);
  }
  restoreAnalysis(key, analysis) { return this.call({ type: "restoreAnalysis", key, analysis }); }
  openTarget(key, input) { return this.call({ type: "openTarget", key, input }, {}, input instanceof Blob ? [] : [input]); }
  render(frames, opts, { onFrame, signal } = {}) { return this.call({ type: "render", frames, opts }, { onFrame, signal }); }
  clearPatches() { return this.call({ type: "clearPatches" }); }
  swapVideo(file, opts, { name, out, onProgress, onPreview, signal } = {}) {
    return this.call({ type: "video", file, opts, name, out }, { onProgress, onPreview, signal });
  }
}

class LocalEngine {
  constructor() {
    this.lost = null;
    this.onLost = null;
    this.waiting = new Set(); // the rejects of the calls running
    this.start();
  }

  start() {
    const session = new Session({ onLost: (info) => { if (this.session === session) this.lose(info); } });
    this.session = session;
  }

  lose(info) {
    if (this.lost) return;
    this.lost = { reason: info.reason, message: info.message };
    const err = lostError(this.lost);
    for (const reject of this.waiting) reject(err);
    this.waiting.clear();
    this.onLost?.(this.lost);
  }

  // A new session in place of the one whose GPU was lost (its own device, from scratch).
  async restart() {
    this.session.closeTarget();
    this.lost = null;
    this.start();
  }

  // runs fn on the session; it fails once the GPU is lost, even when the session's work hangs
  guard(fn) {
    if (this.lost) return Promise.reject(lostError(this.lost));
    let reject;
    const lost = new Promise((_, rej) => { reject = rej; });
    this.waiting.add(reject);
    return Promise.race([Promise.resolve().then(fn), lost]).finally(() => this.waiting.delete(reject));
  }

  get gpu() { return this.session.gpuInfo; }
  load(base, manifest, selection, opts) { return this.guard(() => this.session.load(base, manifest, selection, opts)); }
  embedImage(bitmap) { return this.guard(() => this.session.embedImage(bitmap)); }
  setCharacter(key, embedding) { return this.guard(() => this.session.setCharacter(key, embedding)); }
  removeCharacter(key) { return this.guard(() => this.session.removeCharacter(key)); }
  analyze(key, input, opts, hooks) { return this.guard(() => this.session.analyze(key, input, opts, hooks)); }
  restoreAnalysis(key, analysis) { return this.guard(() => this.session.restoreAnalysis(key, analysis)); }
  openTarget(key, input) { return this.guard(() => this.session.openTarget(key, input)); }
  render(frames, opts, hooks) { return this.guard(() => this.session.render(frames, opts, hooks)); }
  clearPatches() { return this.guard(() => this.session.clearPatches()); }
  swapVideo(file, opts, hooks) { return this.guard(() => this.session.swapVideo(file, opts, hooks)); }
}

// inPage: force the in-page engine (debugging; ?engine=page)
export async function createEngine({ inPage = false } = {}) {
  if (inPage) return new LocalEngine();
  try {
    const worker = newWorker();
    const engine = new WorkerEngine(worker);
    if (await engine.started()) {
      engine.inWorker = true;
      return engine;
    }
    worker.terminate();
    console.warn("WebGPU is not available in workers here; running the engine on the page");
  } catch (e) {
    console.warn("engine worker unavailable, running on the page:", e.message);
  }
  return new LocalEngine();
}
