// Engine worker: hosts the session so model loading, GPU command encoding and video
// decoding/encoding never block the page. Protocol (see engine.js):
//   in : { id, type: "init" | "load" | "embedImage" | "setCharacter" | "removeCharacter" | "analyze" | "restoreAnalysis" | "openTarget" | "render" | "video" | "clearPatches" | "cancel", ... }
//   out: { id, type: "status" | "progress" | "preview" | "frame" | "result" | "error", ... },
//        and { type: "lost", reason, message } (no id) when the GPU is lost

import { Session } from "./session.js";

let session = null;
const aborts = new Map();

async function handle(msg) {
  const { id, type } = msg;
  const reply = (m, transfer = []) => self.postMessage({ id, ...m }, transfer);
  // (a scan's update brings the new thumbnails along)
  const progress = (p) => reply({ type: "progress", progress: p }, (p.scan?.thumbs || []).map(([, b]) => b));
  if (type === "init") {
    reply({ type: "result", ok: !!self.navigator.gpu });
    return;
  }
  if (type === "cancel") {
    aborts.get(msg.target)?.abort();
    return;
  }
  const ac = new AbortController();
  aborts.set(id, ac);
  try {
    session = session || new Session({ onLost: (e) => self.postMessage({ type: "lost", ...e }) });
    if (type === "load") {
      await session.load(new URL(msg.base), msg.manifest, msg.selection, { signal: ac.signal, onStatus: (s) => reply({ type: "status", status: s }) });
      reply({ type: "result", gpu: session.gpuInfo });
    } else if (type === "embedImage") {
      reply({ type: "result", ...(await session.embedImage(msg.bitmap)) });
    } else if (type === "setCharacter") {
      await session.setCharacter(msg.key, msg.embedding);
      reply({ type: "result" });
    } else if (type === "removeCharacter") {
      session.removeCharacter(msg.key);
      reply({ type: "result" });
    } else if (type === "analyze") {
      const r = await session.analyze(msg.key, msg.input, msg.opts, { signal: ac.signal, onProgress: progress });
      reply({ type: "result", ...r }, (r.people || []).map((p) => p.thumb).filter(Boolean));
    } else if (type === "restoreAnalysis") {
      await session.restoreAnalysis(msg.key, msg.analysis);
      reply({ type: "result" });
    } else if (type === "openTarget") {
      await session.openTarget(msg.key, msg.input);
      reply({ type: "result" });
    } else if (type === "render") {
      await session.render(msg.frames, msg.opts, {
        signal: ac.signal,
        onFrame: (f) => reply({ type: "frame", frame: f }, (f.patches || []).map((p) => p.bitmap)),
      });
      reply({ type: "result" });
    } else if (type === "clearPatches") {
      await session.clearPatches();
      reply({ type: "result" });
    } else if (type === "video") {
      const r = await session.swapVideo(msg.file, msg.opts, {
        name: msg.name,
        out: msg.out,
        signal: ac.signal,
        onProgress: progress,
        onPreview: (bitmap) => reply({ type: "preview", bitmap }, [bitmap]),
      });
      reply({ type: "result", ...r, gpu: session.gpuInfo });
    }
  } catch (e) {
    reply({ type: "error", name: e.name, message: e.message, stack: e.stack });
  } finally {
    aborts.delete(id);
  }
}

self.onmessage = (e) => { handle(e.data); };
