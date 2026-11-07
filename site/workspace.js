// web/app/workspace.js for the site, which has no server: the same exports, with the user's files in
// the browser's private storage for this site (OPFS), the models fetched straight from their
// addresses, and finished photos and videos handed over as downloads. Works from the page and from
// the engine's worker.

import { encode, decode } from "./records.js";
import { zipFiles } from "./zip.js";

export { encode, decode };

const OUTPUTS = "outputs", MODELS = "models", CACHE = "cache", WORK = "work", FACES = "work/faces";
const PLAN = "plan";
const STORES = ["characters", "targets"];
// How many bytes of swapped frames to keep at most, and how much of what the browser allows this site.
const CACHE_MAX = 4 * 2 ** 30, CACHE_SHARE = 0.5;
// Outputs are kept this long after they're handed over, for Download again.
const OUTPUT_KEPT = 24 * 60 * 60 * 1000;
const PHOTOS_ZIP = "photos-swapped.zip";

const notFound = (e) => e.name === "NotFoundError";

function roomy(e) {
  if (e.name !== "QuotaExceededError") return e;
  return new Error("The browser has no room left for this site. Free some disk space, or remove the swapped frames in Storage & about.");
}

async function folder(path, create = true) {
  let dir = await navigator.storage.getDirectory();
  for (const part of path.split("/")) dir = await dir.getDirectoryHandle(part, { create });
  return dir;
}

// -> what fn makes of the folder, or `none` when it isn't there
async function inFolder(path, fn, none = null) {
  try {
    return await fn(await folder(path, false));
  } catch (e) {
    if (notFound(e)) return none;
    throw e;
  }
}

const readFile = (path, name) => inFolder(path, async (dir) => (await dir.getFileHandle(name)).getFile());

// (a file being written isn't there until it's closed, so a stop midway leaves the old one, or none)
async function writeFile(path, name, body) {
  const handle = await (await folder(path)).getFileHandle(name, { create: true });
  const out = await handle.createWritable();
  try {
    await out.write(body);
    await out.close();
  } catch (e) {
    await out.abort().catch(() => {});
    throw roomy(e);
  }
}

const removeFile = (path, name) => inFolder(path, (dir) => dir.removeEntry(name));

async function removeFolder(path) {
  const at = path.lastIndexOf("/");
  const parent = at < 0 ? await navigator.storage.getDirectory() : await folder(path.slice(0, at));
  await parent.removeEntry(path.slice(at + 1), { recursive: true }).catch((e) => { if (!notFound(e)) throw e; });
}

// -> [File] of a folder, with the files' names
const filesIn = (path) => inFolder(path, async (dir) => {
  const out = [];
  for await (const entry of dir.values()) if (entry.kind === "file") out.push(await entry.getFile());
  return out;
}, []);

const foldersIn = (path) => inFolder(path, async (dir) => {
  const out = [];
  for await (const entry of dir.values()) if (entry.kind === "directory") out.push(entry.name);
  return out;
}, []);

// There's no server to stop and come back.
export const untilBack = (fn) => fn();
export const watchServer = () => {};

export const status = async () => ({
  app: "facejutsu",
  build: "site",
  models: "this browser",
  outputs: "",
  work: "this browser",
  cache: "this browser",
});

// ------------------------------------------------------------------ models

const siteFile = async (name) => {
  const res = await fetch(new URL(`../models/${name}`, import.meta.url), { cache: "no-cache" });
  if (!res.ok) throw new Error(`could not load ${name} (${res.status})`);
  return res.json();
};

// (an empty one is what a download stopped by closing the page leaves)
export const modelFiles = async () => (await filesIn(MODELS))
  .filter((f) => f.name.endsWith(".safetensors") && f.size)
  .map((f) => ({ name: f.name, size: f.size }));

export const modelsByHand = false;

export const modelUrls = async () => ({ urls: await siteFile("urls.json") });

export async function readModel(url) {
  const name = decodeURIComponent(url.pathname.split("/").pop());
  const file = await readFile(MODELS, name);
  if (!file) return new Response(null, { status: 404 });
  return new Response(file, { headers: { "last-modified": new Date(file.lastModified).toUTCString() } });
}

const sizeInManifest = (manifest, name) => {
  for (const c of Object.values(manifest.components)) {
    for (const b of Object.values(c.builds)) if (b.path === name) return b.size;
  }
  return null;
};

// Fetches a model file into the browser's storage. onProgress(done, total): total may be null.
export async function downloadModel(name, onProgress) {
  navigator.storage.persist?.();
  const [urls, manifest] = await Promise.all([siteFile("urls.json"), siteFile("manifest.json")]);
  if (!urls[name]) throw new Error("it has no address");
  const size = sizeInManifest(manifest, name);
  let res;
  try {
    res = await fetch(urls[name], { credentials: "omit", referrerPolicy: "no-referrer", cache: "no-store" });
  } catch (e) {
    throw new Error(`couldn't reach the address (${e.message})`);
  }
  if (!res.ok) throw new Error(`the address answered ${res.status}`);
  const total = size ?? (Number(res.headers.get("content-length")) || null);
  const handle = await (await folder(MODELS)).getFileHandle(name, { create: true });
  const out = await handle.createWritable();
  try {
    const reader = res.body.getReader();
    let done = 0;
    for (;;) {
      const { done: end, value } = await reader.read();
      if (end) break;
      await out.write(value);
      done += value.byteLength;
      onProgress?.(done, total);
    }
    if (size != null && done !== size) throw new Error(`got ${done} bytes, not the ${size} expected: the address may hold another build of it`);
    await out.close();
  } catch (e) {
    await out.abort().catch(() => {});
    await removeFile(MODELS, name).catch(() => {});
    throw roomy(e);
  }
}

// ------------------------------------------------------------------ records

const recordFolder = (store) => `${WORK}/${store}`;

export async function getRecordNow(store, key) {
  const file = await readFile(recordFolder(store), `${key}.json`);
  return file ? decode(await file.text()) : null;
}
export const getRecord = getRecordNow;
export const allRecords = async (store) => Promise.all((await filesIn(recordFolder(store))).map(async (f) => decode(await f.text())));
export const putRecord = (store, key, value) => writeFile(recordFolder(store), `${key}.json`, encode(value));
export const removeRecord = (store, key) => removeFile(recordFolder(store), `${key}.json`);
export const recordCounts = async () => {
  const [characters, targets] = await Promise.all(STORES.map(async (s) => (await filesIn(recordFolder(s))).length));
  return { characters, targets };
};
export const clearRecords = () => removeFolder(WORK);

export const getFaces = async (key) => {
  const file = await readFile(FACES, `${key}.faces`);
  return file ? file.arrayBuffer() : null;
};
export const putFaces = (key, body) => writeFile(FACES, `${key}.faces`, body);

// ------------------------------------------------------------------ swapped frames

const cacheFolder = (target) => `${CACHE}/${target}`;
const framesIn = async (target) => (await filesIn(cacheFolder(target))).filter((f) => f.name !== PLAN);

async function measure() {
  let bytes = 0, count = 0;
  for (const target of await foldersIn(CACHE)) {
    for (const f of await framesIn(target)) {
      bytes += f.size;
      count++;
    }
  }
  return { bytes, count };
}

// What this page or worker knows of the cache: its size, the most to keep, and the video that filled
// it with nothing left to drop for it, whose new frames aren't kept.
let kept = null;
async function cacheState() {
  kept ??= Promise.all([measure(), navigator.storage.estimate()]).then(([used, { quota }]) => ({
    ...used,
    budget: Math.min(CACHE_MAX, Math.floor((quota || CACHE_MAX) * CACHE_SHARE)),
    stopped: null,
  }));
  return kept;
}

// Removes `frames` in turn from a video's folder, until the cache is down to `goal`.
async function dropFrames(state, target, frames, goal) {
  for (const f of frames) {
    if (state.bytes <= goal) return;
    await removeFile(cacheFolder(target), f.name);
    state.bytes -= f.size;
    state.count--;
  }
}

// Makes room for the frames of `target`: first the other videos', the one last added to longest ago
// first, then those of `target` from before its plan last changed, by name, so in no order of the
// video (going oldest first would drop the very frames a second export comes to next).
async function makeRoom(state, target) {
  const goal = state.budget * 0.9;
  const others = [];
  for (const name of await foldersIn(CACHE)) {
    if (name === target) continue;
    const frames = await framesIn(name);
    others.push({ name, frames, newest: Math.max(0, ...frames.map((f) => f.lastModified)) });
  }
  others.sort((a, b) => a.newest - b.newest);
  for (const { name, frames } of others) {
    if (state.bytes <= goal) return;
    await dropFrames(state, name, frames, goal);
    if (!(await framesIn(name)).length) await removeFolder(cacheFolder(name));
  }
  const plan = await readFile(cacheFolder(target), PLAN);
  if (!plan) return;
  const stale = (await framesIn(target)).filter((f) => f.lastModified < plan.lastModified);
  stale.sort((a, b) => (a.name < b.name ? -1 : 1));
  await dropFrames(state, target, stale, goal);
}

export async function getPatch(key) {
  const [target, name] = key.split("/");
  const file = await readFile(cacheFolder(target), name);
  return file ? new Uint8Array(await file.arrayBuffer()) : null;
}

export const patchNames = async (target) => (await framesIn(target)).map((f) => f.name);

export async function putPatch(key, body) {
  const [target, name] = key.split("/");
  if (await readFile(cacheFolder(target), name)) return;
  const state = await cacheState();
  const fits = () => state.bytes + body.size <= state.budget;
  if (!fits()) {
    if (state.stopped === target) return;
    await makeRoom(state, target);
    state.stopped = fits() ? null : target;
    if (state.stopped) return;
  }
  await writeFile(cacheFolder(target), name, body);
  state.bytes += body.size;
  state.count++;
}

export async function setPatchPlan(target, plan) {
  const old = await readFile(cacheFolder(target), PLAN);
  if (old && (await old.text()) === plan) return;
  await writeFile(cacheFolder(target), PLAN, plan);
  const state = await cacheState();
  state.stopped = null;
}

export const patchUsage = measure;

export async function clearPatches() {
  await removeFolder(CACHE);
  const state = await cacheState();
  state.bytes = state.count = 0;
  state.stopped = null;
}

// ------------------------------------------------------------------ outputs

let serial = 0;

const canSaveAs = () => typeof self.showSaveFilePicker === "function";

// Download: kept in the browser's storage as it's made, then downloaded. Save as…: written straight
// to a file the user picks, so a long video takes no room here. Only Chromium on a computer can.
export const saveChoices = {
  options: [
    { id: "download", label: "Download", tip: "The video goes to your browser's downloads when it's done" },
    { id: "saveas", label: "Save as…", tip: "Asks where to save, then writes the video there as it's made: best for large videos" },
  ],
  help: self.document && !canSaveAs() ? "This browser can't write videos straight to a file. Chrome and Edge on a computer can." : "",
  flag: navigator.brave ? "file-system-access-api" : null,
};

// -> what openOutput is to write: `name`, or the file picked for it. Closing the picker cancels.
export async function chooseOutput(name, format, mode) {
  if (mode !== "saveas" || !canSaveAs()) return name;
  const handle = await showSaveFilePicker({ suggestedName: name, types: [{ description: format.label, accept: { [format.mime]: [format.ext] } }] });
  return { name: handle.name, handle };
}

// A file picked with Save as…. A new, empty one goes again if nothing is saved to it; one picked to
// replace keeps what it had.
async function openPicked({ name, handle }, signal) {
  const fresh = !(await handle.getFile()).size;
  const out = await handle.createWritable();
  return {
    write: async ({ position, data }) => {
      signal?.throwIfAborted();
      await out.write({ type: "write", position, data });
    },
    close: async () => {
      await out.close();
      return { name };
    },
    abort: async () => {
      await out.abort().catch(() => {});
      if (fresh) await handle.remove?.().catch(() => {});
    },
  };
}

// A file in the outputs, written as it's made: write({ position, data }), then close() (-> { name,
// path }: no path when it's a file picked with Save as…) or abort() (nothing is left). Written where
// it can't be seen until it's closed.
export async function openOutput(wish, signal) {
  if (wish.handle) return openPicked(wish, signal);
  const name = wish;
  const path = `${Date.now()}-${serial++}-${Math.random().toString(36).slice(2, 8)}`;
  const handle = await (await folder(OUTPUTS)).getFileHandle(path, { create: true });
  const out = await handle.createWritable();
  return {
    write: async ({ position, data }) => {
      signal?.throwIfAborted();
      await out.write({ type: "write", position, data }).catch((e) => { throw roomy(e); });
    },
    close: async () => {
      await out.close().catch((e) => { throw roomy(e); });
      return { name, path };
    },
    abort: async () => {
      await out.abort().catch(() => {});
      await removeFile(OUTPUTS, path);
    },
  };
}

let last = null; // the output last handed over

// Has the browser download an output, unless it's in a file picked already. Only on the page.
export async function handOver(saved) {
  if (!saved.path) return;
  const file = await readFile(OUTPUTS, saved.path);
  if (!file) throw new Error(`${saved.name} is no longer kept. Export it again.`);
  last = saved;
  const url = URL.createObjectURL(file);
  const link = Object.assign(document.createElement("a"), { href: url, download: saved.name });
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 60000);
}

export async function saveOutput(blob, name, signal) {
  const out = await openOutput(name, signal);
  try {
    await out.write({ position: 0, data: blob });
    const saved = await out.close();
    await handOver(saved);
    return saved;
  } catch (e) {
    await out.abort().catch(() => {});
    throw e;
  }
}

// Several photos come as one zip, so the browser doesn't ask about multiple downloads and they don't
// scatter; one comes as itself. Stopped before finish(), nothing is downloaded.
export function outputSet(signal) {
  const made = [];
  return {
    add: async (blob, name) => { made.push({ blob, name }); },
    finish: async () => {
      if (!made.length) return null;
      if (made.length === 1) {
        await saveOutput(made[0].blob, made[0].name, signal);
        return null;
      }
      return saveOutput(await zipFiles(made), PHOTOS_ZIP, signal);
    },
  };
}

// There are no folders to show: "outputs" downloads the last output again, and "models" removes the
// models, the one way the page offers to get the room back.
export async function openFolder(what) {
  if (what === "outputs") {
    if (last) await handOver(last);
    return;
  }
  await removeFolder(MODELS);
  location.reload();
}

// Whichever of the outputs are old enough to have been downloaded long ago.
if (self.document) {
  filesIn(OUTPUTS).then((files) => Promise.all(files
    .filter((f) => Date.now() - f.lastModified > OUTPUT_KEPT)
    .map((f) => removeFile(OUTPUTS, f.name)))).catch(() => {});
}
