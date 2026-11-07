// The swapped faces already made: a patch is the finished pixels of the part of a frame one face
// (or a few faces close together) changes, under a key naming everything that went into it (the
// video, the frame, the face, its settings, the models). The preview and the export make frames
// the same way, from these patches, so what was seen is what gets exported, and made only once.
//
// Recent patches stay in memory; all of them go to the cache folder (a folder per photo
// or video, whose name starts the key: "folder/name"; one file per patch: 16 bytes of box, then the
// RGBA rows as qoi.js writes them), which the server keeps within its budget.

import { encode, decode } from "./qoi.js";
import { getPatch, patchNames, putPatch, setPatchPlan, clearPatches } from "./workspace.js";

const HEADER = 16;

// SHA-256 of a string, as hex
export async function hash(text) {
  const d = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)));
  return Array.from(d, (b) => b.toString(16).padStart(2, "0")).join("");
}

const bytesOf = (p) => p.data.byteLength;

export class PatchCache {
  constructor({ memory = 256 * 2 ** 20 } = {}) {
    this.memory = memory;
    this.mem = new Map(); // key -> { box, data }, in LRU order
    this.memBytes = 0;
    this.writes = Promise.resolve();
    this.plans = new Map(); // folder -> the plan the server was last told
    this.names = new Map(); // folder -> a promise of the Set of names kept there
  }

  remember(key, p) {
    const old = this.mem.get(key);
    if (old) { this.memBytes -= bytesOf(old); this.mem.delete(key); }
    this.mem.set(key, p);
    this.memBytes += bytesOf(p);
    for (const [k, q] of this.mem) {
      if (this.memBytes <= this.memory) break;
      this.mem.delete(k);
      this.memBytes -= bytesOf(q);
    }
  }

  // the names kept in a folder, asked for once: going through a whole video, a request per frame
  // would add up to minutes (the time is the browser's, whatever the server does)
  namesIn(folder) {
    if (!this.names.has(folder)) {
      this.names.set(folder, patchNames(folder).then((list) => new Set(list), () => {
        this.names.delete(folder);
        return new Set();
      }));
    }
    return this.names.get(folder);
  }

  // -> whether the patch is kept (without reading it). As far as this session knows: one the server
  // has dropped since is found out when it's read, and swapped again.
  async has(key) {
    if (this.mem.has(key)) return true;
    const [folder, name] = key.split("/");
    return (await this.namesIn(folder)).has(name);
  }

  // -> { box: [x1, y1, x2, y2], data (RGBA) } or null
  async get(key) {
    const p = this.mem.get(key);
    if (p) { this.mem.delete(key); this.mem.set(key, p); return p; }
    // (one that can't be read is one to swap again)
    const bytes = await getPatch(key).catch(() => null);
    if (!bytes || bytes.byteLength < HEADER) return null;
    const [x1, y1, x2, y2] = new Uint32Array(bytes.buffer, bytes.byteOffset, HEADER / 4);
    const n = (x2 - x1) * (y2 - y1);
    // (a stream takes at least a byte per 62 pixels, so this size is no surprise)
    if (x2 < x1 || y2 < y1 || n > (bytes.byteLength - HEADER) * 62) return null;
    const data = new Uint8ClampedArray(n * 4);
    if (!decode(bytes.subarray(HEADER), n, data)) return null;
    const q = { box: [x1, y1, x2, y2], data };
    this.remember(key, q);
    return q;
  }

  // Keeps a patch: in memory now, on disk soon after.
  put(key, p) {
    this.remember(key, p);
    const [folder, name] = key.split("/");
    this.names.get(folder)?.then((names) => names.add(name));
    this.writes = this.writes.then(() => putPatch(key, new Blob([new Uint32Array(p.box), encode(p.data, bytesOf(p) / 4)]))).catch((e) => console.warn("a swapped face couldn't be saved:", e.message));
    return this.writes;
  }

  // What the folder's patches are being made with from here on (a string naming it). After the
  // patches on their way, so they count as made before it.
  setPlan(folder, plan) {
    if (this.plans.get(folder) === plan) return this.writes;
    this.plans.set(folder, plan);
    this.writes = this.writes.then(async () => setPatchPlan(folder, await hash(plan))).catch((e) => console.warn("the plan couldn't be saved:", e.message));
    return this.writes;
  }

  async clear() {
    this.mem.clear();
    this.memBytes = 0;
    this.plans.clear();
    this.names.clear();
    await this.writes;
    await clearPatches();
  }
}
