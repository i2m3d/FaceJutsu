// The user's work, kept in the work folder across visits (workspace.js): the
// characters made, and per photo or video (by a hash of its content) the segments and settings
// given to its faces, and apart from them, the faces found in it. A long video's faces are a lot
// to read and write, and an edit doesn't change them.

import * as workspace from "./workspace.js";

// Failures (the server stopped, the disk is full) only cost the saving: logged, not thrown.
async function quiet(what, fn, fallback) {
  try {
    return await fn();
  } catch (e) {
    console.warn(`saved work: couldn't ${what}`, e);
    return fallback;
  }
}

export const all = (store) => quiet(`read the ${store}`, () => workspace.allRecords(store), []);
export const put = (store, value) => quiet(`save to ${store}`, () => workspace.putRecord(store, value.key, value));
export const remove = (store, key) => quiet(`remove from ${store}`, () => workspace.removeRecord(store, key));

// ------------------------------------------------------------------ photos and videos

const SAMPLE = 1 << 20; // bytes hashed at each of the file's start, middle and end

// A hash of a file's content: SHA-256 of its size and three 1 MiB samples (the whole file when
// it's small), so a multi-GB video is told apart in milliseconds. The name isn't part of it: a
// renamed copy is the same video.
export async function fileHash(file) {
  const parts = [new TextEncoder().encode(`${file.size}:`)];
  if (file.size <= 3 * SAMPLE) parts.push(await file.arrayBuffer());
  else for (const at of [0, Math.floor((file.size - SAMPLE) / 2), file.size - SAMPLE]) parts.push(await file.slice(at, at + SAMPLE).arrayBuffer());
  const d = await crypto.subtle.digest("SHA-256", await new Blob(parts).arrayBuffer());
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

// ------------------------------------------------------------------ the faces found, as bytes

// An analysis (session.js, scanner.js) as a file: 4 bytes giving the length of its header, the
// header (JSON, everything but the frames, padded to 8 bytes), then per frame its time, per face
// its numbers and its person, per frame how many faces it has and its state. The numbers are kept
// whole, as float64: a patch's key names its faces' keypoints, so faces read back must be the
// same to the last bit for the patches made from them to be found.
const NUMBERS = 16; // box 4, keypoints 5 * 2, score, detail
const NO_FACES = 0xffff; // a frame not looked at yet (its faces are null)

export function packFaces({ times, faces, state, ...rest }) {
  let total = 0;
  for (const fs of faces) total += fs ? fs.length : 0;
  const head = new TextEncoder().encode(workspace.encode({ ...rest, frames: times.length, total, hasState: !!state }));
  const start = Math.ceil((4 + head.length) / 8) * 8;
  const n = times.length;
  const bytes = new Uint8Array(start + n * 8 + total * NUMBERS * 8 + total * 4 + n * 2 + n);
  new DataView(bytes.buffer).setUint32(0, head.length, true);
  bytes.set(head, 4);
  let at = start;
  new Float64Array(bytes.buffer, at, n).set(times);
  at += n * 8;
  const numbers = new Float64Array(bytes.buffer, at, total * NUMBERS);
  at += total * NUMBERS * 8;
  const people = new Int32Array(bytes.buffer, at, total);
  at += total * 4;
  const counts = new Uint16Array(bytes.buffer, at, n);
  at += n * 2;
  if (state) bytes.set(state, at);
  let v = 0, k = 0;
  for (let i = 0; i < n; i++) {
    const fs = faces[i];
    counts[i] = fs ? fs.length : NO_FACES;
    for (const f of fs || []) {
      numbers.set(f.box, v);
      for (let j = 0; j < 5; j++) { numbers[v + 4 + j * 2] = f.kps[j][0]; numbers[v + 5 + j * 2] = f.kps[j][1]; }
      // (NaN: this face has none, as a photo's faces have no score)
      numbers[v + 14] = f.score ?? NaN;
      numbers[v + 15] = f.detail ?? NaN;
      people[k++] = f.person;
      v += NUMBERS;
    }
  }
  return bytes;
}

export function unpackFaces(file) {
  const headLength = new DataView(file).getUint32(0, true);
  const { frames: n, total, hasState, ...rest } = workspace.decode(new TextDecoder().decode(new Uint8Array(file, 4, headLength)));
  let at = Math.ceil((4 + headLength) / 8) * 8;
  const times = Array.from(new Float64Array(file, at, n));
  at += n * 8;
  const numbers = new Float64Array(file, at, total * NUMBERS);
  at += total * NUMBERS * 8;
  const people = new Int32Array(file, at, total);
  at += total * 4;
  const counts = new Uint16Array(file, at, n);
  at += n * 2;
  const faces = new Array(n);
  let v = 0, k = 0;
  for (let i = 0; i < n; i++) {
    if (counts[i] === NO_FACES) { faces[i] = null; continue; }
    const fs = faces[i] = new Array(counts[i]);
    for (let c = 0; c < fs.length; c++) {
      const f = fs[c] = {
        box: [numbers[v], numbers[v + 1], numbers[v + 2], numbers[v + 3]],
        kps: [[numbers[v + 4], numbers[v + 5]], [numbers[v + 6], numbers[v + 7]], [numbers[v + 8], numbers[v + 9]], [numbers[v + 10], numbers[v + 11]], [numbers[v + 12], numbers[v + 13]]],
        person: people[k++],
      };
      if (!Number.isNaN(numbers[v + 14])) f.score = numbers[v + 14];
      if (!Number.isNaN(numbers[v + 15])) f.detail = numbers[v + 15];
      v += NUMBERS;
    }
  }
  return { ...rest, times, faces, ...(hasState ? { state: new Uint8Array(file.slice(at, at + n)) } : {}) };
}

// ------------------------------------------------------------------ saving them

// A save takes in what's saved already, so they run one at a time.
let saving = Promise.resolve();
const inTurn = (fn) => {
  const run = saving.then(fn);
  saving = run.catch(() => {});
  return run;
};

// The segments and settings saved for a photo or video, without its faces, or null.
export const getEdits = (hash) => quiet("read a saved photo or video", () => inTurn(() => workspace.getRecord("targets", hash)), null);

// What was saved of a photo or video (analysis: its faces), or null.
export const getTarget = (hash) => quiet("read a saved photo or video", () => inTurn(async () => {
  const edits = await workspace.getRecord("targets", hash);
  if (!edits) return null;
  const file = await workspace.getFaces(hash);
  return file ? { ...edits, analysis: unpackFaces(file) } : edits;
}), null);

// Saves a photo's or video's faces (analysis) and their segments (people). A save of only the
// people keeps the analysis saved.
export const putTarget = (hash, { analysis, ...edits }) => quiet("save a photo or video", () => inTurn(async () => {
  const old = await workspace.getRecordNow("targets", hash);
  if (!old && !analysis) return; // its analysis was forgotten meanwhile
  if (analysis) await workspace.putFaces(hash, packFaces(analysis));
  await workspace.putRecord("targets", hash, { ...old, ...edits, hash });
}));

// -> { characters, targets }: how many of each are saved
export const summary = () => quiet("count the saved work", () => workspace.recordCounts(), { characters: 0, targets: 0 });

export const clearAll = () => quiet("forget the saved work", () => inTurn(() => workspace.clearRecords()));
