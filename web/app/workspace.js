// The page's side of the local server (server/src/api.rs): the models folder, and the folders
// where the user's files go. Works from the page and from the engine's worker. The site has its own
// version of this file, with the same exports (site/workspace.js).

import { encode, decode } from "./records.js";

const api =(path) => new URL(`/api/${path}`, self.location.origin);

async function ask(path, init) {
  let res;
  try {
    res = await fetch(api(path), init);
  } catch (e) {
    if (e.name === "AbortError") throw e;
    throw new TypeError("FaceJutsu has stopped. Start it again, then try again.");
  }
  if (!res.ok) throw new Error(`the local server said ${res.status} to ${init?.method || "GET"} ${path}`);
  return res;
}

// Does fn, what reaches the server, and again every second for as long as the server can't be
// reached (a failed fetch, or a body cut off, is a TypeError), until signal says to stop.
export async function untilBack(fn, signal) {
  for (;;) {
    signal?.throwIfAborted();
    try {
      return await fn();
    } catch (e) {
      if (!(e instanceof TypeError)) throw e;
    }
    await new Promise((done) => {
      const stop = () => { clearTimeout(timer); done(); };
      const timer = setTimeout(() => { signal?.removeEventListener("abort", stop); done(); }, 1000);
      signal?.addEventListener("abort", stop, { once: true });
    });
  }
}

// -> { app, version, build, models, outputs, work, cache } (the folders' paths)
export const status = async () => (await ask("status")).json();
// -> [{ name, size }]: the model files in the models folder
export const modelFiles = async () => (await ask("models")).json();
// -> { urls: { file name: address }, problem? }: from config.toml
export const modelUrls = async () => (await ask("models/urls")).json();
// Has the server download a model file into the models folder. onProgress(done, total): total may
// be null. Throws what went wrong.
export async function downloadModel(name, onProgress) {
  const res = await ask(`models/${encodeURIComponent(name)}/download`, { method: "POST" });
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let text = "";
  for (;;) {
    const { done, value } = await reader.read().catch(() => ({ done: true }));
    if (done) throw new Error("the download stopped");
    text += value;
    const lines = text.split("\n");
    text = lines.pop();
    for (const line of lines) {
      const m = JSON.parse(line);
      if (m.error) throw new Error(m.error);
      if (m.ok) return;
      onProgress?.(m.done, m.total);
    }
  }
}
// Calls onChange(up) when the server stops or is back. A socket is open to it while it runs, and
// closes the moment it stops; after that it's tried every second, at once when the tab is shown
// again (a hidden tab's timers can wait a minute).
export function watchServer(onChange) {
  let up = true, retry = 0;
  const connect = () => {
    retry = 0;
    const socket = new WebSocket(`ws://${location.host}/api/alive`);
    socket.onopen = () => {
      if (!up) onChange((up = true));
    };
    socket.onclose = () => {
      if (up) onChange((up = false));
      retry = setTimeout(connect, 1000);
    };
  };
  document.addEventListener("visibilitychange", () => {
    if (document.hidden || !retry) return;
    clearTimeout(retry);
    connect();
  });
  connect();
}

// Whether model files can be put in the models folder by hand.
export const modelsByHand = true;

// A model file's contents, as an answer to a fetch.
export const readModel = (url, init) => fetch(url, init);

// Shows "models" or "outputs" in the system's file manager.
export const openFolder = (what) => ask(`open/${what}`, { method: "POST" });

// ------------------------------------------------------------------ records, as JSON

export { encode, decode };

// -> the record, or null; throws when the server has stopped (a save, which is made again when it's back)
export async function getRecordNow(store, key) {
  const res = await ask(`work/${store}/${key}`);
  return res.status === 200 ? decode(await res.text()) : null;
}
// The same, waiting for a stopped server: null must mean there is none, or what's saved would be
// taken for nothing and saved over.
export const getRecord = (store, key) => untilBack(() => getRecordNow(store, key));
export const allRecords = async (store) => decode(await (await ask(`work/${store}`)).text());
export const putRecord = (store, key, value) => ask(`work/${store}/${key}`, { method: "PUT", body: encode(value) });
export const removeRecord = (store, key) => ask(`work/${store}/${key}`, { method: "DELETE" });
// -> { characters, targets }: how many of each
export const recordCounts = async () => (await ask("work")).json();
export const clearRecords = () => ask("work", { method: "DELETE" });

// ------------------------------------------------------------------ the faces found in a photo or video

// -> the file saved.js made of them (an ArrayBuffer), or null. Waits for a stopped server.
export const getFaces = (key) => untilBack(async () => {
  const res = await ask(`work/faces/${key}`);
  return res.status === 200 ? await res.arrayBuffer() : null;
});
export const putFaces = (key, body) => ask(`work/faces/${key}`, { method: "PUT", body });

// ------------------------------------------------------------------ swapped frames

// A patch's key is the folder of its photo or video, then its own name: "folder/name".
// -> the patch file's bytes, or null
export async function getPatch(key) {
  const res = await ask(`cache/${key}`);
  return res.status === 200 ? new Uint8Array(await res.arrayBuffer()) : null;
}
// -> the names of the patches kept in a folder
export const patchNames = async (folder) => (await ask(`cache/${folder}`)).json();
export const putPatch =(key, body) => ask(`cache/${key}`, { method: "PUT", body });
// Says what a photo's or video's patches are being made with now (plan: a name for it). When the
// cache is full, its patches from before the plan last changed are the first of it to go.
export const setPatchPlan = (folder, plan) => ask(`cache/${folder}`, { method: "POST", body: plan });
// -> { bytes, count }
export const patchUsage = async () => (await ask("cache")).json();
export const clearPatches = () => ask("cache", { method: "DELETE" });

// ------------------------------------------------------------------ outputs

// A file in the outputs, written as it's made: write({ position, data }), then close() (-> { name,
// path }: what it ended up as) or abort() (nothing is left). `name` is taken as a wish: the server
// keeps it from replacing a file already there. While the server is stopped these wait, and go on
// when it's back with what was written so far kept, until signal says to stop.
export async function openOutput(name, signal) {
  const send = (path, init) => untilBack(() => ask(path, { ...init, signal }), signal);
  const named = { headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) };
  const { id } = await (await send("outputs", { method: "POST", ...named })).json();
  return {
    write: ({ position, data }) => send(`outputs/${id}?at=${position}`, { method: "PUT", body: data }),
    close: async () => (await send(`outputs/${id}/close`, { method: "POST", ...named })).json(),
    abort: () => ask(`outputs/${id}/abort`, { method: "DELETE" }),
  };
}

// Called on the page with what an openOutput in the worker closed. (It's in the outputs already.)
export const handOver = () => {};

// The choices of where a video goes, when there are any: everything goes to the outputs here.
export const saveChoices = null;
// What to give openOutput for a file to be called `name`, of format `format`, as `mode` of
// saveChoices says. On the page, and the first thing after a click.
export const chooseOutput = async (name) => name;

// Saves several photos as one lot: add(blob, name) each, then finish() (-> what they were saved
// together as, if they were). Here each goes straight to the outputs.
export const outputSet = (signal) => ({
  add: (blob, name) => saveOutput(blob, name, signal),
  finish: async () => null,
});

// Saves a whole blob to the outputs. -> { name, path }
export async function saveOutput(blob, name, signal) {
  const out = await openOutput(name, signal);
  try {
    await out.write({ position: 0, data: blob });
    return await out.close();
  } catch (e) {
    await out.abort().catch(() => {});
    throw e;
  }
}
