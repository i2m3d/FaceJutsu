// The faces of a video, told apart: a tracker links each frame's detections into tracklets (a
// face's box overlapping its box in a recent frame), and each tracklet is given a person as it
// starts, the one its ArcFace embedding is most like (scanner.js). Without embeddings each
// tracklet is its own person. Runs on the CPU over the detector's output.

import { iou, area } from "./face.js";

const LINK_IOU = 0.3; // overlap with a tracklet's last box that continues it
export const SAME_PERSON = 0.4; // cosine similarity of a face to a person's mean embedding that makes it them
const EMBEDS = 6; // embeddings per tracklet at most...
const EMBED_EVERY = 12; // ...one every this many of its frames
export const MIN_FRAMES = 3; // people seen in fewer frames are taken for false detections and not shown

export function normalize(v) {
  let n = 0;
  for (let i = 0; i < v.length; i++) n += v[i] * v[i];
  n = Math.sqrt(n) || 1;
  return Float32Array.from(v, (x) => x / n);
}

export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

// how good a view of the face is for its thumbnail: big and confidently detected
export const quality = (f) => area(f) * f.score;

export class Tracker {
  // maxGap: frames a tracklet survives without a detection (a missed frame, a hand passing by)
  constructor(maxGap = 12) {
    this.maxGap = maxGap;
    this.tracklets = []; // { id, frames: [frame index], box, last, embeds: [Float32Array], person }
    this.active = [];
  }

  // Links the faces of frame i (in order of frames) to tracklets: sets f.tracklet on each.
  push(i, faces) {
    this.active = this.active.filter((t) => i - t.last <= this.maxGap);
    const pairs = [];
    faces.forEach((f, fi) => {
      for (const t of this.active) {
        const o = iou(f.box, t.box);
        if (o >= LINK_IOU) pairs.push([o, fi, t]);
      }
    });
    pairs.sort((a, b) => b[0] - a[0]);
    const done = new Set(), taken = new Set();
    for (const [, fi, t] of pairs) {
      if (done.has(fi) || taken.has(t)) continue;
      done.add(fi);
      taken.add(t);
      faces[fi].tracklet = t;
    }
    faces.forEach((f, fi) => {
      if (done.has(fi)) return;
      f.tracklet = { id: this.tracklets.length, frames: [], box: null, last: i, embeds: [], person: null };
      this.tracklets.push(f.tracklet);
      this.active.push(f.tracklet);
    });
    for (const f of faces) {
      const t = f.tracklet;
      t.frames.push(i);
      t.box = f.box;
      t.last = i;
    }
  }

  // whether the tracklet wants an embedding of its face in the frame just pushed
  wantsEmbedding(t) {
    return t.embeds.length < EMBEDS && (t.frames.length - 1) % EMBED_EVERY === 0;
  }
}

// The stretches each person of an analysis (scanner.js) is on screen in, from the frames looked at
// so far: person -> { frames (how many), loose (seen on a keyframe the full pass hasn't reached
// yet), runs: [[first, last + 1)] frame indices }. A run bridges up to maxGap frames looked at
// without the face, and frames not looked at for up to `span` seconds (between the keyframes the
// loose pass looks at).
export function stretches(a, { maxGap, span }) {
  const out = new Map();
  let k = 0; // frames looked at before frame i
  for (let i = 0; i < a.faces.length; i++) {
    const fs = a.faces[i];
    if (!fs) continue;
    for (const f of fs) {
      if (f.person < 0) continue;
      let p = out.get(f.person);
      if (!p) out.set(f.person, (p = { frames: 0, loose: false, runs: [], last: -1, k: -1 }));
      p.frames++;
      if (a.state[i] === 1) p.loose = true;
      const r = p.runs[p.runs.length - 1];
      const looked = k - p.k - 1, skipped = i - p.last - 1 - looked;
      if (r && looked <= maxGap && (!skipped || a.times[i] - a.times[p.last] <= span)) r[1] = i + 1;
      else p.runs.push([i, i + 1]);
      p.last = i;
      p.k = k;
    }
    k++;
  }
  for (const p of out.values()) { delete p.last; delete p.k; }
  return out;
}

// index of the value in ascending `times` nearest to t
export function nearest(times, t) {
  let lo = 0, hi = times.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] < t) lo = mid + 1; else hi = mid;
  }
  return lo > 0 && t - times[lo - 1] < times[lo] - t ? lo - 1 : lo;
}
