// Finding the faces of a video a stretch at a time, around the playhead, so they can be worked
// with long before the whole video has been looked at. A run around a point has two passes side
// by side, the GPU going to the most wanted work first (queue.js):
//
// - the full pass looks at every frame of a window around the point: the few seconds around it
//   first (PRIO.NEAR, taking turns with swapping them), then on from there, a few seconds ahead
//   and a few behind in turn (PRIO.WINDOW). It links the faces from frame to frame (tracks.js
//   Tracker) and gives each new tracklet the person its embedding is most like (one found
//   before, or a new one);
// - the skim looks at keyframes of the whole video (between the in and out points: they decode
//   on their own, so it's quick) about every SKIM_STEP seconds, coarse to fine and the near ones
//   first, and finds the people in it, by their ArcFace embeddings (PRIO.SKIM: once the window's
//   done).
//
// Frames looked at stay looked at: a later run (around another point, or the export, which looks
// at the frames it reaches that no run has) only looks at the rest. A stretch's faces go on into
// the faces looked at before and after it, so a face keeps its person across them.
//
// The analysis (kept here, mirrored by the page, saved with the work): { kind: "video", recognize,
// t0 (the media time of the first frame shown), duration, times (media seconds of every frame of
// the video, ascending), keys (the keyframes' indices), state (per frame: 0 not looked at, 1 by
// the skim, 2 by the full pass), faces (per frame: null, or [{ box, kps, score, person,
// detail (the face's own, pipeline faceDetails; the full pass measures it) }]),
// people (per person: { sum (of their normalized embeddings) | null, n, into (the person they were
// found to be, when the two are one: their faces keep their own) }) }

import { Tracker, normalize, dot, quality, nearest, SAME_PERSON } from "./tracks.js";
import { PRIO, NEAR_SECONDS } from "./queue.js";
import { Input, ALL_FORMATS, BlobSource, EncodedPacketSink, VideoSampleSink, CanvasSink } from "./vendor/mediabunny.min.mjs";

export const GAP = 12; // frames a face may go missing and stay on its track
const BACK = 60, AHEAD = 300; // seconds of the window behind and ahead of its point
const CHUNK = 5; // seconds looked at in one go, ahead or behind the point, past the near ones
const SKIM_STEP = 2; // seconds between the skim's keyframes, at the finest (or the video's keyframes')
const SKIM_BIAS = 30; // seconds: how much nearer the playhead the skim looks in finer steps first
const FLUSH = 150; // ms between the updates sent while a run goes on
const THUMB = 96; // px of a face's thumbnail

const cancelled = () => new DOMException("Cancelled", "AbortError");

// A square thumbnail of a face, a bit wider than its box, from a frame's image source.
export function thumbnail(src, f) {
  const [x1, y1, x2, y2] = f.box;
  const s = Math.round(Math.max(x2 - x1, y2 - y1) * 1.5);
  return createImageBitmap(src, Math.round((x1 + x2 - s) / 2), Math.round((y1 + y2 - s) / 2), s, s,
    { resizeWidth: THUMB, resizeHeight: THUMB, resizeQuality: "medium" });
}

// The frames [lo, hi) of the window around frame `at`: BACK seconds before it and AHEAD after,
// moved in from the ends of frames [from, to) (the whole video by default) to keep its length.
export function windowAround(times, at, from = 0, to = times.length) {
  const t = times[at], first = times[from], last = times[to - 1];
  let a = t - BACK, b = t + AHEAD;
  if (a < first) { b += first - a; a = first; }
  if (b > last) { a = Math.max(first, a - (b - last)); b = last; }
  return [after(times, a - 1e-6), after(times, b + 1e-6)];
}

// the index of the first of ascending `times` at or after t
function after(times, t) {
  let lo = 0, hi = times.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] < t) lo = mid + 1; else hi = mid;
  }
  return lo;
}

// The keyframes of frames [from, to) the skim looks at, in turn: the keyframes nearest the points
// of grids SKIM_STEP * 2^m seconds apart, coarse to fine. A point's turn is its distance from
// frame `at` (plus SKIM_BIAS) over its grid's step, so the finer points near `at` come before the
// coarse ones far from it.
export function skimOrder(times, keys, at, from, to) {
  const ks = keys.filter((k) => k >= from && k < to);
  if (!ks.length) return [];
  const first = times[from], last = times[to - 1], t = times[at];
  let top = SKIM_STEP;
  while (top * 2 <= last - first) top *= 2;
  const points = [];
  for (let d = top; d >= SKIM_STEP; d /= 2) {
    for (let m = 0; first + m * d <= last + 1e-9; m++) {
      if (d < top && m % 2 === 0) continue; // (a point of the coarser grid)
      const g = first + m * d;
      points.push({ g, turn: (Math.abs(g - t) + SKIM_BIAS) / d });
    }
  }
  points.sort((x, y) => x.turn - y.turn);
  const out = [], taken = new Set();
  for (const { g } of points) {
    // the keyframe nearest g
    let lo = 0, hi = ks.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (times[ks[mid]] < g) lo = mid + 1; else hi = mid;
    }
    const k = lo > 0 && g - times[ks[lo - 1]] < times[ks[lo]] - g ? ks[lo - 1] : ks[lo];
    if (!taken.has(k)) { taken.add(k); out.push(k); }
  }
  return out;
}

export class VideoScan {
  // Opens `file` to look at; analysis: what was looked at before (else a new one, of every frame
  // of the video). queue: the GPU's (queue.js GpuQueue).
  static async open(pipe, queue, file, analysis, { recognize, detectSize }) {
    const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS });
    try {
      const track = await input.getPrimaryVideoTrack();
      if (!track) throw new Error("This file has no video track.");
      if (!(await track.canDecode())) throw new Error("This browser can't decode this video's format.");
      // a rotated (phone) video is looked at upright, as the swap sees it: through a canvas
      const upright = !!((await track.getRotation()) || (await track.getFlip?.()));
      if (!analysis) {
        const times = [], keyTimes = [];
        for await (const p of new EncodedPacketSink(track).packets(undefined, undefined, { metadataOnly: true })) {
          times.push(p.timestamp);
          if (p.type === "key") keyTimes.push(p.timestamp);
        }
        if (!times.length) throw new Error("This video has no frames.");
        times.sort((x, y) => x - y); // (packets come in decoding order)
        const keys = [...new Set(keyTimes.map((t) => nearest(times, t)))].sort((x, y) => x - y);
        analysis = {
          kind: "video", recognize,
          t0: times.find((t) => t >= -1e-9) ?? times[0],
          duration: await track.computeDuration(),
          times, keys,
          state: new Uint8Array(times.length),
          faces: new Array(times.length).fill(null),
          people: [],
        };
      }
      return new VideoScan(pipe, queue, input, track, upright, analysis, detectSize);
    } catch (e) {
      input.dispose();
      throw e;
    }
  }

  constructor(pipe, queue, input, track, upright, analysis, detectSize) {
    Object.assign(this, { pipe, queue, input, track, upright, a: analysis, detectSize });
    this.recognize = analysis.recognize;
    this.centroids = []; // person -> their mean embedding (normalized), made when wanted
    this.seen = new Map(); // person (who they are: root()) -> Set of the frames they're in
    analysis.faces.forEach((fs, i) => { for (const r of fs || []) this.seenIn(r.person).add(i); });
    this.best = new Map(); // person -> quality of the view their thumbnail is of
    this.fresh = true; // the page hasn't had the analysis yet
    this.dirty = new Set(); // frames changed since the last update
    this.measuring = new Map(); // frame -> its faces' detail on the way (measure)
    this.dirtyPeople = new Set();
    this.thumbs = new Map(); // person -> a new thumbnail not sent yet
    this.hold = 0; // stretches whose faces may still change person (not sent until they're done)
  }

  close() {
    for (const b of this.thumbs.values()) b.close();
    this.thumbs.clear();
    this.input.dispose();
  }

  // Frame i's faces (recs) get their own detail once its read (pipeline faceDetails, a promise)
  // lands, and go to the page again.
  measure(i, recs, details) {
    const p = details
      .then((d) => {
        recs.forEach((r, j) => { r.detail = d[j]; });
        this.dirty.add(i);
      }, () => {}) // (none: the restore goes by the pixelshuffle alone)
      .finally(() => this.measuring.delete(i));
    this.measuring.set(i, p);
  }

  // (once frame i's faces have their detail, when it's on the way)
  measured(i) {
    return this.measuring.get(i);
  }

  // ------------------------------------------------------------ the update for the page

  // What changed since the last call: { open (the analysis without faces, the first time),
  // frames: [[i, state, faces]], people: [[person, sum, n, into]], thumbs: [[person, ImageBitmap]] }, or
  // null for nothing
  update() {
    if (this.recognize && this.dirtyPeople.size) this.regroup();
    if (!this.fresh && !this.dirty.size && !this.dirtyPeople.size && !this.thumbs.size) return null;
    const a = this.a;
    const u = {
      frames: [...this.dirty].map((i) => [i, a.state[i], a.faces[i]]),
      people: [...this.dirtyPeople].map((p) => [p, a.people[p].sum, a.people[p].n, a.people[p].into]),
      thumbs: [...this.thumbs],
    };
    if (this.fresh) {
      const { kind, recognize, t0, duration, times, keys } = a;
      u.open = { kind, recognize, t0, duration, times, keys };
      this.fresh = false;
    }
    this.dirty.clear();
    this.dirtyPeople.clear();
    this.thumbs.clear();
    return u;
  }

  // how much of frames [lo, hi) the full pass has looked at (0..1)
  coverage(lo, hi) {
    let n = 0;
    for (let i = lo; i < hi; i++) if (this.a.state[i] === 2) n++;
    return hi > lo ? n / (hi - lo) : 1;
  }

  // ------------------------------------------------------------ people

  newPerson() {
    this.a.people.push({ sum: null, n: 0 });
    this.dirtyPeople.add(this.a.people.length - 1);
    return this.a.people.length - 1;
  }

  // who person p is: the person they were found to be one with, or themselves
  root(p) {
    while (this.a.people[p].into != null) p = this.a.people[p].into;
    return p;
  }

  seenIn(p) {
    const r = this.root(p);
    if (!this.seen.has(r)) this.seen.set(r, new Set());
    return this.seen.get(r);
  }

  // adds embedding e to who person p is (or takes it away, sign -1)
  addEmbed(p, e, sign = 1) {
    p = this.root(p);
    const x = this.a.people[p];
    x.sum ||= new Float32Array(e.length);
    for (let k = 0; k < e.length; k++) x.sum[k] += sign * e[k];
    x.n += sign;
    if (x.n <= 0) { x.sum = null; x.n = 0; }
    this.centroids[p] = null;
    this.dirtyPeople.add(p);
  }

  // the person embedding e is of: the most like it of those not in `here`, or a new one
  match(e, here) {
    let best = SAME_PERSON, who = -1;
    this.a.people.forEach((x, p) => {
      if (!x.sum || x.into != null || here.has(p)) return;
      const c = (this.centroids[p] ||= normalize(x.sum));
      const s = dot(c, e);
      if (s >= best) { best = s; who = p; }
    });
    return who >= 0 ? who : this.newPerson();
  }

  // Tracklet t is someone else: person p. Its faces so far go to them.
  relabel(t, p) {
    if (t.person === p) return;
    for (const e of t.embeds) {
      this.addEmbed(t.person, e, -1);
      this.addEmbed(p, e);
    }
    for (const [i, r] of t.recs) {
      this.seenIn(r.person).delete(i);
      this.seenIn(p).add(i);
      r.person = p;
      this.dirty.add(i);
    }
    t.person = p;
  }

  // Finds the people found as two that are one (a face that turned away gets a new tracklet, and
  // its first face may look like no one yet): while the two whose mean embeddings are most alike
  // reach SAME_PERSON and are never on screen together, the later one is found to be the earlier.
  // (Looked at all at once, a video's people were grouped this way.)
  regroup() {
    const people = this.a.people;
    for (;;) {
      const live = [];
      people.forEach((x, p) => { if (x.sum && x.into == null) live.push(p); });
      const c = live.map((p) => (this.centroids[p] ||= normalize(people[p].sum)));
      const pairs = [];
      for (let i = 0; i < live.length; i++) {
        for (let j = i + 1; j < live.length; j++) {
          const s = dot(c[i], c[j]);
          if (s >= SAME_PERSON) pairs.push([s, live[i], live[j]]);
        }
      }
      pairs.sort((x, y) => y[0] - x[0]);
      const pair = pairs.find(([, p, q]) => {
        const [small, large] = [this.seenIn(p), this.seenIn(q)].sort((x, y) => x.size - y.size);
        for (const f of small) if (large.has(f)) return false;
        return true;
      });
      if (!pair) return;
      const [, p, q] = pair; // (p < q)
      const x = people[p], y = people[q];
      for (let k = 0; k < x.sum.length; k++) x.sum[k] += y.sum[k];
      x.n += y.n;
      y.into = p;
      for (const f of this.seen.get(q) || []) this.seenIn(p).add(f);
      this.seen.delete(q);
      this.centroids[p] = this.centroids[q] = null;
      this.dirtyPeople.add(p);
      this.dirtyPeople.add(q);
    }
  }

  // a new thumbnail of person p only for a clearly better view
  async thumb(p, f, src) {
    const q = quality(f), b = this.best.get(p);
    if (b && q <= b * 1.3) return;
    this.best.set(p, q);
    const bitmap = await thumbnail(src, f);
    this.thumbs.get(p)?.close();
    this.thumbs.set(p, bitmap);
  }

  // ------------------------------------------------------------ frames

  sink() {
    return this.upright ? new CanvasSink(this.track, { poolSize: 2 }) : new VideoSampleSink(this.track);
  }

  // the frames from `start` up to `end` seconds: { src, w, h, timestamp, close() }
  async *frames(sink, start, end) {
    if (this.upright) {
      for await (const c of sink.canvases(start, end)) yield { src: c.canvas, w: c.canvas.width, h: c.canvas.height, timestamp: c.timestamp, close() {} };
      return;
    }
    for await (const s of sink.samples(start, end)) {
      const vf = s.toVideoFrame();
      yield { src: vf, w: vf.displayWidth, h: vf.displayHeight, timestamp: s.timestamp, close() { vf.close(); s.close(); } };
    }
  }

  // the frame at `t` seconds (a keyframe decodes on its own), or null
  async frameAt(sink, t) {
    if (this.upright) {
      const c = await sink.getCanvas(t);
      return c && { src: c.canvas, w: c.canvas.width, h: c.canvas.height, timestamp: c.timestamp, close() {} };
    }
    const s = await sink.getSample(t);
    if (!s) return null;
    const vf = s.toVideoFrame();
    return { src: vf, w: vf.displayWidth, h: vf.displayHeight, timestamp: s.timestamp, close() { vf.close(); s.close(); } };
  }

  // ------------------------------------------------------------ runs

  // Looks at the window around `t` seconds (from the first frame shown) until the full pass has
  // looked at all of it, and skims the video. With `range` ([first, last] seconds, the frames
  // starting then), both stay in those frames, the window around t's (or the nearest end's).
  // Without `whole`, the window is only the frames near t (NEAR_SECONDS): the GPU works no further
  // ahead than the playhead needs. onUpdate(update(), coverage)
  // as it goes, and once more at the end (cancelled too).
  async run(t, { range, whole = true, signal, onUpdate }) {
    const a = this.a;
    const [from, to] = range ? [nearest(a.times, a.t0 + range[0]), nearest(a.times, a.t0 + range[1]) + 1] : [0, a.times.length];
    const at = Math.min(Math.max(nearest(a.times, a.t0 + t), from), to - 1);
    const [lo, hi] = whole ? windowAround(a.times, at, from, to)
      : [Math.max(from, after(a.times, a.times[at] - NEAR_SECONDS.behind)), Math.min(to, after(a.times, a.times[at] + NEAR_SECONDS.ahead))];
    const ac = new AbortController(); // (one pass failing stops the other)
    const stop = () => ac.abort();
    signal?.addEventListener("abort", stop, { once: true });
    const send = () => {
      const u = this.update();
      if (u) onUpdate?.(u, this.coverage(lo, hi));
    };
    const timer = setInterval(() => { if (!this.hold) send(); }, FLUSH);
    send();
    const passes = [this.fullPass(at, lo, hi, ac.signal)];
    if (this.recognize) passes.push(this.skim(at, from, to, ac.signal));
    try {
      const done = await Promise.allSettled(passes.map((p) => p.catch((e) => { ac.abort(); throw e; })));
      if (signal?.aborted) throw cancelled();
      const failed = done.find((d) => d.status === "rejected");
      if (failed) throw failed.reason;
      return { coverage: this.coverage(lo, hi) };
    } finally {
      clearInterval(timer);
      signal?.removeEventListener("abort", stop);
      await Promise.all(this.measuring.values());
      send();
    }
  }

  // Every frame of [lo, hi): the near ones first (NEAR_SECONDS around `at`), then a CHUNK ahead
  // and a CHUNK behind in turn, on out from them.
  async fullPass(at, lo, hi, signal) {
    const times = this.a.times, t = times[at];
    const ahead = Math.min(hi, after(times, t + NEAR_SECONDS.ahead)), back = Math.max(lo, after(times, t - NEAR_SECONDS.behind));
    await this.cover(at, ahead, { signal, prio: PRIO.NEAR });
    await this.cover(back, at, { signal, prio: PRIO.NEAR, hold: true });
    for (let f = ahead, b = back; f < hi || b > lo;) {
      if (f < hi) {
        const e = Math.min(hi, after(times, times[f] + CHUNK));
        await this.cover(f, e, { signal, prio: PRIO.WINDOW });
        f = e;
      }
      if (b > lo) {
        const s = Math.max(lo, after(times, times[b - 1] - CHUNK));
        await this.cover(s, b, { signal, prio: PRIO.WINDOW, hold: true });
        b = s;
      }
    }
  }

  // the frames of [lo, hi) the full pass hasn't looked at, a stretch at a time
  async cover(lo, hi, opts) {
    const a = this.a;
    for (let i = lo; i < hi;) {
      while (i < hi && a.state[i] === 2) i++;
      let j = i;
      while (j < hi && a.state[j] !== 2) j++;
      if (j > i) await this.stretch(i, j, opts);
      i = j;
    }
  }

  // The skim: the keyframes of [from, to) not looked at yet, in skimOrder's turn.
  async skim(at, from, to, signal) {
    const a = this.a;
    const keys = skimOrder(a.times, a.keys, at, from, to).filter((k) => !a.state[k]);
    if (!keys.length) return;
    const release = this.queue.claim(PRIO.SKIM);
    const sink = this.sink();
    let frame = null;
    try {
      for (const k of keys) {
        if (signal.aborted) throw cancelled();
        if (a.state[k]) continue;
        const item = await this.frameAt(sink, a.times[k]);
        if (!item) continue;
        try {
          await this.queue.run(async () => {
            if (signal.aborted) throw cancelled();
            if (a.state[k]) return;
            frame = this.pipe.upload(item.src, item.w, item.h, frame);
            await this.glance(k, frame, item.src);
          }, PRIO.SKIM);
        } finally {
          item.close();
        }
      }
    } finally {
      release();
      if (frame) await this.queue.run(() => this.pipe.destroyFrame(frame));
    }
  }

  // The skim's look at keyframe k: its faces and who they are.
  async glance(k, frame, src) {
    const found = await this.pipe.detect(frame, { size: this.detectSize });
    const here = new Set(), recs = [];
    for (const f of found) {
      const e = normalize(await this.pipe.embed(frame, f.kps));
      const p = this.match(e, here);
      this.addEmbed(p, e);
      here.add(p);
      await this.thumb(p, f, src);
      recs.push({ box: f.box, kps: f.kps, score: f.score, person: p });
      this.seenIn(p).add(k);
    }
    this.a.faces[k] = recs;
    this.a.state[k] = 1;
    this.dirty.add(k);
  }

  // The full pass over frames [lo, hi) (none of them looked at by it yet), going on from the
  // faces just before them and into the ones just after. With hold, nothing is sent until it's
  // done: its faces may still go to the people after it. It claims the GPU at `prio` while it goes on.
  async stretch(lo, hi, { signal, prio, hold = false }) {
    const a = this.a, n = a.times.length;
    const pass = this.pass();
    for (let j = Math.max(0, lo - GAP); j < lo; j++) if (a.state[j] === 2) pass.known(j);
    if (hold) this.hold++;
    const release = this.queue.claim(prio);
    const sink = this.sink();
    let frame = null;
    try {
      for await (const item of this.frames(sink, a.times[lo], a.times[hi - 1] + 1e-4)) {
        try {
          if (signal.aborted) throw cancelled();
          const i = nearest(a.times, item.timestamp);
          if (i < lo) continue;
          if (i >= hi) break;
          if (a.state[i] === 2) { pass.known(i); continue; }
          await this.queue.run(async () => {
            if (signal.aborted) throw cancelled();
            frame = this.pipe.upload(item.src, item.w, item.h, frame);
            await pass.look(i, frame, item.src);
          }, prio);
        } finally {
          item.close();
        }
      }
      for (let j = hi; j < Math.min(n, hi + GAP) && a.state[j] === 2; j++) pass.known(j);
    } finally {
      if (hold) this.hold--;
      release();
      if (frame) await this.queue.run(() => this.pipe.destroyFrame(frame));
    }
  }

  // A walk over frames in order (a stretch of the full pass, or the export), linking their faces.
  pass() {
    const scan = this, a = this.a, tracker = new Tracker(GAP);
    return {
      // Frame i, looked at by the full pass before: its faces go on into the tracklets, and a
      // tracklet of this walk they continue is their person.
      known(i) {
        const fs = a.faces[i].map((r) => ({ box: r.box, rec: r }));
        tracker.push(i, fs);
        for (const f of fs) {
          const t = f.tracklet, p = f.rec.person;
          if (t.person == null) { t.person = p; t.known = true; continue; }
          if (scan.root(t.person) === scan.root(p) || t.known) continue;
          // (unless that'd put them on screen twice)
          const twice = t.recs.some(([j, r]) => a.faces[j].some((x) => x !== r && scan.root(x.person) === scan.root(p)));
          if (!twice) scan.relabel(t, p);
          t.known = true;
        }
      },

      // Frame i, uploaded to `frame` (src: its image), not looked at by the full pass yet.
      async look(i, frame, src) {
        const loose = a.state[i] === 1 ? a.faces[i] : null;
        const found = loose
          ? loose.map((r) => ({ box: r.box, kps: r.kps, score: r.score, person: r.person }))
          : await scan.pipe.detect(frame, { size: scan.detectSize });
        tracker.push(i, found);
        // the tracklets going on first: their people are taken here
        const order = [...found].sort((x, y) => (x.tracklet.person == null) - (y.tracklet.person == null));
        const here = new Set();
        for (const f of order) {
          const t = f.tracklet;
          t.recs ||= [];
          if (t.person != null) t.person = scan.root(t.person);
          if (scan.recognize && !loose && tracker.wantsEmbedding(t)) {
            const e = normalize(await scan.pipe.embed(frame, f.kps));
            t.embeds.push(e);
            if (t.person != null) scan.addEmbed(t.person, e);
          }
          // a new tracklet is who the skim took it for, or who it looks like; one going on
          // is who it was (a tracklet that'd put them on screen twice is someone else)
          if (t.person == null || here.has(t.person)) {
            const last = t.embeds[t.embeds.length - 1];
            const was = loose && scan.root(f.person);
            const p = loose && !here.has(was) ? was : scan.recognize && last ? scan.match(last, here) : scan.newPerson();
            if (t.person == null) {
              t.person = p;
              for (const x of t.embeds) scan.addEmbed(p, x);
            } else scan.relabel(t, p);
          }
          here.add(t.person);
          f.person = t.person;
          await scan.thumb(t.person, f, src);
        }
        // each face's own detail (the enhancer's Detail restore), measured while the frame is on
        // the GPU; its read lands later, without holding up the scan
        const details = scan.pipe.faceDetails(frame, found.map((f) => f.kps));
        const recs = found.map((f) => {
          const r = { box: f.box, kps: f.kps, score: f.score, person: f.person };
          f.tracklet.recs.push([i, r]);
          scan.seenIn(f.person).add(i);
          return r;
        });
        a.faces[i] = recs;
        a.state[i] = 2;
        scan.dirty.add(i);
        scan.measure(i, recs, details);
      },
    };
  }
}
