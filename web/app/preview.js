// The preview: the swapped faces of the frame on screen and the frames around it, made by the
// engine exactly as the export makes them (session.js render), and drawn over the original frame.
//
// Frames are made in batches, the most wanted first: the frame on screen, the next few seconds,
// a little behind it. While nothing is wanted there, the rest of the video is made in the
// background (kept by the engine, not sent here) so the export has less left to do. While playing,
// the frames ahead are only fetched when the engine has them: one is swapped when playback reaches
// it (swapping ahead would hold up the GPU, and so playback, for a few frames' gain). Each frame's
// patches are kept under a key of what they show (its faces' settings), so an edit only remakes
// the frames it changes.
//
// Only frames between the in and out points are made (playback loops there, and the export
// takes only those); what was made outside them is kept for when they move again.

const AHEAD = 3; // seconds made ahead of the playhead
const AHEAD_PLAYING = 4;
const NEAR = 0.3; // seconds ahead of the playhead swapped while playing
const BEHIND = 0.5; // and behind it
const BATCH = 12; // frames per engine call
const SPAN = 2; // seconds of video one batch may span (each one decodes from a keyframe)

export class Preview {
  // engine; hooks: { count(), range() -> [first, last] (the frames to make), fps(), index(f) (the analysed frame of frame f), key(f) (what
  // frame f's patches show, null: nothing to swap), opts(), prepare() (async: opens the target
  // in the engine; false when it can't be yet), canRun(), background() (whether the rest is made
  // in the background too), focus() -> { frame, playing }, onChange(f) }
  constructor(engine, hooks, { budget = 256 * 2 ** 20 } = {}) {
    this.engine = engine;
    this.hooks = hooks;
    this.budget = budget;
    this.made = new Map(); // frame -> { key, patches: [{ box, bitmap }] | null (kept by the engine only) }, in LRU order
    this.missing = new Map(); // frame -> the key the engine had no patches for (while playing)
    this.bytes = 0;
    this.keys = new Map(); // frame -> key (memo, until the plan changes)
    this.run = null; // the batch of frames to show: { ac, want: Set of frames not back yet, keys }
    this.bg = null; // the batch made in the background, the same
    this.original = false; // draw the frames as they are (hold to compare)
    this.error = null;
    this.waiters = [];
    this.kicking = false;
  }

  // ------------------------------------------------------------ state

  key(f) {
    if (!this.keys.has(f)) this.keys.set(f, this.hooks.key(f));
    return this.keys.get(f);
  }

  // what's to swap changed (the plan, the models, the characters): keys again
  invalidate() {
    this.keys.clear();
    this.error = null;
    this.kick();
  }

  // a new photo or video, or none
  reset() {
    this.stop();
    for (const e of this.made.values()) e.patches?.forEach((p) => p.bitmap.close());
    this.made.clear();
    this.missing.clear();
    this.bytes = 0;
    this.keys.clear();
    this.error = null;
  }

  inRange(f) {
    const [a, b] = this.hooks.range();
    return f >= a && f <= b;
  }
  needs(f) { return this.key(f) != null; }
  entry(f) {
    const e = this.made.get(f);
    return e && e.key === this.key(f) ? e : null;
  }
  // frame f can be shown as the export will have it
  ready(f) { return !this.needs(f) || !!this.entry(f)?.patches; }
  // 0: nothing to swap (or outside the in and out points, not made), 1: to make, 2: made (shown
  // or kept by the engine)
  state(f) { return !this.needs(f) ? 0 : this.entry(f) ? 2 : this.inRange(f) ? 1 : 0; }

  // changes when what draw() draws over frame f does
  token(f) {
    if (this.original || !this.needs(f)) return "";
    return this.entry(f)?.patches ? this.key(f) : null;
  }

  // draws frame f's patches (when made) onto ctx, over the original frame
  draw(ctx, f) {
    if (this.original) return false;
    const e = this.entry(f);
    if (!e?.patches) return false;
    for (const p of e.patches) ctx.drawImage(p.bitmap, p.box[0], p.box[1]);
    return true;
  }

  // -> a promise for when frame f can be shown (made), rejected when making it failed (after an
  // earlier failure, it tries again)
  ensure(f) {
    if (this.ready(f)) return Promise.resolve();
    this.error = null;
    return new Promise((resolve, reject) => {
      this.waiters.push({ f, resolve, reject });
      this.kick();
    });
  }

  settle() {
    this.waiters = this.waiters.filter((w) => {
      if (this.ready(w.f)) w.resolve();
      else if (this.error) w.reject(this.error);
      else return true;
      return false;
    });
  }

  put(f, key, patches) {
    const old = this.made.get(f);
    if (old) this.drop(f, old);
    this.made.set(f, { key, patches });
    if (patches) this.bytes += patches.reduce((a, p) => a + p.bitmap.width * p.bitmap.height * 4, 0);
    this.evict();
    this.hooks.onChange(f);
    this.settle();
  }

  drop(f, e) {
    if (e.patches) {
      this.bytes -= e.patches.reduce((a, p) => a + p.bitmap.width * p.bitmap.height * 4, 0);
      e.patches.forEach((p) => p.bitmap.close());
    }
    this.made.delete(f);
  }

  // over the budget: the patches of the frames used longest ago go (the engine keeps them)
  evict() {
    if (this.bytes <= this.budget) return;
    const { frame } = this.hooks.focus();
    const near = Math.ceil(AHEAD_PLAYING * this.hooks.fps());
    for (const [f, e] of this.made) {
      if (this.bytes <= this.budget) break;
      if (!e.patches || Math.abs(f - frame) <= near) continue;
      this.drop(f, e);
      this.made.set(f, { key: e.key, patches: null });
    }
  }

  // ------------------------------------------------------------ scheduling

  // the frames to make next, most wanted first: [{ frames, send }] (send: the patches are
  // wanted here, else only made); null when there's nothing to do
  next() {
    const n = this.hooks.count();
    if (!n) return null;
    const { frame, playing } = this.hooks.focus();
    const [lo, hi] = this.hooks.range();
    const fps = this.hooks.fps();
    const ahead = Math.min(hi, frame + Math.ceil((playing ? AHEAD_PLAYING : AHEAD) * fps));
    const behind = Math.max(lo, frame - Math.ceil(BEHIND * fps));
    const span = Math.ceil(SPAN * fps);
    // a batch from start, onwards in `dir`, of frames `todo` says are to make
    const take = (start, end, dir, todo) => {
      const out = [];
      for (let f = start; dir > 0 ? f <= end : f >= end; f += dir) {
        if (out.length >= BATCH || Math.abs(f - start) > span) break;
        if (todo(f)) out.push(f);
      }
      return out.sort((a, b) => a - b);
    };
    const inR = (f) => f >= lo && f <= hi;
    const toShow = (f) => inR(f) && this.needs(f) && !this.entry(f)?.patches;
    const toMake = (f) => inR(f) && this.needs(f) && !this.entry(f);
    const first = (from, to, dir, todo) => {
      for (let f = from; dir > 0 ? f <= to : f >= to; f += dir) if (todo(f)) return f;
      return -1;
    };
    let f;
    if (playing) {
      // what plays next, looping from the out point back to the in point: [from, to] stretches
      const upTo = (sec) => {
        const end = frame + Math.ceil(sec * fps);
        if (end <= hi || frame > hi) return [[frame, Math.min(n - 1, end)]];
        return [[frame, hi], [lo, Math.min(frame - 1, lo + end - hi - 1)]];
      };
      const nearBy = upTo(NEAR);
      for (const [a, b] of nearBy) {
        f = first(a, b, 1, toShow);
        if (f >= 0) return { frames: take(f, b, 1, toShow), send: true, swap: true };
      }
      const isNear = (g) => nearBy.some(([a, b]) => g >= a && g <= b);
      const kept = (g) => !isNear(g) && toShow(g) && this.missing.get(g) !== this.key(g);
      for (const [a, b] of upTo(AHEAD_PLAYING)) {
        f = first(a, b, 1, kept);
        if (f >= 0) return { frames: take(f, b, 1, kept), send: true, swap: false };
      }
      return null;
    }
    f = first(frame, ahead, 1, toShow);
    if (f >= 0) return { frames: take(f, ahead, 1, toShow), send: true, swap: true };
    f = first(frame - 1, behind, -1, toShow);
    if (f >= 0) return { frames: take(f, behind, -1, toShow), send: true, swap: true };
    if (this.hooks.background?.() === false) return null;
    f = first(Math.max(lo, ahead + 1), hi, 1, toMake);
    if (f >= 0) return { frames: take(f, hi, 1, toMake), send: false, swap: true };
    f = first(lo, Math.min(hi, behind - 1), 1, toMake);
    if (f >= 0) return { frames: take(f, behind - 1, 1, toMake), send: false, swap: true };
    return null;
  }

  // Something changed (the playhead, an edit, the engine became free): keep the batch running
  // when it still makes the most wanted frame, else start the most wanted one.
  kick() {
    if (this.kicking) return;
    this.kicking = true;
    queueMicrotask(() => {
      this.kicking = false;
      this.pump();
    });
  }

  stop() {
    this.run?.ac.abort();
    this.bg?.ac.abort();
  }

  // Two batches may run at once: one of frames to show (near the playhead) and one made in the
  // background. Frames to show never wait for the background: it's called off when they're wanted
  // (the engine sends kept frames without waiting for its GPU), and it doesn't run while playing.
  pump() {
    if (!this.hooks.canRun() || this.error) {
      this.stop();
      this.settle();
      return;
    }
    const job = this.next();
    const playing = this.hooks.focus().playing;
    if (this.bg && (job?.send || playing || this.hooks.background?.() === false)) this.bg.ac.abort();
    if (this.run) {
      // still useful: it holds the most wanted frame, with the keys it was asked for
      const r = this.run;
      const keep = job?.send && r.want.has(job.frames[0]) && (r.swap || !job.swap) && [...r.want].every((f) => r.keys.get(f) === this.key(f));
      if (!keep) r.ac.abort();
      return; // the next one starts when this one ends
    }
    if (!job) return;
    if (job.send) this.start(job, "run");
    else if (!this.bg && !playing) this.start(job, "bg");
  }

  // runs job in lane "run" or "bg"
  async start(job, lane) {
    const ac = new AbortController();
    const keys = new Map(job.frames.map((f) => [f, this.key(f)]));
    const run = this[lane] = { ac, want: new Set(job.frames), keys, swap: job.swap };
    try {
      if (!(await this.hooks.prepare())) return;
      if (ac.signal.aborted) return;
      // analysed frame -> the frame asked for
      const byIndex = new Map();
      for (const f of job.frames) if (!byIndex.has(this.hooks.index(f))) byIndex.set(this.hooks.index(f), f);
      await this.engine.render([...byIndex.keys()].sort((a, b) => a - b), { ...this.hooks.opts(), send: job.send, swap: job.swap }, {
        signal: ac.signal,
        onFrame: ({ i, patches, missing }) => {
          const f = byIndex.get(i);
          run.want.delete(f);
          if (missing) { if (f !== undefined) this.missing.set(f, keys.get(f)); return; }
          // a frame edited since it was asked for is made again (and nothing of a batch called off
          // is taken: the photo or video may have changed)
          if (ac.signal.aborted || f === undefined || keys.get(f) !== this.key(f)) { patches?.forEach((p) => p.bitmap.close()); return; }
          // (one shown meanwhile keeps its patches)
          if (!patches && this.entry(f)?.patches) return;
          this.put(f, keys.get(f), patches);
        },
      });
    } catch (e) {
      // (a lost GPU: the page starts the engine again and resets the preview)
      if (e.name !== "AbortError" && e.name !== "GPULostError") {
        console.error("the preview couldn't be made", e);
        this.error = e;
        this.settle();
      }
    } finally {
      if (this[lane] === run) this[lane] = null;
    }
    this.kick();
  }
}
