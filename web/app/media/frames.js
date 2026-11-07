// Decodes video frames on demand (WebCodecs) and keeps them as full-resolution ImageBitmaps.
//
// One decode session runs at a time: it starts at the keyframe a wanted frame needs and feeds
// packets in decode order until every wanted frame has come out. Wanted frames ahead in the same
// or the next GOP come from the running session, so playback and stepping forward never
// re-decode; anything else restarts the session at the right keyframe. Frames nobody asked for
// are closed straight away without a copy.

export class FrameServer {
  constructor(demux, budget = 512 * 1024 * 1024) {
    this.demux = demux;
    this.budget = budget;
    this.onFrame = () => {};
    this.onError = () => {};
    this.cache = new Map(); // in LRU order
    this.bytes = 0;
    this.goal = [];
    this.goalSet = new Set();
    this.goalMax = -1; // highest wanted frame; a few frames past it are kept too (see output)
    this.waiters = new Map();
    this.inflight = new Map();
    this.missing = new Set();
    this.retries = new Map();
    this.decoder = null;
    this.iter = null;
    this.session = 0;
    this.startGop = -1;
    this.fedGop = -1;
    this.lastOutUs = -Infinity;
    this.seen = new Set();
    this.ended = false;
    this.pumping = false;
    this.disposed = false;
  }

  // how many frames fit in the cache budget
  get capacity() {
    return Math.max(4, Math.floor(this.budget / (this.demux.info.rawW * this.demux.info.rawH * 4)));
  }

  get(idx) {
    const b = this.cache.get(idx);
    if (b) { this.cache.delete(idx); this.cache.set(idx, b); }
    return b;
  }

  has(idx) { return this.cache.has(idx); }

  // replace the wanted set; order is priority: the frame on screen first, then what plays next
  want(indices) {
    const seen = new Set();
    this.goal = [];
    for (const i of indices) if (i >= 0 && i < this.demux.index.count && !seen.has(i)) { seen.add(i); this.goal.push(i); }
    for (const i of this.waiters.keys()) if (!seen.has(i)) { seen.add(i); this.goal.push(i); }
    this.goalSet = seen;
    this.goalMax = this.goal.length ? Math.max(...this.goal) : -1;
    this.pump();
  }

  request(idx) {
    const b = this.get(idx);
    if (b) return Promise.resolve(b);
    if (this.missing.has(idx)) return Promise.resolve(null);
    return new Promise((resolve) => {
      const list = this.waiters.get(idx) ?? [];
      list.push(resolve);
      this.waiters.set(idx, list);
      if (!this.goalSet.has(idx)) { this.goal.unshift(idx); this.goalSet.add(idx); }
      this.pump();
    });
  }

  dispose() {
    this.disposed = true;
    this.session++;
    this.iter?.return(undefined);
    if (this.decoder && this.decoder.state !== "closed") this.decoder.close();
    for (const b of this.cache.values()) b.close();
    this.cache.clear();
    this.bytes = 0;
    for (const list of this.waiters.values()) list.forEach((r) => r(null));
    this.waiters.clear();
  }

  resolve(idx, b) {
    const list = this.waiters.get(idx);
    if (list) { this.waiters.delete(idx); list.forEach((r) => r(b)); }
  }

  put(idx, bmp) {
    const old = this.cache.get(idx);
    if (old) { this.bytes -= old.width * old.height * 4; old.close(); this.cache.delete(idx); }
    this.cache.set(idx, bmp);
    this.bytes += bmp.width * bmp.height * 4;
    this.retries.clear(); // progress was made, so earlier restarts weren't a failure loop
    this.evict();
    this.resolve(idx, bmp);
    this.onFrame(idx);
  }

  evict() {
    if (this.bytes <= this.budget) return;
    for (const [i, b] of this.cache) {
      if (this.bytes <= this.budget) break;
      if (this.goalSet.has(i)) continue;
      this.bytes -= b.width * b.height * 4; // before close: a closed bitmap reports 0×0
      b.close();
      this.cache.delete(i);
    }
  }

  markMissing(idx) {
    this.missing.add(idx);
    this.resolve(idx, null);
    this.onFrame(idx);
  }

  output(frame, session) {
    if (session !== this.session) { frame.close(); return; }
    const us = frame.timestamp;
    if (us > this.lastOutUs) this.lastOutUs = us;
    const idx = this.demux.index.byUs.get(us);
    if (idx === undefined) { frame.close(); return; }
    this.seen.add(idx);
    // the decoder hands frames over in bursts, so the frames just past the wanted range arrive
    // with the last wanted one: keep them, playback wants them a moment later
    const ahead = idx > this.goalMax && idx <= this.goalMax + 12;
    if (this.cache.has(idx) || this.inflight.has(idx) || !(this.goalSet.has(idx) || this.waiters.has(idx) || ahead)) {
      frame.close();
      return;
    }
    const p = createImageBitmap(frame)
      .then((bmp) => { if (this.disposed) bmp.close(); else this.put(idx, bmp); })
      .catch((e) => console.warn("frame copy failed", e))
      .finally(() => { frame.close(); this.inflight.delete(idx); });
    this.inflight.set(idx, p);
  }

  reachable(target) {
    const { gop, us } = this.demux.index;
    return !!this.decoder && this.decoder.state === "configured" && !!this.iter && !this.ended
      && gop[target] >= this.startGop && gop[target] <= this.fedGop + 1 && us[target] > this.lastOutUs;
  }

  restart(target) {
    this.session++;
    const session = this.session;
    this.iter?.return(undefined);
    if (!this.decoder || this.decoder.state === "closed") {
      this.decoder = new VideoDecoder({
        output: (f) => this.output(f, this.session),
        error: (e) => {
          console.warn("decoder error", e);
          this.onError(`Decoder error: ${e.message}`);
          if (this.session === session) this.decoder = null;
        },
      });
    } else {
      this.decoder.reset();
    }
    this.decoder.configure(this.demux.config);
    this.iter = this.demux.packets(target);
    this.startGop = this.demux.index.gop[target];
    this.fedGop = -1;
    this.lastOutUs = -Infinity;
    this.seen.clear();
    this.ended = false;
  }

  async pump() {
    if (this.pumping || this.disposed) return;
    this.pumping = true;
    try {
      for (;;) {
        if (this.disposed) return;
        const target = this.goal.find((i) => !this.cache.has(i) && !this.missing.has(i));
        if (target === undefined) return;
        const flight = this.inflight.get(target);
        if (flight) { await flight; continue; }
        if (!this.reachable(target)) {
          const { us, gop } = this.demux.index;
          // the running session went past it without the decoder ever producing it
          const skipped = this.decoder && this.iter && us[target] <= this.lastOutUs && !this.seen.has(target) && gop[target] >= this.startGop;
          const tries = (this.retries.get(target) ?? 0) + 1;
          this.retries.set(target, tries);
          if (skipped || tries > 3) { this.markMissing(target); continue; }
          this.restart(target);
        }
        await this.feed();
      }
    } catch (e) {
      console.error(e);
      this.onError(String(e?.message ?? e));
      this.decoder = null;
    } finally {
      this.pumping = false;
    }
  }

  async feed() {
    const decoder = this.decoder, session = this.session;
    if (decoder.decodeQueueSize > 4) {
      await new Promise((r) => {
        const t = setTimeout(r, 20);
        decoder.addEventListener("dequeue", () => { clearTimeout(t); r(); }, { once: true });
      });
      return;
    }
    const { done, value } = await this.iter.next();
    if (session !== this.session || decoder.state !== "configured") return;
    if (done) {
      this.ended = true;
      await decoder.flush().catch(() => {});
      await Promise.all(this.inflight.values());
      // anything wanted from this session that never came out doesn't exist
      for (const i of this.goal) {
        if (!this.cache.has(i) && this.demux.index.gop[i] >= this.startGop && !this.inflight.has(i) && !this.seen.has(i)) this.markMissing(i);
      }
      return;
    }
    const idx = this.demux.index.byUs.get(value.timestamp);
    if (idx !== undefined) this.fedGop = Math.max(this.fedGop, this.demux.index.gop[idx]);
    decoder.decode(value);
  }
}
