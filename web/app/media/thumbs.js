// Filmstrip thumbnails, one per GOP: only keyframes are decoded, so this stays cheap on long videos.

export class ThumbServer {
  constructor(demux, height = 72) {
    const { rawW, rawH } = demux.info;
    this.demux = demux;
    this.height = height;
    this.width = Math.max(8, Math.round((height * rawW) / rawH));
    this.onThumb = () => {};
    this.cache = new Map(); // GOP ordinal -> ImageBitmap (raw orientation)
    this.queue = [];
    this.busy = false;
    this.failed = false;
    this.disposed = false;
    this.paused = false;
    this.decoder = null;
    this.outputs = [];
  }

  // hold off while playing, so thumbnails don't compete with playback for the decoder
  setPaused(p) {
    this.paused = p;
    if (!p) this.run();
  }

  // the thumbnail of the GOP holding frame idx
  get(idx) {
    return this.cache.get(this.demux.index.gop[idx]);
  }

  // replace the queue with these frames, most important first
  request(indices) {
    if (this.failed) return;
    const gops = new Set();
    this.queue = [];
    for (const i of indices) {
      const g = this.demux.index.gop[i];
      if (!this.cache.has(g) && !gops.has(g)) { gops.add(g); this.queue.push(i); }
    }
    this.run();
  }

  dispose() {
    this.disposed = true;
    if (this.decoder && this.decoder.state !== "closed") this.decoder.close();
    for (const b of new Set(this.cache.values())) b.close();
    this.cache.clear();
  }

  ensureDecoder() {
    if (this.decoder && this.decoder.state === "configured") return this.decoder;
    this.decoder = new VideoDecoder({
      output: (frame) => {
        const idx = this.demux.index.byUs.get(frame.timestamp);
        const p = createImageBitmap(frame, { resizeWidth: this.width, resizeHeight: this.height, resizeQuality: "medium" })
          .then((b) => {
            if (idx === undefined || this.disposed) { b.close(); return; }
            this.cache.set(this.demux.index.gop[idx], b);
          })
          .catch(() => {})
          .finally(() => frame.close());
        this.outputs.push(p);
      },
      error: (e) => { console.warn("thumbnail decoder", e); this.decoder = null; },
    });
    this.decoder.configure(this.demux.config);
    return this.decoder;
  }

  async run() {
    if (this.busy || this.disposed) return;
    this.busy = true;
    let errors = 0;
    try {
      while (this.queue.length && !this.disposed && !this.paused) {
        // a few keyframes per flush: each one decodes on its own
        const batch = this.queue.splice(0, 4);
        try {
          const decoder = this.ensureDecoder();
          for (const idx of batch) {
            const chunk = await this.demux.keyChunk(idx);
            if (chunk && decoder.state === "configured") decoder.decode(chunk);
          }
          if (decoder.state === "configured") await decoder.flush();
          await Promise.all(this.outputs);
          this.outputs = [];
          // a keyframe that came back under another timestamp borrows a neighbour's thumbnail,
          // so it isn't retried forever
          for (const idx of batch) {
            const g = this.demux.index.gop[idx];
            if (!this.cache.has(g)) {
              for (let d = 1; d < 8; d++) {
                const b = this.cache.get(g - d) ?? this.cache.get(g + d);
                if (b) { this.cache.set(g, b); break; }
              }
            }
          }
          this.onThumb();
        } catch (e) {
          console.warn("thumbnail", e);
          this.decoder = null;
          if (++errors > 5) { this.failed = true; return; }
        }
      }
    } finally {
      this.busy = false;
    }
  }
}
