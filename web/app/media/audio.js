// Audio for playback: Web Audio, scheduled a little ahead and kept in step with the frame-paced
// video clock by the player (which restarts it when it drifts).

const AHEAD = 0.6; // seconds of audio scheduled ahead of the output
const MUTED_KEY = "facejutsu.muted";

export class AudioPlayer {
  constructor() {
    this.ctx = null;
    this.gain = null;
    this.session = null;
    try { this._muted = localStorage.getItem(MUTED_KEY) === "1"; } catch { this._muted = false; }
  }

  get muted() { return this._muted; }
  set muted(m) {
    this._muted = m;
    try { localStorage.setItem(MUTED_KEY, m ? "1" : "0"); } catch { /* private mode */ }
    if (this.gain) this.gain.gain.setTargetAtTime(m ? 0 : 1, this.ctx.currentTime, 0.015);
  }

  get latency() {
    return (this.ctx.outputLatency || 0) + (this.ctx.baseLatency || 0);
  }

  // play from media time t, now (call from a user gesture the first time)
  start(stream, t) {
    this.stop();
    this.warm();
    if (this.ctx.state === "suspended") this.ctx.resume();
    // what is scheduled now is heard `latency` later, so start that far into the media
    const s = { stream, t0: t + this.latency, origin: this.ctx.currentTime, nodes: new Set(), dead: false };
    this.session = s;
    this.run(s).catch((e) => { if (!s.dead) console.warn("audio playback stopped", e); });
  }

  // Opens the audio output ahead of the first play: making the context takes ~100 ms (the device
  // opens), which would otherwise land on the press of play. It stays suspended until start()
  // resumes it, in that press.
  warm() {
    if (this.ctx) return;
    this.ctx = new AudioContext({ latencyHint: "interactive" });
    this.gain = this.ctx.createGain();
    this.gain.gain.value = this._muted ? 0 : 1;
    this.gain.connect(this.ctx.destination);
  }

  stop() {
    const s = this.session;
    if (!s) return;
    s.dead = true;
    for (const n of s.nodes) { try { n.stop(); } catch { /* not started */ } n.disconnect(); }
    s.nodes.clear();
    this.session = null;
  }

  // the media time being heard now, or null when not playing
  position() {
    const s = this.session;
    return s && this.ctx ? s.t0 + (this.ctx.currentTime - s.origin) - this.latency : null;
  }

  async run(s) {
    const ctx = this.ctx;
    for await (const c of s.stream.buffers(s.t0)) {
      if (s.dead) return;
      this.schedule(s, c);
      // stay a little ahead of the output, no further
      while (!s.dead && s.origin + (c.t + c.buf.duration - s.t0) - ctx.currentTime > AHEAD) await new Promise((r) => setTimeout(r, 50));
      if (s.dead) return;
    }
  }

  schedule(s, c) {
    const ctx = this.ctx;
    const a = Math.max(c.t, s.t0), b = c.t + c.buf.duration;
    if (b <= a) return;
    let when = s.origin + (a - s.t0), offset = a - c.t, dur = b - a;
    const late = ctx.currentTime - when;
    if (late > 0) {
      if (late >= dur) return;
      when += late; offset += late; dur -= late;
    }
    const node = ctx.createBufferSource();
    node.buffer = c.buf;
    node.connect(this.gain);
    node.start(when, offset, dur);
    s.nodes.add(node);
    node.onended = () => { s.nodes.delete(node); node.disconnect(); };
  }
}
