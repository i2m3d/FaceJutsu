// Plays a video frame-exactly on a <canvas>: frames come from a WebCodecs FrameServer, so
// seeking shows the frame (or the nearest decoded one) right away, and playback advances by whole
// frames, waiting for one that isn't decoded yet rather than skipping it. Audio plays through
// Web Audio and is restarted at the picture's time when it drifts.
//
// An overlay (the swapped faces of the preview) is drawn over each frame; with canShow set,
// playback also waits for a frame until canShow says it's ready.
//
// In and out points mark the part of the clip that plays: playback starts at the in point and
// loops back to it after the out point. Without them it's the whole clip.
//
// Frames are numbered from the first one meant to be shown (0 .. count - 1).

import { openVideo } from "./media/demux.js";
import { FrameServer } from "./media/frames.js";
import { ThumbServer } from "./media/thumbs.js";
import { AudioPlayer } from "./media/audio.js";

const AUDIO_DRIFT = 0.1; // seconds of audio/picture drift before the audio is restarted

// raw decoded pixels -> display-oriented pixels
export function rawToDisplay({ rawW, rawH, rotation, flip }) {
  let m;
  switch (rotation) {
    case 90: m = new DOMMatrix([0, 1, -1, 0, rawH, 0]); break;
    case 180: m = new DOMMatrix([-1, 0, 0, -1, rawW, rawH]); break;
    case 270: m = new DOMMatrix([0, -1, 1, 0, 0, rawW]); break;
    default: m = new DOMMatrix();
  }
  if (flip) m = new DOMMatrix([-1, 0, 0, 1, rotation % 180 ? rawH : rawW, 0]).multiply(m);
  return m;
}

export class Player {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.media = null; // { demux, frames, thumbs, info, fps, count, first }
    this.playhead = 0;
    this.tracking = false; // the playhead marks where an export is (track), not what's shown
    this.range = null; // { a, b }: the in and out points (frames, both shown); null: the whole clip
    this.play = null; // { nextDue, audioSynced, stalled (waiting for a frame) }
    this.audio = new AudioPlayer();
    this.drawn = null; // the bitmap on the canvas
    this.drawnOver = null; // and the overlay's token for it
    // { token(f) (changes when what it draws over frame f does), draw(ctx, f) }, or null
    this.overlay = null;
    this.canShow = null; // (f) -> whether playback may show frame f yet; null: once decoded
    this.drawers = [];
    this.beforeRange = null; // () called before setIn, setOut or setRange changes the in and out points
    this.requested = false;
    this.opening = 0;
  }

  get total() { return this.media?.count ?? 0; }
  get fps() { return this.media?.fps ?? 30; }
  get playing() { return !!this.play; }
  get duration() { return this.media ? this.time(this.total) : 0; }
  get inPoint() { return this.range ? this.range.a : 0; }
  get outPoint() { return this.range ? this.range.b : Math.max(0, this.total - 1); }
  inRange(f) { return f >= this.inPoint && f <= this.outPoint; }
  get muted() { return this.audio.muted; }
  set muted(m) { this.audio.muted = m; this.invalidate(); }

  // seconds from the first frame to the start of frame f (f = total: the end of the video)
  time(f) {
    const m = this.media;
    if (!m) return 0;
    const { pts, end } = m.demux.index;
    return (f >= m.count ? end : pts[m.first + Math.max(0, f)]) - pts[m.first];
  }

  // the frame showing at `sec` seconds from the first frame
  frameAt(sec) {
    const m = this.media;
    if (!m) return 0;
    const { pts } = m.demux.index, t = sec + pts[m.first];
    let lo = 0, hi = m.count - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (pts[m.first + mid] <= t + 1e-9) lo = mid; else hi = mid - 1;
    }
    return lo;
  }

  // called once per animation frame when anything visible changed
  onDraw(fn) { this.drawers.push(fn); }

  invalidate() {
    if (this.requested) return;
    this.requested = true;
    requestAnimationFrame((t) => this.frame(t));
  }

  // file: a File or Blob; keepFrame: stay on the same frame number (the swapped result of the
  // video on screen). -> { width, height, duration, fps, count }
  async open(file, keepFrame = false) {
    const token = ++this.opening;
    const demux = await openVideo(file);
    if (token !== this.opening) { demux.dispose(); throw Object.assign(new Error("superseded"), { name: "AbortError" }); }
    const at = keepFrame ? this.playhead : 0;
    this.close();
    const frames = new FrameServer(demux);
    const thumbs = new ThumbServer(demux);
    frames.onFrame = () => this.invalidate();
    frames.onError = (msg) => console.warn(msg);
    thumbs.onThumb = () => this.invalidate();
    const first = demux.index.first;
    this.media = { demux, frames, thumbs, info: demux.info, fps: demux.fps, first, count: demux.index.count - first };
    this.canvas.width = demux.info.width;
    this.canvas.height = demux.info.height;
    this.drawn = null;
    if (!keepFrame) this.range = null;
    else if (this.range) this.range = { a: Math.min(this.range.a, this.total - 1), b: Math.min(this.range.b, this.total - 1) };
    this.playhead = Math.min(at, this.total - 1);
    this.updateWants();
    this.invalidate();
    // a video with sound: open the audio output while idle, not on the first press of play
    if (demux.audio) (window.requestIdleCallback || setTimeout)(() => this.audio.warm(), { timeout: 2000 });
    return { width: demux.info.width, height: demux.info.height, duration: this.duration, fps: demux.fps, count: this.total };
  }

  close() {
    this.stop();
    const m = this.media;
    if (!m) return;
    this.media = null;
    m.frames.dispose();
    m.thumbs.dispose();
    m.demux.dispose();
    this.drawn = null;
    this.playhead = 0;
    this.range = null;
    this.invalidate();
  }

  // the in point at frame f (an out point before it goes); null clears it
  setIn(f) {
    if (!this.total) return;
    if (f == null) { this.setRange(0, this.range?.b ?? this.total - 1); return; }
    f = Math.max(0, Math.min(this.total - 1, Math.round(f)));
    this.setRange(f, this.range && this.range.b >= f ? this.range.b : this.total - 1);
  }

  // the out point at frame f (an in point after it goes); null clears it
  setOut(f) {
    if (!this.total) return;
    if (f == null) { this.setRange(this.range?.a ?? 0, this.total - 1); return; }
    f = Math.max(0, Math.min(this.total - 1, Math.round(f)));
    this.setRange(this.range && this.range.a <= f ? this.range.a : 0, f);
  }

  // in and out points at frames a and b; the whole clip clears them
  setRange(a, b) {
    this.beforeRange?.();
    const n = this.total;
    a = Math.max(0, Math.min(n - 1, a));
    b = Math.max(a, Math.min(n - 1, b));
    this.range = a === 0 && b === n - 1 ? null : { a, b };
    this.updateWants();
    this.invalidate();
  }

  seek(f) {
    if (!this.total) return;
    f = Math.max(0, Math.min(this.total - 1, Math.round(f)));
    if (f === this.playhead) return;
    this.playhead = f;
    if (this.play) { this.play.nextDue = performance.now() + 1000 / this.fps; this.startAudio(); }
    this.updateWants();
    this.invalidate();
  }

  // Moves the playhead to frame f to mark where work elsewhere (an export) is, without fetching or
  // showing the frame; the timeline keeps it in view. Until untrack().
  track(f) {
    if (!this.total) return;
    this.tracking = true;
    f = Math.max(0, Math.min(this.total - 1, Math.round(f)));
    if (f === this.playhead) return;
    this.playhead = f;
    this.invalidate();
  }

  untrack() {
    this.tracking = false;
    this.updateWants();
    this.invalidate();
  }

  step(n) {
    this.stop();
    this.seek(this.playhead + n);
  }

  toggle() {
    if (this.play) { this.stop(); return; }
    if (!this.media) return;
    if (!this.inRange(this.playhead) || this.playhead >= this.outPoint) this.seek(this.inPoint);
    this.play = { nextDue: performance.now(), audioSynced: 0 };
    this.startAudio();
    this.media.thumbs.setPaused(true);
    this.updateWants();
    this.invalidate();
  }

  stop() {
    if (!this.play) return;
    this.play = null;
    this.audio.stop();
    this.media?.thumbs.setPaused(false);
    this.updateWants();
    this.invalidate();
  }

  // the source frame index of frame f
  src(f) { return this.media.first + Math.max(0, Math.min(this.total - 1, f)); }

  // the frame played after frame f: the in point again after the out point
  after(f) { return f === this.outPoint ? this.inPoint : f + 1; }

  // frames to decode now, most urgent first: what plays next while playing (looping back to the
  // in point), else the neighbours
  updateWants() {
    const m = this.media;
    if (!m) return;
    const list = [this.src(this.playhead)];
    if (this.play) {
      const room = Math.max(4, m.frames.capacity - 14);
      const ahead = Math.min(Math.max(4, Math.floor(room / 2)), Math.ceil(m.fps * 1.5) + 8);
      for (let i = 1, f = this.playhead; i < ahead; i++) {
        f = this.after(f);
        if (f >= m.count || f === this.playhead) break;
        list.push(this.src(f));
      }
    } else {
      for (let i = 1; i <= 4; i++) if (this.playhead + i < m.count) list.push(this.src(this.playhead + i));
      for (let i = 1; i <= 6; i++) if (this.playhead - i >= 0) list.push(this.src(this.playhead - i));
    }
    m.frames.want(list);
  }

  frame(now) {
    this.requested = false;
    if (this.play) this.tick(now);
    this.drawFrame();
    this.drawers.forEach((f) => f());
    if (this.play) this.invalidate();
  }

  // advance by whole frames, never skipping one: if the next isn't decoded yet, wait for it
  tick(now) {
    const m = this.media, play = this.play;
    const dt = 1000 / m.fps;
    if (now - play.nextDue > 250) play.nextDue = now;
    let advanced = false, resumed = false;
    while (now >= play.nextDue) {
      // past the out point (or the end) it loops back to the in point
      const next = this.playhead >= this.outPoint ? this.inPoint : this.playhead + 1;
      const looped = next <= this.playhead;
      if (!m.frames.has(this.src(next)) || (this.canShow && !this.canShow(next))) {
        // waiting for a frame: the sound waits too
        if (!play.stalled && now - play.nextDue > 100) { play.stalled = true; this.audio.stop(); }
        break;
      }
      if (play.stalled) { play.stalled = false; play.nextDue = now; resumed = true; }
      this.playhead = next;
      play.nextDue += dt;
      advanced = true;
      if (looped) resumed = true; // the sound starts over with it
    }
    if (advanced) this.updateWants();
    if (resumed && this.play) this.startAudio();
    else if (this.play && !play.stalled) this.syncAudio(now);
  }

  drawFrame() {
    const m = this.media;
    if (!m) return;
    // only the exact frame: until it's decoded the last one stays up (the frames decoded on the
    // way to it from the keyframe would otherwise flash by)
    const bmp = m.frames.get(this.src(this.playhead));
    const over = this.overlay?.token(this.playhead) ?? null;
    if (!bmp || (bmp === this.drawn && over === this.drawnOver)) return;
    const ctx = this.ctx;
    ctx.setTransform(rawToDisplay(m.info));
    ctx.drawImage(bmp, 0, 0);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.overlay?.draw(ctx, this.playhead);
    this.drawn = bmp;
    this.drawnOver = over;
  }

  // media time (the clock audio timestamps use) at the start of frame f
  mediaTime(f) { return this.media.demux.index.pts[this.src(f)]; }

  startAudio(frac = 0) {
    const m = this.media;
    if (!m?.demux.audio || !this.play) return;
    this.play.audioSynced = performance.now();
    this.audio.start(m.demux.audio, this.mediaTime(this.playhead) + frac);
  }

  syncAudio(now) {
    const pos = this.audio.position();
    if (pos == null) return;
    const dt = 1000 / this.fps;
    // how far into the current frame's display time we are
    const frac = Math.max(0, Math.min(dt, now - (this.play.nextDue - dt))) / 1000;
    const drift = pos - (this.mediaTime(this.playhead) + frac);
    if (Math.abs(drift) > AUDIO_DRIFT && now - this.play.audioSynced > 300) this.startAudio(frac);
  }
}
