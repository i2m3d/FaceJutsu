// The GPU's frame work, one job at a time, the most wanted first. Each job asks for the GPU one
// frame at a time (run), so the work around the playhead never waits behind more than a frame of
// the rest. A job also claims its priority while it goes on (claim), between its frames too
// (while the next one decodes): nothing less wanted starts meanwhile.

export const PRIO = {
  NOW: 0, // loading the models, opening a photo or video, the export
  NEAR: 1, // finding the faces around the playhead, and swapping the frames there (taking turns)
  WINDOW: 2, // finding every face of the window around the playhead
  SKIM: 3, // a look at the keyframes of the whole video (between the in and out points)
  BACKGROUND: 4, // swapping the rest, so the export has less to do
};

// the frames around the playhead, seconds behind and ahead of it: the faces found first there
export const NEAR_SECONDS = { behind: 1, ahead: 4 };

export class GpuQueue {
  constructor() {
    this.waiting = []; // [{ fn, prio, resolve, reject }], in the order asked
    this.claims = []; // the priorities of the jobs going on
    this.busy = false;
  }

  // Runs fn with the GPU to itself, once nothing more wanted waits or goes on.
  run(fn, prio = PRIO.NOW) {
    return new Promise((resolve, reject) => {
      this.waiting.push({ fn, prio, resolve, reject });
      this.next();
    });
  }

  // Claims `prio` for a job until the function returned is called.
  claim(prio) {
    this.claims.push(prio);
    let done = false;
    return () => {
      if (done) return;
      done = true;
      this.claims.splice(this.claims.indexOf(prio), 1);
      this.next();
    };
  }

  next() {
    if (this.busy || !this.waiting.length) return;
    let k = 0; // the most wanted, the first asked of those
    for (let i = 1; i < this.waiting.length; i++) if (this.waiting[i].prio < this.waiting[k].prio) k = i;
    const w = this.waiting[k];
    if (this.claims.some((p) => p < w.prio)) return; // (called again when that claim ends)
    this.waiting.splice(k, 1);
    this.busy = true;
    Promise.resolve().then(w.fn).then(w.resolve, w.reject).finally(() => {
      this.busy = false;
      this.next();
    });
  }
}
