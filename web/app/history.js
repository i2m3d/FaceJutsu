// Undo and redo for the edits of the open photo or video: a stack of snapshots of what they
// change (the page makes them, main.js snapshot), each with a label naming the edit ("split").
// Undoing a step hands back the snapshot from before it and keeps the one it replaces for redo.

const LIMIT = 200; // steps kept; the oldest go first

export class History {
  constructor() {
    this.done = []; // [{ label, snap }]: snap is the state before that step
    this.undone = []; // [{ label, snap }]: snap is the state after that step
  }

  get canUndo() { return this.done.length > 0; }
  get canRedo() { return this.undone.length > 0; }
  get undoLabel() { return this.done.at(-1)?.label ?? ""; }
  get redoLabel() { return this.undone.at(-1)?.label ?? ""; }

  // A step was made: `before` is the state it changed. A new step drops what was undone. -> the step
  push(label, before) {
    const step = { label, snap: before };
    this.done.push(step);
    if (this.done.length > LIMIT) this.done.shift();
    this.undone = [];
    return step;
  }

  // Takes a step out again, as if it hadn't been made (an edit that failed).
  drop(step) {
    const i = this.done.indexOf(step);
    if (i >= 0) this.done.splice(i, 1);
  }

  // -> the state to go back to, or null; `now` is the state it replaces (for redo)
  undo(now) {
    const step = this.done.pop();
    if (!step) return null;
    this.undone.push({ label: step.label, snap: now });
    return step.snap;
  }

  // -> the state to go forward to, or null; `now` is the state it replaces (for undo)
  redo(now) {
    const step = this.undone.pop();
    if (!step) return null;
    this.done.push({ label: step.label, snap: now });
    return step.snap;
  }

  clear() {
    this.done = [];
    this.undone = [];
  }
}
