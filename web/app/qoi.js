// RGBA rows as QOI's stream of operations (qoiformat.org), without its header: lossless, in one
// pass, and fast enough in plain JS to run for every frame kept. A swapped face is smooth, so most
// pixels fit in a byte or two, as a small step from the last one.

const hash = (r, g, b, a) => ((r * 3 + g * 5 + b * 7 + a * 11) % 64) * 4;

// -> the stream of the n pixels of px (a view into a buffer of its own: n * 5 bytes at most)
export function encode(px, n) {
  const out = new Uint8Array(n * 5);
  const seen = new Uint8Array(256); // 64 pixels seen before, by their hash
  let o = 0, run = 0, r = 0, g = 0, b = 0, a = 255;
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const R = px[p], G = px[p + 1], B = px[p + 2], A = px[p + 3];
    if (R === r && G === g && B === b && A === a) {
      if (++run === 62 || i === n - 1) { out[o++] = 0xc0 | (run - 1); run = 0; }
      continue;
    }
    if (run) { out[o++] = 0xc0 | (run - 1); run = 0; }
    const h = hash(R, G, B, A);
    if (seen[h] === R && seen[h + 1] === G && seen[h + 2] === B && seen[h + 3] === A) out[o++] = h >> 2;
    else {
      seen[h] = R; seen[h + 1] = G; seen[h + 2] = B; seen[h + 3] = A;
      if (A === a) {
        const dr = ((R - r + 128) & 255) - 128, dg = ((G - g + 128) & 255) - 128, db = ((B - b + 128) & 255) - 128;
        const drg = dr - dg, dbg = db - dg;
        if (dr >= -2 && dr <= 1 && dg >= -2 && dg <= 1 && db >= -2 && db <= 1) out[o++] = 0x40 | ((dr + 2) << 4) | ((dg + 2) << 2) | (db + 2);
        else if (dg >= -32 && dg <= 31 && drg >= -8 && drg <= 7 && dbg >= -8 && dbg <= 7) { out[o++] = 0x80 | (dg + 32); out[o++] = ((drg + 8) << 4) | (dbg + 8); }
        else { out[o++] = 0xfe; out[o++] = R; out[o++] = G; out[o++] = B; }
      } else { out[o++] = 0xff; out[o++] = R; out[o++] = G; out[o++] = B; out[o++] = A; }
    }
    r = R; g = G; b = B; a = A;
  }
  return out.subarray(0, o);
}

// Writes the n pixels of a stream into px. -> whether it was the stream of exactly n pixels
//
// It never reads past the end of src: one read there (a file cut short) leaves the JS engine
// running this function many times slower for as long as the worker lives.
export function decode(src, n, px) {
  const seen = new Uint8Array(256), end = src.length;
  let s = 0, run = 0, r = 0, g = 0, b = 0, a = 255;
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    if (run > 0) run--;
    else {
      if (s >= end) return false;
      const op = src[s++];
      if (op === 0xfe) {
        if (s + 3 > end) return false;
        r = src[s++]; g = src[s++]; b = src[s++];
      } else if (op === 0xff) {
        if (s + 4 > end) return false;
        r = src[s++]; g = src[s++]; b = src[s++]; a = src[s++];
      } else if (op < 0x40) {
        const h = op * 4;
        r = seen[h]; g = seen[h + 1]; b = seen[h + 2]; a = seen[h + 3];
      } else if (op < 0x80) {
        r = (r + ((op >> 4) & 3) - 2) & 255; g = (g + ((op >> 2) & 3) - 2) & 255; b = (b + (op & 3) - 2) & 255;
      } else if (op < 0xc0) {
        if (s >= end) return false;
        const next = src[s++], dg = (op & 0x3f) - 32;
        r = (r + dg - 8 + (next >> 4)) & 255; g = (g + dg) & 255; b = (b + dg - 8 + (next & 15)) & 255;
      } else run = op & 0x3f;
      const h = hash(r, g, b, a);
      seen[h] = r; seen[h + 1] = g; seen[h + 2] = b; seen[h + 3] = a;
    }
    px[p] = r; px[p + 1] = g; px[p + 2] = b; px[p + 3] = a;
  }
  return s === end && run === 0;
}
