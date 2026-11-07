// Frame-exact access to a video file through Mediabunny: an index of every frame's timestamp
// (read from the container's sample table, without decoding), the packets a decoder needs to
// reach any frame, and the audio track decoded on demand.

import { Input, BlobSource, ALL_FORMATS, EncodedPacketSink, AudioBufferSink } from "../vendor/mediabunny.min.mjs";

export const toUs = (sec) => Math.round(sec * 1e6);

// packets: [{ ts, key }] in any order -> { count, pts, us, key, gop, first, end, byUs }
// pts: seconds, ascending; us: the same as integer microseconds (what decoders echo back);
// gop: ordinal of the keyframe that starts each frame's GOP; first: first frame meant to be shown
// (frames with negative pts are edit-list lead-in); end: end of the last frame, seconds
export function buildIndex(packets, lastDuration) {
  packets.sort((a, b) => a.ts - b.ts);
  // some containers repeat timestamps; keep the first so indices stay unique
  const uniq = [];
  for (const p of packets) {
    const last = uniq[uniq.length - 1];
    if (last && toUs(last.ts) === toUs(p.ts)) { last.key ||= p.key; continue; }
    uniq.push(p);
  }
  const count = uniq.length;
  const pts = new Float64Array(count), us = new Float64Array(count);
  const key = new Uint8Array(count), gop = new Int32Array(count);
  const byUs = new Map();
  let ord = -1, first = 0;
  for (let i = 0; i < count; i++) {
    pts[i] = uniq[i].ts;
    us[i] = toUs(uniq[i].ts);
    key[i] = uniq[i].key ? 1 : 0;
    if (key[i]) ord++;
    gop[i] = Math.max(0, ord);
    byUs.set(us[i], i);
  }
  while (first < count - 1 && pts[first] < -1e-9) first++;
  let frameDur = lastDuration;
  if (!(frameDur > 0) && count > 1) frameDur = (pts[count - 1] - pts[first]) / Math.max(1, count - 1 - first);
  const end = count ? pts[count - 1] + (frameDur > 0 ? frameDur : 1 / 30) : 0;
  return { count, pts, us, key, gop, first, end, byUs };
}

const COMMON = [
  [24000, 1001], [24, 1], [25, 1], [30000, 1001], [30, 1], [48, 1], [50, 1],
  [60000, 1001], [60, 1], [90, 1], [100, 1], [120000, 1001], [120, 1], [144, 1], [240, 1],
  [15, 1], [12, 1], [10, 1], [8, 1], [6, 1], [5, 1],
];

// the nominal frame rate: the mean frame spacing (per-frame gaps suffer from timestamp rounding),
// snapped to a common rate when close
export function detectFps(index) {
  const n = index.count - index.first;
  if (n < 2) return 30;
  const raw = (n - 1) / (index.pts[index.count - 1] - index.pts[index.first]);
  let best = 0, err = Infinity;
  for (const [num, den] of COMMON) {
    const e = Math.abs(raw - num / den) / (num / den);
    if (e < err) { err = e; best = num / den; }
  }
  return err < 0.0005 ? best : Math.round(raw * 1000) / 1000;
}

// file: a File or Blob -> { info, index, fps, config, audio, packets(idx), keyChunk(idx), dispose() }
// info: { rawW, rawH (decoded size), rotation (clockwise, for display), flip, width, height (displayed) }
export async function openVideo(file) {
  const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS });
  try {
    const track = await input.getPrimaryVideoTrack();
    if (!track) throw new Error("this file has no video track");
    const config = await track.getDecoderConfig();
    if (!config) throw new Error("this browser can't decode this video's format");
    // decoders hand back raw frames; the rotation is applied when drawing
    const plain = { ...config, optimizeForLatency: true };
    delete plain.rotation;
    delete plain.flip;
    if (!(await VideoDecoder.isConfigSupported(plain).then((s) => s.supported, () => false))) {
      throw new Error(`this browser can't decode ${(await track.getCodec()) || config.codec} video`);
    }
    const sink = new EncodedPacketSink(track);
    const raw = [];
    let lastDuration = 0;
    for await (const p of sink.packets(undefined, undefined, { metadataOnly: true })) {
      raw.push({ ts: p.timestamp, key: p.type === "key" });
      lastDuration = p.duration;
    }
    if (!raw.length) throw new Error("the video track has no frames");
    const index = buildIndex(raw, lastDuration);
    const rotation = ((track.rotation % 360) + 360) % 360;
    const flip = await track.getFlip?.().catch(() => false) ?? false;
    const rawW = track.codedWidth, rawH = track.codedHeight;
    const info = { rawW, rawH, rotation, flip, width: rotation % 180 ? rawH : rawW, height: rotation % 180 ? rawW : rawH };

    let audio = null;
    const audioTrack = await input.getPrimaryAudioTrack().catch(() => null);
    if (audioTrack && (await audioTrack.canDecode().catch(() => false))) {
      const audioSink = new AudioBufferSink(audioTrack);
      audio = {
        // decoded audio from about `from` seconds (the buffer containing it) up to `to`: { t, buf }
        async *buffers(from, to) {
          for await (const b of audioSink.buffers(from, to)) yield { t: b.timestamp, buf: b.buffer };
        },
      };
    }

    const chunk = (p) => new EncodedVideoChunk({ type: p.type, timestamp: toUs(p.timestamp), duration: Math.max(0, toUs(p.duration)), data: p.data });
    const keyPacket = async (idx) => (await sink.getKeyPacket(index.pts[idx], { verifyKeyPackets: true })) ?? sink.getFirstKeyPacket({ verifyKeyPackets: true });
    return {
      info, index, fps: detectFps(index), config: plain, audio,
      // chunks in decode order, from the keyframe frame idx needs
      async *packets(idx) {
        const start = await keyPacket(idx);
        if (!start) return;
        for await (const p of sink.packets(start)) yield chunk(p);
      },
      // the keyframe chunk at or before frame idx (for thumbnails)
      async keyChunk(idx) {
        const p = await keyPacket(idx);
        return p ? chunk(p) : null;
      },
      dispose: () => input.dispose(),
    };
  } catch (e) {
    input.dispose();
    throw e;
  }
}
