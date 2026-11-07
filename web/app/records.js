// How the saved work is written as text. JSON keeps numbers and strings, so what it doesn't keep is
// written as { $t: its type, v }.

const TYPED = { Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Uint8Array, Uint8ClampedArray, Uint16Array, Uint32Array };

function toBase64(bytes) {
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

function fromBase64(text) {
  const s = atob(text);
  const bytes = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i);
  return bytes;
}

export const encode = (value) => JSON.stringify(value, (_, v) => {
  if (typeof v === "number" && !Number.isFinite(v)) return { $t: "number", v: String(v) };
  if (ArrayBuffer.isView(v)) return { $t: v.constructor.name, v: toBase64(new Uint8Array(v.buffer, v.byteOffset, v.byteLength)) };
  if (v instanceof Map) return { $t: "Map", v: [...v] };
  if (v instanceof Set) return { $t: "Set", v: [...v] };
  return v;
});

export const decode = (text) => JSON.parse(text, (_, v) => {
  if (!v?.$t) return v;
  if (v.$t === "number") return Number(v.v);
  if (v.$t === "Map") return new Map(v.v);
  if (v.$t === "Set") return new Set(v.v);
  const Type = TYPED[v.$t];
  return Type ? new Type(fromBase64(v.v).buffer) : v;
});
