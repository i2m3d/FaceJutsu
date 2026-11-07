// A zip of files kept as they are (the "stored" method): photos are compressed already, so
// deflating them again would only cost time. Names are UTF-8; no zip64, so under 4 GB in all.

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

// MS-DOS time and date, as zip headers keep them
function dosTime(d) {
  return {
    time: (d.getHours() << 11) | (d.getMinutes() << 5) | (d.getSeconds() >> 1),
    date: ((d.getFullYear() - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate(),
  };
}

// files: [{ name, blob }] -> a Blob of type application/zip. A name used before gets " (2)",
// " (3)", ... before its extension.
export async function zipFiles(files, when = new Date()) {
  const enc = new TextEncoder(), { time, date } = dosTime(when);
  const parts = [], central = [], used = new Set();
  let offset = 0;
  for (const { name, blob } of files) {
    let unique = name;
    for (let i = 2; used.has(unique.toLowerCase()); i++) unique = name.replace(/(\.[^.]*)?$/, (ext) => ` (${i})${ext}`);
    used.add(unique.toLowerCase());
    const nm = enc.encode(unique), data = new Uint8Array(await blob.arrayBuffer());
    const crc = crc32(data), size = data.length;
    if (offset + size > 0xffffffff) throw new Error("These photos are too big together for one zip (over 4 GB).");
    // the fields local and central headers share: version 2.0, UTF-8 names, stored, time, sizes
    const common = (v) => {
      v.setUint16(0, 20, true);
      v.setUint16(2, 0x0800, true);
      v.setUint16(4, 0, true);
      v.setUint16(6, time, true);
      v.setUint16(8, date, true);
      v.setUint32(10, crc, true);
      v.setUint32(14, size, true);
      v.setUint32(18, size, true);
      v.setUint16(22, nm.length, true);
    };
    const local = new DataView(new ArrayBuffer(30));
    local.setUint32(0, 0x04034b50, true);
    common(new DataView(local.buffer, 4));
    parts.push(local, nm, data);
    const cd = new DataView(new ArrayBuffer(46));
    cd.setUint32(0, 0x02014b50, true);
    cd.setUint16(4, 20, true); // made by: version 2.0
    common(new DataView(cd.buffer, 6));
    cd.setUint32(42, offset, true);
    central.push(cd, nm);
    offset += 30 + nm.length + size;
  }
  const cdSize = central.reduce((a, p) => a + p.byteLength, 0);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true);
  end.setUint16(8, files.length, true);
  end.setUint16(10, files.length, true);
  end.setUint32(12, cdSize, true);
  end.setUint32(16, offset, true);
  return new Blob([...parts, ...central, end], { type: "application/zip" });
}
