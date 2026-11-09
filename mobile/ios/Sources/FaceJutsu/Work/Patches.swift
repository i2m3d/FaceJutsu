import CoreVideo
import CryptoKit
import FaceJutsuEngine
import Foundation

// The swapped faces already made, as the desktop keeps them (web/app/patches.js): a patch is the
// part of a frame a swap changed, its finished pixels kept losslessly, under a key naming
// everything that went into it (the frame, its faces, their characters and settings, the engine).
// The picture and the export both make frames from patches, so what was seen is what gets
// exported, and each is made once.
//
// On disk: Application Support/FaceJutsu/Cache/<work>/<key>.patch, 16 bytes of box (x1, y1, x2,
// y2, little-endian), 4 of the length of the frame's samples, those samples, then the box's BGRA
// rows, each byte less the one 4 before it in its row, LZFSE-compressed. The oldest go when the
// folder grows past its budget.

struct Patch: Sendable {
    /// the box [x1, x2) x [y1, y2) of the stored frame
    let x1, y1, x2, y2: Int
    /// its BGRA rows, (x2 - x1) * 4 bytes each: alpha 0 where the swap left the frame as it was,
    /// so only what it changed goes over a frame
    let pixels: Data
    /// BGR of the frame it was made from at `samplePoints`, outside the box, which a swap never
    /// touches: to tell that it's going over the same frame
    let samples: Data
}

enum Patches {
    /// what's kept at most, all works together
    static let budget: Int64 = 2_000_000_000
    /// the most the samples may differ on average, in levels of 255: two decodings of a frame can
    /// differ a little
    static let sameFrame = 3.0
    /// changes when the engine makes different pixels for the same settings, or patches change
    private static let made = "5 \(engineVersion())"
    private static let writes = DispatchQueue(label: "patches", qos: .utility)
    nonisolated(unsafe) private static var written = 0

    static var folder: URL {
        let url = Store.root.appendingPathComponent("Cache", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private static func url(_ work: String, _ key: String) -> URL {
        folder.appendingPathComponent(work, isDirectory: true).appendingPathComponent(key + ".patch")
    }

    /// The key of frame `frame` (stored `width` x `height`, shown turned `rotation`) swapped as
    /// `plan` says.
    static func key(frame: Int, width: Int, height: Int, rotation: Int, plan: [PlannedFace]) -> String {
        var text = "\(made)|\(frame)|\(width)x\(height)|\(rotation)"
        for p in plan {
            let kps = p.face.kps.map { String(format: "%.3f", $0) }.joined(separator: ",")
            text += "|\(p.identity) \(kps) \(p.detail) \(p.enhance) \(p.subtle) \(p.strength) \(p.edges) \(p.objects) \(p.coverage)"
            if let d = p.ownDetail { text += String(format: " %.4f", d) }
        }
        return SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    static func get(work: String, key: String) -> Patch? {
        guard let file = try? Data(contentsOf: url(work, key)), file.count > 20 else { return nil }
        let box = file.prefix(20).withUnsafeBytes { raw in (0..<5).map { Int(UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) } }
        let (x1, y1, x2, y2) = (box[0], box[1], box[2], box[3])
        guard x2 > x1, y2 > y1, file.count > 20 + box[4] else { return nil }
        let samples = Data(file.dropFirst(20).prefix(box[4]))
        guard let raw = try? (Data(file.dropFirst(20 + box[4])) as NSData).decompressed(using: .lzfse) as Data else { return nil }
        let row = (x2 - x1) * 4
        guard raw.count == row * (y2 - y1) else { return nil }
        var pixels = raw
        pixels.withUnsafeMutableBytes { (b: UnsafeMutableRawBufferPointer) in
            let p = b.bindMemory(to: UInt8.self)
            for y in 0..<(y2 - y1) {
                let r = y * row
                for i in 4..<row { p[r + i] &+= p[r + i - 4] }
            }
        }
        return Patch(x1: x1, y1: y1, x2: x2, y2: y2, pixels: pixels, samples: samples)
    }

    /// Keeps a patch, soon after (on a queue of their own).
    static func put(work: String, key: String, _ patch: Patch) {
        writes.async {
            let row = (patch.x2 - patch.x1) * 4
            var delta = patch.pixels
            delta.withUnsafeMutableBytes { (b: UnsafeMutableRawBufferPointer) in
                let p = b.bindMemory(to: UInt8.self)
                for y in 0..<(patch.y2 - patch.y1) {
                    let r = y * row
                    for i in stride(from: row - 1, through: 4, by: -1) { p[r + i] &-= p[r + i - 4] }
                }
            }
            guard let packed = try? (delta as NSData).compressed(using: .lzfse) as Data else { return }
            var file = Data()
            for v in [patch.x1, patch.y1, patch.x2, patch.y2, patch.samples.count] { withUnsafeBytes(of: UInt32(v).littleEndian) { file.append(contentsOf: $0) } }
            file.append(patch.samples)
            file.append(packed)
            let dest = url(work, key)
            try? FileManager.default.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
            try? file.write(to: dest, options: .atomic)
            written += 1
            if written % 100 == 1 { trim() }
        }
    }

    /// The keys of a work's patches, once those on their way are written.
    static func keys(work: String) -> Set<String> {
        writes.sync {}
        let names = (try? FileManager.default.contentsOfDirectory(atPath: folder.appendingPathComponent(work, isDirectory: true).path)) ?? []
        return Set(names.filter { $0.hasSuffix(".patch") }.map { String($0.dropLast(6)) })
    }

    /// The part of `output` that differs from `input` (the same size, BGRA), as a patch; nil when
    /// nothing changed.
    static func diff(_ input: CVPixelBuffer, _ output: CVPixelBuffer) -> Patch? {
        withPixels(input) { a, w, h, s -> Patch? in
            withPixels(output) { o, _, _, os -> Patch? in
                let (w, h) = (Int(w), Int(h))
                guard let ap = UnsafeRawPointer(bitPattern: UInt(a)), let op = UnsafeRawPointer(bitPattern: UInt(o)) else { return nil }
                var (x1, y1, x2, y2) = (w, h, 0, 0)
                for y in 0..<h {
                    let inRow = (ap + y * Int(s)).assumingMemoryBound(to: UInt32.self)
                    let outRow = (op + y * Int(os)).assumingMemoryBound(to: UInt32.self)
                    if memcmp(inRow, outRow, w * 4) == 0 { continue }
                    var first = 0
                    while inRow[first] == outRow[first] { first += 1 }
                    var last = w - 1
                    while inRow[last] == outRow[last] { last -= 1 }
                    x1 = min(x1, first)
                    x2 = max(x2, last + 1)
                    y1 = min(y1, y)
                    y2 = y + 1
                }
                guard x2 > x1, y2 > y1 else { return nil }
                let row = (x2 - x1) * 4
                var pixels = Data(count: row * (y2 - y1))
                pixels.withUnsafeMutableBytes { (b: UnsafeMutableRawBufferPointer) in
                    for y in y1..<y2 {
                        let inRow = (ap + y * Int(s)).assumingMemoryBound(to: UInt32.self)
                        let outRow = (op + y * Int(os)).assumingMemoryBound(to: UInt32.self)
                        let to = (b.baseAddress! + (y - y1) * row).assumingMemoryBound(to: UInt32.self)
                        for x in x1..<x2 { to[x - x1] = inRow[x] == outRow[x] ? 0 : outRow[x] | 0xff00_0000 }
                    }
                }
                var samples = Data()
                for (x, y) in samplePoints(w, h, x1: x1, y1: y1, x2: x2, y2: y2) {
                    samples.append(contentsOf: UnsafeRawBufferPointer(start: ap + y * Int(s) + x * 4, count: 3))
                }
                return Patch(x1: x1, y1: y1, x2: x2, y2: y2, pixels: pixels, samples: samples)
            }
        }
    }

    /// Points spread over a w x h frame, outside the box a patch covers.
    private static func samplePoints(_ w: Int, _ h: Int, x1: Int, y1: Int, x2: Int, y2: Int) -> [(Int, Int)] {
        var points: [(Int, Int)] = []
        for j in 0..<16 {
            for i in 0..<16 {
                let (x, y) = (w * (2 * i + 1) / 32, h * (2 * j + 1) / 32)
                if x < x1 || x >= x2 || y < y1 || y >= y2 { points.append((x, y)) }
            }
        }
        return points
    }

    /// How far `frame` is, on average in levels of 255, from the frame the patch was made from,
    /// where the swap didn't touch it; nil when it isn't laid out as that frame was.
    static func difference(_ patch: Patch, frame: CVPixelBuffer) -> Double? {
        withPixels(frame) { a, w, h, s -> Double? in
            guard let ap = UnsafeRawPointer(bitPattern: UInt(a)), patch.x2 <= Int(w), patch.y2 <= Int(h) else { return nil }
            let points = samplePoints(Int(w), Int(h), x1: patch.x1, y1: patch.y1, x2: patch.x2, y2: patch.y2)
            guard patch.samples.count == points.count * 3 else { return nil }
            var sum = 0
            patch.samples.withUnsafeBytes { (b: UnsafeRawBufferPointer) in
                for (n, p) in points.enumerated() {
                    let q = ap + p.1 * Int(s) + p.0 * 4
                    for c in 0..<3 { sum += abs(Int(b[n * 3 + c]) - Int(q.load(fromByteOffset: c, as: UInt8.self))) }
                }
            }
            return Double(sum) / Double(max(points.count * 3, 1))
        }
    }

    /// `frame` with what the patch changed over it, into `out` (the same size, BGRA). False, with
    /// `out` left alone, when `frame` isn't the frame the patch was made from.
    static func apply(_ patch: Patch, frame: CVPixelBuffer, into out: CVPixelBuffer) -> Bool {
        guard let d = difference(patch, frame: frame), d <= sameFrame else { return false }
        return withPixels(frame) { a, w, h, s -> Bool in
            withPixels(out, readOnly: false) { o, _, _, os -> Bool in
                guard let ap = UnsafeRawPointer(bitPattern: UInt(a)), let op = UnsafeMutableRawPointer(bitPattern: UInt(o)) else { return false }
                for y in 0..<Int(h) { memcpy(op + y * Int(os), ap + y * Int(s), Int(w) * 4) }
                let row = (patch.x2 - patch.x1) * 4
                patch.pixels.withUnsafeBytes { (b: UnsafeRawBufferPointer) in
                    for y in patch.y1..<patch.y2 {
                        let from = b.baseAddress! + (y - patch.y1) * row
                        let to = (op + y * Int(os) + patch.x1 * 4).assumingMemoryBound(to: UInt32.self)
                        for x in 0..<(patch.x2 - patch.x1) {
                            let v = from.loadUnaligned(fromByteOffset: x * 4, as: UInt32.self)
                            if v >> 24 != 0 { to[x] = v }
                        }
                    }
                }
                return true
            }
        }
    }

    /// What's kept, in bytes.
    static var bytes: Int64 {
        files().reduce(0) { $0 + $1.bytes }
    }

    /// Forgets a work's patches, or all of them.
    static func remove(work: String? = nil) {
        writes.async {
            try? FileManager.default.removeItem(at: work.map { folder.appendingPathComponent($0, isDirectory: true) } ?? folder)
        }
    }

    private static func files() -> [(url: URL, bytes: Int64, date: Date)] {
        let keys: [URLResourceKey] = [.fileSizeKey, .contentModificationDateKey]
        guard let e = FileManager.default.enumerator(at: folder, includingPropertiesForKeys: keys) else { return [] }
        var out: [(URL, Int64, Date)] = []
        for case let f as URL in e where f.pathExtension == "patch" {
            let v = try? f.resourceValues(forKeys: Set(keys))
            out.append((f, Int64(v?.fileSize ?? 0), v?.contentModificationDate ?? .distantPast))
        }
        return out
    }

    /// The oldest go until what's kept is well within the budget.
    private static func trim() {
        var all = files()
        var total = all.reduce(0) { $0 + $1.bytes }
        guard total > budget else { return }
        all.sort { $0.date < $1.date }
        for f in all where total > budget * 9 / 10 {
            try? FileManager.default.removeItem(at: f.url)
            total -= f.bytes
        }
    }
}
