import AVFoundation
import CoreImage
import CoreVideo
import FaceJutsuEngine
import Foundation
import UIKit

// Pictures and videos for the engine: frames as BGRA pixel buffers in their stored orientation (the
// engine finds faces the right way up from the rotation), and pictures of them for the screen.

/// An error's words, for a message.
func describe(_ e: Error) -> String {
    if let e = e as? EngineError, case .Failed(let why) = e { return why }
    return (e as? LocalizedError)?.errorDescription ?? e.localizedDescription
}

struct VideoError: LocalizedError {
    let errorDescription: String?
    init(_ why: String) { errorDescription = why }
}

/// The clockwise quarter turns, in degrees, that a display matrix shows a frame with (0, 90, 180
/// or 270); mirroring is ignored.
func rotationDegrees(_ t: CGAffineTransform) -> Int {
    let quarters = Int((atan2(t.b, t.a) * 2 / .pi).rounded())
    return (quarters * 90 % 360 + 360) % 360
}

/// The engine, loaded once: the GPU and the models.
@MainActor
final class Engine {
    static let shared = Engine()
    private(set) var swapper: FaceSwapper?
    private(set) var failed: String?
    private var loading: Task<FaceSwapper?, Never>?

    static var modelsFolder: URL { ModelFiles.folder }

    /// The swapper, once the models are loaded (nil if they can't be).
    func ready() async -> FaceSwapper? {
        if let swapper { return swapper }
        if loading == nil {
            let dir = Self.modelsFolder.path
            loading = Task.detached(priority: .userInitiated) { () -> FaceSwapper? in
                do { return try FaceSwapper(modelsDir: dir) } catch {
                    let why = describe(error)
                    await MainActor.run { Engine.shared.failed = why }
                    return nil
                }
            }
        }
        swapper = await loading?.value
        // (the models may be here next time)
        if swapper == nil { loading = nil }
        return swapper
    }
}

/// A BGRA pixel buffer the GPU can use in place.
func makePixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
    var pb: CVPixelBuffer?
    let attrs: [String: Any] = [
        kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
        kCVPixelBufferMetalCompatibilityKey as String: true,
        kCVPixelBufferCGImageCompatibilityKey as String: true,
        kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
    ]
    CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
    return pb
}

/// A picture drawn into a BGRA pixel buffer, as it's stored.
func pixelBuffer(from image: CGImage) -> CVPixelBuffer? {
    guard let pb = makePixelBuffer(width: image.width, height: image.height) else { return nil }
    CVPixelBufferLockBaseAddress(pb, [])
    defer { CVPixelBufferUnlockBaseAddress(pb, []) }
    let info = CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
    guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb), width: image.width, height: image.height, bitsPerComponent: 8,
                              bytesPerRow: CVPixelBufferGetBytesPerRow(pb), space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: info)
    else { return nil }
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
    return pb
}

/// A photo upright, as its orientation says.
func uprightImage(_ image: UIImage) -> CGImage? {
    if image.imageOrientation == .up, let cg = image.cgImage { return cg }
    let format = UIGraphicsImageRendererFormat()
    format.scale = 1
    let r = UIGraphicsImageRenderer(size: image.size, format: format)
    return r.image { _ in image.draw(in: CGRect(origin: .zero, size: image.size)) }.cgImage
}

/// Runs `body` with a pixel buffer's pixels: their address as the engine takes it, its size and
/// row stride.
func withPixels<T>(_ pb: CVPixelBuffer, readOnly: Bool = true, _ body: (UInt64, UInt32, UInt32, UInt32) throws -> T) rethrows -> T {
    let flags: CVPixelBufferLockFlags = readOnly ? .readOnly : []
    CVPixelBufferLockBaseAddress(pb, flags)
    defer { CVPixelBufferUnlockBaseAddress(pb, flags) }
    let base = CVPixelBufferGetBaseAddress(pb).map { UInt64(UInt(bitPattern: $0)) } ?? 0
    return try body(base, UInt32(CVPixelBufferGetWidth(pb)), UInt32(CVPixelBufferGetHeight(pb)), UInt32(CVPixelBufferGetBytesPerRow(pb)))
}

enum Pictures {
    static let context = CIContext(options: [.cacheIntermediates: false])

    static func orientation(_ rotation: Int) -> CGImagePropertyOrientation {
        switch rotation {
        case 90: .right
        case 180: .down
        case 270: .left
        default: .up
        }
    }

    /// A frame turned upright, at most `maxSide` px on its longer side.
    static func image(_ pb: CVPixelBuffer, rotation: Int, maxSide: CGFloat = 1600) -> UIImage? {
        var ci = CIImage(cvPixelBuffer: pb).oriented(orientation(rotation))
        let side = max(ci.extent.width, ci.extent.height)
        if side > maxSide {
            let k = maxSide / side
            ci = ci.transformed(by: CGAffineTransform(scaleX: k, y: k))
        }
        guard let cg = context.createCGImage(ci, from: ci.extent.integral) else { return nil }
        return UIImage(cgImage: cg)
    }

    /// A square cut around a face of an upright picture (`box` in its pixels, x1 y1 x2 y2),
    /// a little wider than the box, `side` px.
    static func face(_ image: CGImage, box: [Double], side: CGFloat = 120) -> UIImage? {
        guard box.count == 4 else { return nil }
        let w = box[2] - box[0], h = box[3] - box[1], s = max(w, h) * 1.3
        let rect = CGRect(x: (box[0] + box[2]) / 2 - s / 2, y: (box[1] + box[3]) / 2 - s / 2, width: s, height: s)
            .intersection(CGRect(x: 0, y: 0, width: image.width, height: image.height))
        guard !rect.isNull, let cut = image.cropping(to: rect.integral) else { return nil }
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        return UIGraphicsImageRenderer(size: CGSize(width: side, height: side), format: format).image { _ in
            UIImage(cgImage: cut).draw(in: CGRect(x: 0, y: 0, width: side, height: side))
        }
    }

    /// A character's picture: 3:4 around the face, `height` px.
    static func portrait(_ image: CGImage, box: [Double], height: CGFloat = 400) -> UIImage? {
        guard box.count == 4 else { return nil }
        let w = box[2] - box[0], h = box[3] - box[1]
        let ch = max(h * 1.9, w * 1.9 * 4 / 3), cw = ch * 3 / 4
        var rect = CGRect(x: (box[0] + box[2]) / 2 - cw / 2, y: (box[1] + box[3]) / 2 - ch * 0.45, width: cw, height: ch)
        rect = rect.intersection(CGRect(x: 0, y: 0, width: image.width, height: image.height))
        guard !rect.isNull, let cut = image.cropping(to: rect.integral) else { return nil }
        let size = CGSize(width: height * rect.width / rect.height, height: height)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        return UIGraphicsImageRenderer(size: size, format: format).image { _ in UIImage(cgImage: cut).draw(in: CGRect(origin: .zero, size: size)) }
    }

    /// An upright frame's face cut out, from the frame as stored.
    static func face(_ pb: CVPixelBuffer, rotation: Int, box: [Double], side: CGFloat = 120) -> UIImage? {
        let ci = CIImage(cvPixelBuffer: pb).oriented(orientation(rotation))
        guard let cg = context.createCGImage(ci, from: ci.extent.integral) else { return nil }
        return face(cg, box: box, side: side)
    }
}

// MARK: - Videos

/// A video to swap: its track, how it's shown, and its frames' times.
final class VideoSource: @unchecked Sendable {
    let url: URL
    let asset: AVURLAsset
    let track: AVAssetTrack
    /// as stored
    let size: CGSize
    let transform: CGAffineTransform
    /// the clockwise quarter turns it's shown with
    let rotation: Int
    let duration: Double
    let fps: Double
    let bitrate: Float
    /// the first frame's time: the track's start, until `frameTable` has seen the frames
    private(set) var start: CMTime
    let hasAudio: Bool

    var shown: CGSize { rotation % 180 == 0 ? size : CGSize(width: size.height, height: size.width) }

    private init(url: URL, asset: AVURLAsset, track: AVAssetTrack, size: CGSize, transform: CGAffineTransform, duration: Double, fps: Double, bitrate: Float, start: CMTime, hasAudio: Bool) {
        self.hasAudio = hasAudio
        self.url = url
        self.asset = asset
        self.track = track
        self.size = size
        self.transform = transform
        rotation = rotationDegrees(transform)
        self.duration = duration
        self.fps = fps
        self.bitrate = bitrate
        self.start = start
    }

    static func open(_ url: URL) async throws -> VideoSource {
        let asset = AVURLAsset(url: url, options: [AVURLAssetPreferPreciseDurationAndTimingKey: true])
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw VideoError("no video track") }
        let readable = try await asset.load(.isReadable)
        let locked = try await asset.load(.hasProtectedContent)
        guard readable, !locked else { throw VideoError("unreadable") }
        let (size, transform, fps, rate, range) = try await track.load(.naturalSize, .preferredTransform, .nominalFrameRate, .estimatedDataRate, .timeRange)
        let duration = try await asset.load(.duration).seconds
        guard size.width > 0, size.height > 0, duration > 0 else { throw VideoError("empty") }
        return VideoSource(url: url, asset: asset, track: track, size: size, transform: transform, duration: duration,
                           fps: Double(fps > 0 ? fps : 30), bitrate: rate, start: range.start,
                           hasAudio: !(try await asset.loadTracks(withMediaType: .audio)).isEmpty)
    }

    /// Every frame's time (seconds from the first) and the keyframes, from the packets, without
    /// decoding them.
    func frameTable() throws -> (times: [Double], keys: [Int]) {
        let reader = try AVAssetReader(asset: asset)
        let out = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
        out.alwaysCopiesSampleData = false
        reader.add(out)
        guard reader.startReading() else { throw VideoError("can't read the video") }
        var frames: [(Double, Bool)] = []
        var first = CMTime.positiveInfinity
        while let s = out.copyNextSampleBuffer() {
            let n = CMSampleBufferGetNumSamples(s)
            guard n > 0 else { continue }
            var key = true
            if let a = CMSampleBufferGetSampleAttachmentsArray(s, createIfNecessary: false) as? [[CFString: Any]], let first = a.first {
                key = !((first[kCMSampleAttachmentKey_NotSync] as? Bool) ?? false)
            }
            let t = CMSampleBufferGetPresentationTimeStamp(s)
            if t.isValid {
                frames.append((t.seconds, key))
                first = CMTimeMinimum(first, t)
            }
        }
        guard reader.status != .failed, !frames.isEmpty else { throw VideoError("no frames") }
        // times count from the first frame, and so does every reader from here on: a frame's time
        // is the one in the table, exactly
        start = first
        frames.sort { $0.0 < $1.0 }
        let t0 = frames[0].0
        let times = frames.map { $0.0 - t0 }
        var keys = frames.indices.filter { frames[$0].1 }
        if keys.first != 0 { keys.insert(0, at: 0) }
        return (times, keys)
    }

    /// The frame at `t` seconds (from the first frame), exactly, as stored: through the reader that
    /// playing and exporting use, so a frame has the same pixels however it's reached.
    func frame(at t: Double) async throws -> CVPixelBuffer {
        let reader = try frames(from: t, to: t + 2 / fps)
        defer { reader.cancel() }
        while let item = reader.next() {
            let (pb, ft) = item
            if ft >= t - 0.001 { return pb }
        }
        throw VideoError("no frame at \(t) s")
    }

    /// The frames from `t` seconds on (to `end`, else the end), in order, as stored.
    func frames(from t: Double, to end: Double? = nil) throws -> FrameReader {
        try FrameReader(self, from: t, to: end)
    }

    /// Small pictures across the video for the timeline: one every `every` seconds.
    func strip(every: Double, height: CGFloat) async -> [UIImage] {
        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        gen.maximumSize = CGSize(width: height * 4, height: height)
        gen.requestedTimeToleranceBefore = CMTime(seconds: every / 2, preferredTimescale: 600)
        gen.requestedTimeToleranceAfter = CMTime(seconds: every / 2, preferredTimescale: 600)
        let n = max(1, Int((duration / every).rounded(.up)))
        var out: [UIImage] = []
        for i in 0..<n {
            let at = CMTimeAdd(start, CMTime(seconds: (Double(i) + 0.5) * every, preferredTimescale: 600))
            if let r = try? await gen.image(at: at) { out.append(UIImage(cgImage: r.image)) } else if let last = out.last { out.append(last) }
        }
        return out
    }
}

/// Decodes a video's frames in order.
final class FrameReader {
    private let reader: AVAssetReader
    private let out: AVAssetReaderTrackOutput
    private let start: CMTime

    init(_ v: VideoSource, from t: Double, to end: Double?) throws {
        reader = try AVAssetReader(asset: v.asset)
        let from = CMTimeAdd(v.start, CMTime(seconds: max(0, t - 0.0005), preferredTimescale: 60000))
        let until = end.map { CMTimeAdd(v.start, CMTime(seconds: $0 + 0.0005, preferredTimescale: 60000)) } ?? .positiveInfinity
        reader.timeRange = CMTimeRange(start: from, end: until)
        out = AVAssetReaderTrackOutput(track: v.track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                                                                         kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()])
        out.alwaysCopiesSampleData = false
        reader.add(out)
        start = v.start
        guard reader.startReading() else { throw VideoError("can't read the video: \(reader.error?.localizedDescription ?? "?")") }
    }

    /// The next frame and its time (seconds from the first frame), or nil at the end.
    func next() -> (CVPixelBuffer, Double)? {
        while let s = out.copyNextSampleBuffer() {
            guard let pb = CMSampleBufferGetImageBuffer(s) else { continue }
            return (pb, CMTimeSubtract(CMSampleBufferGetPresentationTimeStamp(s), start).seconds)
        }
        return nil
    }

    func cancel() { reader.cancelReading() }
}

/// The index of the value in ascending `times` nearest to t.
func nearestIndex(_ times: [Double], _ t: Double) -> Int {
    guard !times.isEmpty else { return 0 }
    var lo = 0, hi = times.count - 1
    while lo < hi {
        let mid = (lo + hi) / 2
        if times[mid] < t { lo = mid + 1 } else { hi = mid }
    }
    return lo > 0 && t - times[lo - 1] < times[lo] - t ? lo - 1 : lo
}
