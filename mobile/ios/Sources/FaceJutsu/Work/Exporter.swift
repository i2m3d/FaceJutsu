import AVFoundation
import CoreVideo
import FaceJutsuEngine
import Foundation
import Photos
import UIKit

/// What an export made.
struct ExportResult {
    var file: URL
    var frames = 0
    var seconds = 0.0
    var bytes = 0
    var size = CGSize.zero
}

/// The export's bitrate, against the video's own, for the Quality setting.
func qualityFactor(_ quality: String) -> Float {
    switch quality {
    case "Good": 1.0
    case "Very high": 2.5
    default: 1.5
    }
}

enum Exporter {
    /// Swaps a whole video into an MP4 at `output`: every frame looked at in order (the ones the
    /// scan hasn't reached yet too), each face swapped with its track's character at that time. The
    /// same size, display matrix and frames, the audio copied. Frames already swapped come from
    /// the work's patches, and those swapped here are kept. `progress(done, total)` after each
    /// frame, and `shown(frame, picture)` every moment with the frame just made; while `paused()`
    /// it waits (the GPU can't be used from the background).
    static func video(source: VideoSource, scan: VideoScan, times: [Double], planner: Planner, work: String, quality: String, output: URL,
                      progress: @escaping @Sendable (Int, Int) -> Void, shown: @escaping @Sendable (Int, UIImage) -> Void,
                      paused: @escaping @Sendable () async -> Bool) async throws -> ExportResult {
        guard let swapper = await Engine.shared.ready() else { throw VideoError("The models couldn't be loaded.") }
        let asset = source.asset, video = source.track
        let audio = try await asset.loadTracks(withMediaType: .audio).first
        let audioHint = try await audio?.load(.formatDescriptions).first
        let total = max(1, times.count)
        let (w, h) = (Int(source.size.width), Int(source.size.height))

        let videoReader = try AVAssetReader(asset: asset)
        let videoOut = AVAssetReaderTrackOutput(track: video, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                                                                                kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()])
        videoOut.alwaysCopiesSampleData = false
        videoReader.add(videoOut)
        var audioReader: AVAssetReader?
        var audioOut: AVAssetReaderTrackOutput?
        if let audio {
            let r = try AVAssetReader(asset: asset)
            let o = AVAssetReaderTrackOutput(track: audio, outputSettings: nil)
            r.add(o)
            (audioReader, audioOut) = (r, o)
        }

        try? FileManager.default.removeItem(at: output)
        let writer = try AVAssetWriter(outputURL: output, fileType: .mp4)
        let bitrate = max(Int(source.bitrate * qualityFactor(quality)), 2_000_000)
        let videoIn = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: w,
            AVVideoHeightKey: h,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: bitrate,
                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
            ],
            AVVideoColorPropertiesKey: [
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2,
            ],
        ])
        videoIn.transform = source.transform
        videoIn.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: videoIn, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
        ])
        writer.add(videoIn)
        var audioIn: AVAssetWriterInput?
        if audio != nil {
            let i = AVAssetWriterInput(mediaType: .audio, outputSettings: nil, sourceFormatHint: audioHint)
            i.expectsMediaDataInRealTime = false
            writer.add(i)
            audioIn = i
        }

        guard videoReader.startReading() else { throw VideoError("can't read the video: \(videoReader.error?.localizedDescription ?? "?")") }
        if let audioReader, !audioReader.startReading() { throw VideoError("can't read the audio: \(audioReader.error?.localizedDescription ?? "?")") }
        guard writer.startWriting() else { throw VideoError("can't write the video: \(writer.error?.localizedDescription ?? "?")") }
        writer.startSession(atSourceTime: source.start)

        let start = Date()
        var done = 0
        var videoDone = false
        var audioDone = audioIn == nil
        var pendingAudio: CMSampleBuffer?
        var lastVideo = CMTime.zero
        let lead = CMTime(seconds: 1, preferredTimescale: 600)
        let rotation = UInt32(source.rotation)
        scan.beginPass(from: 0)
        var next = 0
        var lastShown = Date.distantPast
        do {
            while !videoDone || !audioDone {
                try Task.checkCancellation()
                var moved = false
                if !videoDone && videoIn.isReadyForMoreMediaData {
                    moved = true
                    if let sample = videoOut.copyNextSampleBuffer() {
                        guard let frame = CMSampleBufferGetImageBuffer(sample) else { continue }
                        let pts = CMSampleBufferGetPresentationTimeStamp(sample)
                        // the frame's own number, as playing and scanning give it (a frame that
                        // comes again at the start only isn't looked at twice)
                        let i = min(times.count - 1, nearestIndex(times, CMTimeSubtract(pts, source.start).seconds))
                        // (every frame has been looked at by now, unless one wouldn't come)
                        let isNew = i >= next && scan.state(frame: UInt32(i)) != 2
                        // a frame tried again while the app is away, until it's back
                        while true {
                            while await paused() { try await Task.sleep(for: .milliseconds(200)) }
                            do {
                                let made = try swapOne(frame, index: i, isNew: isNew, at: pts, scan: scan, swapper: swapper, planner: planner, times: times, work: work, adaptor: adaptor, rotation: rotation)
                                if Date().timeIntervalSince(lastShown) > 0.15, let picture = Pictures.image(made, rotation: Int(rotation), maxSide: 1200) {
                                    lastShown = Date()
                                    shown(i, picture)
                                }
                                break
                            } catch {
                                if await paused() { continue }
                                throw error
                            }
                        }
                        next = max(next, i + 1)
                        done += 1
                        lastVideo = pts
                        progress(done, max(total, done))
                    } else {
                        if videoReader.status == .failed { throw VideoError("reading the video: \(videoReader.error?.localizedDescription ?? "?")") }
                        videoIn.markAsFinished()
                        videoDone = true
                    }
                }
                if !audioDone, let audioIn, let audioOut, audioIn.isReadyForMoreMediaData {
                    if pendingAudio == nil {
                        pendingAudio = audioOut.copyNextSampleBuffer()
                        if pendingAudio == nil {
                            audioIn.markAsFinished()
                            audioDone = true
                            moved = true
                        }
                    }
                    if let a = pendingAudio, videoDone || !videoIn.isReadyForMoreMediaData || CMSampleBufferGetPresentationTimeStamp(a) <= lastVideo + lead {
                        guard audioIn.append(a) else { throw VideoError("writing the audio: \(writer.error?.localizedDescription ?? "?")") }
                        pendingAudio = nil
                        moved = true
                    }
                }
                if writer.status == .failed { throw VideoError("writing: \(writer.error?.localizedDescription ?? "?")") }
                if !moved { try await Task.sleep(for: .milliseconds(1)) }
            }
        } catch {
            scan.endPass(to: UInt32(next))
            videoReader.cancelReading()
            audioReader?.cancelReading()
            writer.cancelWriting()
            try? FileManager.default.removeItem(at: output)
            throw error
        }
        scan.endPass(to: UInt32(next))
        await writer.finishWriting()
        guard writer.status == .completed else { throw VideoError("finishing: \(writer.error?.localizedDescription ?? "?")") }
        let bytes = (try? output.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        return ExportResult(file: output, frames: done, seconds: Date().timeIntervalSince(start), bytes: bytes, size: source.shown)
    }

    /// Looks at one frame (the next of the pass), swaps its faces into a new pixel buffer (or
    /// takes its patch) and appends it. Returns that buffer.
    private static func swapOne(_ frame: CVPixelBuffer, index i: Int, isNew: Bool, at pts: CMTime, scan: VideoScan, swapper: FaceSwapper, planner: Planner, times: [Double],
                                work: String, adaptor: AVAssetWriterInputPixelBufferAdaptor, rotation: UInt32) throws -> CVPixelBuffer {
        try autoreleasepool {
            guard let pool = adaptor.pixelBufferPool else { throw VideoError("the writer has no pixel buffers") }
            var made: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &made)
            guard let out = made else { throw VideoError("no pixel buffer for the output") }
            let plan = try withPixels(frame) { a, w, h, s in
                if isNew { _ = try scan.look(frame: UInt32(i), input: a, width: w, height: h, stride: s, rotation: rotation) }
                return planner.plan(scan.faces(frame: UInt32(i)), at: times[i])
            }
            let key = plan.isEmpty ? nil : Patches.key(frame: i, width: CVPixelBufferGetWidth(frame), height: CVPixelBufferGetHeight(frame), rotation: Int(rotation), plan: plan)
            if let key, let patch = Patches.get(work: work, key: key), Patches.apply(patch, frame: frame, into: out) {
                // made before
            } else {
                try withPixels(frame) { a, w, h, s in
                    try withPixels(out, readOnly: false) { o, _, _, os in
                        _ = try swapper.swapPlanned(input: a, output: o, width: w, height: h, inputStride: s, outputStride: os, rotation: rotation, plan: plan)
                    }
                }
                if let key, let patch = Patches.diff(frame, out) { Patches.put(work: work, key: key, patch) }
            }
            CVBufferPropagateAttachments(frame, out)
            guard adaptor.append(out, withPresentationTime: pts) else { throw VideoError("writing a frame failed") }
            return out
        }
    }

    /// A photo with its faces swapped as planned, full size, upright.
    static func photo(_ image: CGImage, plan: [PlannedFace], maxSide: CGFloat = .infinity) async throws -> UIImage {
        guard let swapper = await Engine.shared.ready() else { throw VideoError("The models couldn't be loaded.") }
        guard let input = pixelBuffer(from: image), let out = makePixelBuffer(width: image.width, height: image.height) else { throw VideoError("no pixel buffer") }
        return try await Task.detached(priority: .userInitiated) {
            try withPixels(input) { a, w, h, s in
                try withPixels(out, readOnly: false) { o, _, _, os in
                    _ = try swapper.swapPlanned(input: a, output: o, width: w, height: h, inputStride: s, outputStride: os, rotation: 0, plan: plan)
                }
            }
            guard let picture = Pictures.image(out, rotation: 0, maxSide: maxSide) else { throw VideoError("no picture") }
            return picture
        }.value
    }
}

/// Where finished files go: Photos, or Documents (Files).
enum Saver {
    private static func mayAdd() async throws {
        let status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
        guard status == .authorized || status == .limited else {
            throw VideoError("FaceJutsu can't add to Photos. Allow it in Settings, or save to Files.")
        }
    }

    static func videoToPhotos(_ url: URL) async throws {
        try await mayAdd()
        try await PHPhotoLibrary.shared().performChanges {
            PHAssetCreationRequest.forAsset().addResource(with: .video, fileURL: url, options: nil)
        }
    }

    static func photoToPhotos(_ data: Data) async throws {
        try await mayAdd()
        try await PHPhotoLibrary.shared().performChanges {
            PHAssetCreationRequest.forAsset().addResource(with: .photo, data: data, options: nil)
        }
    }

    /// Copies a file into Documents as `name` (a number added when that's taken).
    static func toFiles(_ url: URL, name: String) throws -> URL {
        let base = (name as NSString).deletingPathExtension, ext = (name as NSString).pathExtension
        var dest = Store.documents.appendingPathComponent(name)
        var n = 2
        while FileManager.default.fileExists(atPath: dest.path) {
            dest = Store.documents.appendingPathComponent("\(base) \(n).\(ext)")
            n += 1
        }
        try FileManager.default.copyItem(at: url, to: dest)
        return dest
    }
}
