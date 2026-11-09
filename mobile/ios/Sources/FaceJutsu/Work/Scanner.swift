import CoreVideo
import FaceJutsuEngine
import Foundation
import UIKit

/// Finds a video's people as the desktop does (web/app/scanner.js): every frame from a second
/// behind the playhead to a few ahead gets looked at first, following the playhead wherever it
/// goes, and once that's done a skim of keyframes over the whole video says who is in it. The
/// export holds it while it walks the whole video itself.
actor Scanner {
    private let scan: VideoScan
    private let source: VideoSource
    private let times: [Double]
    private var playhead: Double = 0
    private var playing = false
    private var held = false
    private var busy = false
    private var skim: [UInt32] = []
    private var skimAt: Double = -100
    private var loop: Task<Void, Never>?
    /// nothing left that it looks at by itself; and how many looks it's taken
    private var idle = false
    private var looks = 0
    private let onThumb: @Sendable (Int, UIImage) -> Void
    private let onChange: @Sendable () -> Void

    /// seconds of the window behind and ahead of the playhead (ahead while playing)
    static let behind = 1.0, ahead = 4.0, aheadPlaying = 6.0
    /// seconds between the skim's keyframes, at the finest
    static let skimStep = 2.0
    /// seconds without getting further, while waiting for it, before going on without that frame
    static let givesUp = 3.0

    init(scan: VideoScan, source: VideoSource, times: [Double], onThumb: @escaping @Sendable (Int, UIImage) -> Void, onChange: @escaping @Sendable () -> Void) {
        self.scan = scan
        self.source = source
        self.times = times
        self.onThumb = onThumb
        self.onChange = onChange
    }

    func start() {
        guard loop == nil else { return }
        loop = Task { await run() }
    }

    func stop() {
        loop?.cancel()
        loop = nil
    }

    /// Follows the playhead.
    func follow(_ t: Double, playing: Bool) {
        playhead = t
        self.playing = playing
    }

    /// Stops looking (what it found stays) until `release`, once the stretch going on has ended.
    func hold() async {
        held = true
        while busy { try? await Task.sleep(for: .milliseconds(20)) }
    }

    func release() { held = false }

    /// Walks the playhead over every frame not looked at yet, so the whole video is found exactly
    /// as scrubbing through it would (the same passes, people regrouped as they go). `progress`
    /// with the frames looked at and all of them. While `paused()` it waits; a frame that won't
    /// come ends the walk after a while.
    func lookAtAll(progress: @Sendable (Int, Int) -> Void, paused: @Sendable () async -> Bool) async throws {
        try await settle()
        var at = -1, since = Date()
        while true {
            let states = [UInt8](scan.states())
            guard let f = states.firstIndex(where: { $0 != 2 }) else { break }
            progress(states.reduce(0) { $0 + ($1 == 2 ? 1 : 0) }, states.count)
            playhead = times[f]
            playing = false
            if await paused() || f != at {
                at = f
                since = Date()
            } else if Date().timeIntervalSince(since) > Self.givesUp {
                break
            }
            try await Task.sleep(for: .milliseconds(30))
        }
    }

    /// Waits until it has looked at all it looks at by itself: the window and the skim over the
    /// whole video, which say who is in it. (Gives up on a frame that won't come after a while.)
    private func settle() async throws {
        var seen = looks, since = Date()
        while !idle {
            try await Task.sleep(for: .milliseconds(30))
            if looks != seen {
                seen = looks
                since = Date()
            } else if Date().timeIntervalSince(since) > Self.givesUp {
                return
            }
        }
    }

    /// The frames around the playhead it looks at first: [lo, hi).
    private func window() -> (Int, Int) {
        let lo = nearestIndex(times, playhead - Self.behind)
        let hi = min(times.count, nearestIndex(times, playhead + (playing ? Self.aheadPlaying : Self.ahead)) + 1)
        return (lo, max(hi, lo + 1))
    }

    /// How much of the window has been looked at, 0 to 1.
    func coverage() -> Double {
        let (lo, hi) = window()
        return scan.coverage(from: UInt32(lo), to: UInt32(hi))
    }

    private func run() async {
        while !Task.isCancelled {
            if held {
                try? await Task.sleep(for: .milliseconds(100))
                continue
            }
            let states = [UInt8](scan.states())
            let (lo, hi) = window()
            let at = min(max(nearestIndex(times, playhead), lo), hi - 1)
            // the frames near the playhead: ahead of it first, then behind
            if let f = (at..<hi).first(where: { states[$0] != 2 }) ?? (lo..<at).first(where: { states[$0] != 2 }) {
                idle = false
                var end = f
                while end < hi && states[end] != 2 { end += 1 }
                await pass(from: f, to: end)
                continue
            }
            // then the skim, its order made again when the playhead has moved on
            if abs(playhead - skimAt) > 5 || skim.isEmpty && skimAt < -50 {
                skim = scan.skimOrder(at: UInt32(at), from: 0, to: UInt32(times.count), step: Self.skimStep)
                skimAt = playhead
            }
            while let k = skim.first, states.indices.contains(Int(k)), states[Int(k)] != 0 { skim.removeFirst() }
            if let k = skim.first {
                idle = false
                skim.removeFirst()
                await glance(Int(k))
                looks += 1
                continue
            }
            idle = true
            try? await Task.sleep(for: .milliseconds(150))
        }
    }

    private func thumbs(_ wanted: [ThumbWanted], _ pb: CVPixelBuffer) {
        for w in wanted {
            if let image = Pictures.face(pb, rotation: source.rotation, box: w.bbox) { onThumb(Int(w.person), image) }
        }
    }

    private func glance(_ k: Int) async {
        busy = true
        defer { busy = false }
        guard let pb = try? await source.frame(at: times[k]) else { return }
        let rotation = UInt32(source.rotation)
        do {
            let wanted = try withPixels(pb) { a, w, h, s in try scan.glance(frame: UInt32(k), input: a, width: w, height: h, stride: s, rotation: rotation) }
            thumbs(wanted, pb)
            onChange()
        } catch {
            // (the GPU refuses work while the app is in the background: tried again later)
            try? await Task.sleep(for: .milliseconds(500))
        }
    }

    /// A pass over frames [from, to), none looked at yet, stopping early when the playhead moves
    /// away or the export wants the GPU.
    private func pass(from: Int, to: Int) async {
        busy = true
        defer { busy = false }
        guard let reader = try? source.frames(from: times[from], to: times[to - 1] + 2 / source.fps) else {
            try? await Task.sleep(for: .milliseconds(300))
            return
        }
        defer { reader.cancel() }
        scan.beginPass(from: UInt32(from))
        var next = from
        let rotation = UInt32(source.rotation)
        while let item = reader.next() {
            let (pb, t) = item
            let i = nearestIndex(times, t)
            if i < next { continue }
            if i >= to { break }
            do {
                let wanted = try withPixels(pb) { a, w, h, s in try scan.look(frame: UInt32(i), input: a, width: w, height: h, stride: s, rotation: rotation) }
                thumbs(wanted, pb)
            } catch {
                break
            }
            next = i + 1
            looks += 1
            onChange()
            await Task.yield()
            let (lo, hi) = window()
            if held || Task.isCancelled || next < lo || next >= hi { break }
        }
        scan.endPass(to: UInt32(next))
        onChange()
        if next == from { try? await Task.sleep(for: .milliseconds(400)) }
    }
}
