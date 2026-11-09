import AVFoundation
import CoreVideo
import FaceJutsuEngine
import Foundation
import SwiftUI
import UIKit

/// The work going on for the open video or photos: the scan, the picture on the stage (the paused
/// frame swapped, or playback swapping as it goes, slower than real time as the desktop's is), and
/// the export.
@MainActor
final class LiveWork {
    unowned let model: EditorModel
    let id: String
    var source: VideoSource?
    var scan: VideoScan?
    private var scanner: Scanner?
    /// who each analysis person is (root)
    private(set) var roots: [Int: Int] = [:]
    private var refreshDue = false
    private var pictureTask: Task<Void, Never>?
    /// fetching paused frames, and the frame the picture should come to
    private var frameTask: Task<Void, Never>?
    private var wanted: Int?
    private var fetchRun = 0
    /// what plays the video's sound while the picture plays, and when it was last put right
    private var sound: AVPlayer?
    private var soundSynced = Date.distantPast
    /// working out the green line: with what plan, and the frames made since it began
    private var madeTask: Task<Void, Never>?
    private var madePlanner: Planner?
    private var madeSince: Set<Int> = []
    private var playTask: Task<Void, Never>?
    private var exportTask: Task<Void, Never>?
    /// swapped frames: frame -> (the plan they were made with, the picture)
    private var cache: [Int: (planner: Planner, image: UIImage)] = [:]
    private var original: (frame: Int, pb: CVPixelBuffer, image: UIImage)?
    /// the app is in front (the GPU can't be used from the background)
    var active = true
    private var backgroundTask: UIBackgroundTaskIdentifier = .invalid

    init(model: EditorModel, id: String) {
        self.model = model
        self.id = id
    }

    var folder: URL { Store.workFolder(id) }

    // MARK: the scan

    func startScan(source: VideoSource, scan: VideoScan) {
        self.source = source
        self.scan = scan
        let s = Scanner(scan: scan, source: source, times: model.times,
                        onThumb: { [weak self] person, image in Task { @MainActor in self?.took(person, image) } },
                        onChange: { [weak self] in Task { @MainActor in self?.scanChanged() } })
        scanner = s
        let t = model.playhead
        Task {
            await s.follow(t, playing: false)
            await s.start()
        }
        refresh()
        refreshMade()
    }

    private func took(_ person: Int, _ image: UIImage) {
        model.faceThumbs[person] = image
        let url = folder.appendingPathComponent("p\(person).jpg")
        Task.detached { try? image.jpegData(compressionQuality: 0.8)?.write(to: url, options: .atomic) }
    }

    private func scanChanged() {
        guard !refreshDue else { return }
        refreshDue = true
        Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(250))
            refreshDue = false
            refresh()
        }
    }

    /// Takes in what the scan has found.
    func refresh() {
        guard let scan else { return }
        let people = scan.people()
        roots = Dictionary(people.map { (Int($0.person), Int($0.root)) }, uniquingKeysWith: { a, _ in a })
        model.states = [UInt8](scan.states())
        model.applyScan(people)
        if model.planner != madePlanner { refreshMade() }
        facesAtPlayhead()
        if let scanner {
            Task {
                let c = await scanner.coverage()
                await MainActor.run { self.model.scouting = c < 1 ? c : nil }
            }
        }
        if !model.playing && model.stageFrame != model.frameIndex || model.swapping { refreshPicture() }
    }

    /// The faces at the playhead, by track.
    func facesAtPlayhead() {
        guard let scan, !model.times.isEmpty else { return }
        let i = model.frameIndex
        let faces = scan.faces(frame: UInt32(i))
        let size = model.videoSize
        var boxes: [Int: CGRect] = [:]
        for f in faces {
            guard let track = model.doc.people.first(where: { p in p.ids.contains { (roots[$0] ?? $0) == Int(f.person) } }), f.face.bbox.count == 4 else { continue }
            // (a face deleted before its track was merged isn't the track's here)
            let t = model.playhead, person = Int(f.person)
            if track.leftOut?.contains(where: { l in t >= l.t0 && t < l.t1 && l.ids.contains { (roots[$0] ?? $0) == person } }) == true { continue }
            let b = f.face.bbox
            boxes[track.id] = CGRect(x: b[0] / size.width, y: b[1] / size.height, width: (b[2] - b[0]) / size.width, height: (b[3] - b[1]) / size.height)
        }
        model.faceBoxes = boxes
        for (id, b) in boxes { model.lastBoxes[id] = (model.playhead, b) }
    }

    func follow() {
        guard let scanner else { return }
        let t = model.playhead, playing = model.playing
        Task { await scanner.follow(t, playing: playing) }
    }

    // MARK: characters

    func syncIdentities(_ characters: [SwapCharacter]) {
        Task {
            guard let swapper = await Engine.shared.ready() else { return }
            for c in characters { try? swapper.setIdentity(id: UInt32(c.id), embedding: c.embedding) }
            self.cache = [:]
            self.refreshPicture()
        }
    }

    // MARK: the picture

    private func show(_ image: UIImage?, frame: Int) {
        model.stageImage = image
        model.stageFrame = frame
    }

    /// Shows the paused frame: as it is at once, swapped once its faces are found. While the
    /// playhead moves (scrubbing), each frame fetched is shown as it comes (swapped when that's
    /// kept in memory) and the next fetch is of where the playhead is by then; the swap waits until
    /// it stops.
    func refreshPicture() {
        if model.mode == .photo {
            refreshPhoto()
            return
        }
        guard let source, !model.playing, !model.times.isEmpty else { return }
        wanted = model.frameIndex
        // a swap of a frame left behind is no use
        pictureTask?.cancel()
        if frameTask == nil { fetchFrames(source) }
    }

    /// Fetches the frame wanted until it's the one showing, then swaps it.
    private func fetchFrames(_ source: VideoSource) {
        let rotation = source.rotation
        fetchRun &+= 1
        let run = fetchRun
        frameTask = Task {
            // (a fetch stopped by playing mustn't let go of the one after it)
            defer { if fetchRun == run { frameTask = nil } }
            while let i = wanted, !Task.isCancelled {
                if original?.frame != i {
                    guard let pb = try? await source.frame(at: model.times[i]) else { break }
                    let image = await Task.detached { Pictures.image(pb, rotation: rotation) }.value
                    guard !Task.isCancelled, let image else { break }
                    original = (i, pb, image)
                    if wanted != i {
                        // on the way: show it, and go on to where the playhead is now
                        let kept = await keptPicture(i, pb)
                        guard !Task.isCancelled else { break }
                        show(kept ?? image, frame: i)
                        continue
                    }
                }
                wanted = nil
                settle(i)
            }
        }
    }

    /// Frame `i` as already swapped with the tracks as they are now, in memory or on disk; nil
    /// when it isn't (or the original is what's wanted).
    private func keptPicture(_ i: Int, _ pb: CVPixelBuffer) async -> UIImage? {
        let planner = model.planner
        guard !model.before, !planner.isEmpty, let source, let scan, model.times.indices.contains(i) else { return nil }
        if let c = cache[i], c.planner == planner { return c.image }
        guard scan.state(frame: UInt32(i)) == 2 else { return nil }
        let t = model.times[i], rotation = source.rotation, work = id
        let image = await Task.detached { () -> UIImage? in
            let plan = planner.plan(scan.faces(frame: UInt32(i)), at: t)
            return plan.isEmpty ? nil : patched(pb, index: i, plan: plan, rotation: rotation, work: work)
        }.value
        if let image { remember(i, planner, image) }
        return image
    }

    /// The paused frame `i`, fetched: shown, and swapped.
    private func settle(_ i: Int) {
        guard let source, let original, original.frame == i, model.times.indices.contains(i) else { return }
        let scan = self.scan
        let t = model.times[i], rotation = source.rotation
        pictureTask = Task {
            let planner = model.planner
            guard let scan else {
                show(original.image, frame: i)
                return
            }
            if model.before || planner.isEmpty {
                show(original.image, frame: i)
                model.swapping = false
                return
            }
            if let c = cache[i], c.planner == planner {
                show(c.image, frame: i)
                model.swapping = false
                return
            }
            // kept from before: no flash of the original
            if let kept = await keptPicture(i, original.pb) {
                guard !Task.isCancelled else { return }
                show(kept, frame: i)
                model.swapping = false
                return
            }
            guard !Task.isCancelled else { return }
            if model.stageFrame != i || model.stageImage == nil { show(original.image, frame: i) }
            model.swapping = true
            while scan.state(frame: UInt32(i)) != 2 {
                try? await Task.sleep(for: .milliseconds(80))
                if Task.isCancelled { return }
            }
            let plan = planner.plan(scan.faces(frame: UInt32(i)), at: t)
            if plan.isEmpty {
                show(original.image, frame: i)
                model.swapping = false
                return
            }
            guard let swapper = await Engine.shared.ready() else { return }
            let pb = original.pb, work = id
            let image = await Task.detached { swapped(pb, index: i, plan: plan, swapper: swapper, rotation: rotation, work: work) }.value
            guard !Task.isCancelled else { return }
            if let image {
                remember(i, planner, image)
                show(image, frame: i)
            }
            model.swapping = false
        }
    }

    private func remember(_ i: Int, _ planner: Planner, _ image: UIImage) {
        // (a few: each is several MB, and the disk has them all)
        if cache.count > 30, let k = cache.keys.first { cache.removeValue(forKey: k) }
        cache[i] = (planner, image)
        if planner == model.planner {
            madeSince.insert(i)
            if model.made.indices.contains(i) { model.made[i] = true }
        }
    }

    /// Works out which frames are already swapped as the tracks now say (the timeline's green
    /// line): each frame's key against the patches kept. Soon, off the main thread; a newer call
    /// takes over.
    func refreshMade() {
        guard let source, let scan, !model.times.isEmpty else { return }
        let planner = model.planner, times = model.times, work = id, rotation = source.rotation
        let (w, h) = (Int(source.size.width), Int(source.size.height))
        if model.made.count != times.count { model.made = [Bool](repeating: false, count: times.count) }
        madePlanner = planner
        madeSince = []
        madeTask?.cancel()
        madeTask = Task.detached(priority: .utility) { [weak self] in
            try? await Task.sleep(for: .milliseconds(300))
            if Task.isCancelled { return }
            let keys = Patches.keys(work: work)
            var found = [Bool](repeating: false, count: times.count)
            if !keys.isEmpty && !planner.isEmpty {
                for i in times.indices {
                    if i % 256 == 0 && Task.isCancelled { return }
                    let plan = planner.plan(scan.faces(frame: UInt32(i)), at: times[i])
                    if !plan.isEmpty && keys.contains(Patches.key(frame: i, width: w, height: h, rotation: rotation, plan: plan)) { found[i] = true }
                }
            }
            let done = found
            await MainActor.run {
                guard let self, !Task.isCancelled else { return }
                var made = done
                for i in self.madeSince where made.indices.contains(i) { made[i] = true }
                self.model.made = made
            }
        }
    }

    /// The photo showing, with its faces swapped as planned.
    private func refreshPhoto() {
        guard model.doc.photos.indices.contains(model.photoIndex) else { return }
        let photo = model.doc.photos[model.photoIndex]
        let plan = model.photoPlan(photo)
        // (names everything the swap depends on, its settings too)
        let key = Patches.key(frame: 0, width: photo.width, height: photo.height, rotation: 0, plan: plan)
        if plan.isEmpty {
            model.photoAfter[photo.id] = nil
            return
        }
        if photoKeys[photo.id] == key, model.photoAfter[photo.id] != nil { return }
        let url = folder.appendingPathComponent(photo.file)
        pictureTask?.cancel()
        model.swapping = true
        pictureTask = Task {
            // (a slider's drag swaps once it rests, not at every step of it)
            try? await Task.sleep(for: .milliseconds(120))
            if Task.isCancelled { return }
            // the swap on the photo as saved (its faces are in its pixels), shown smaller
            guard let cg = await Task.detached(operation: { UIImage(contentsOfFile: url.path)?.cgImage }).value,
                  let image = try? await Exporter.photo(cg, plan: plan, maxSide: 2048), !Task.isCancelled else { return }
            photoKeys[photo.id] = key
            model.photoAfter[photo.id] = image
            model.swapping = false
        }
    }

    private var photoKeys: [Int: String] = [:]

    // MARK: playing

    /// Plays from the playhead: each frame swapped once its faces are found, never faster than
    /// real time.
    func play() {
        guard let source, let scan, !model.times.isEmpty else { return }
        // from the end: from the start
        if model.frameIndex >= model.times.count - 1 { model.playhead = 0 }
        model.playing = true
        follow()
        pictureTask?.cancel()
        frameTask?.cancel()
        frameTask = nil
        wanted = nil
        let from = model.playhead, times = model.times, rotation = source.rotation, work = id
        playTask?.cancel()
        playTask = Task.detached(priority: .userInitiated) { [weak self] in
            guard let reader = try? source.frames(from: from) else { return }
            defer { reader.cancel() }
            let swapper = await Engine.shared.ready()
            var wall = Date()
            var first: Double?
            while let item = reader.next() {
                if Task.isCancelled { break }
                let (pb, ft) = item
                let i = nearestIndex(times, ft)
                guard let now = await MainActor.run(body: { () -> (Bool, Planner, UIImage?)? in
                    guard let self else { return nil }
                    let p = self.model.planner
                    return (self.model.before, p, self.cache[i].flatMap { $0.planner == p ? $0.image : nil })
                }) else { break }
                let (before, planner, cached) = now
                var image = cached
                if image == nil, !before, !planner.isEmpty, let swapper {
                    while scan.state(frame: UInt32(i)) != 2 {
                        try? await Task.sleep(for: .milliseconds(30))
                        if Task.isCancelled { return }
                    }
                    let plan = planner.plan(scan.faces(frame: UInt32(i)), at: times[i])
                    if !plan.isEmpty {
                        image = swapped(pb, index: i, plan: plan, swapper: swapper, rotation: rotation, work: work)
                        if let image { await MainActor.run { self?.remember(i, planner, image) } }
                    }
                }
                if image == nil { image = Pictures.image(pb, rotation: rotation) }
                let start = first ?? ft
                first = start
                let due = wall.addingTimeInterval(ft - start).timeIntervalSinceNow
                // behind (a swap took a while): real time goes on from this frame, without the
                // sound, which only plays while the picture keeps up
                let keepingUp = due > -0.15
                if !keepingUp { wall = Date().addingTimeInterval(start - ft) }
                if due > 0 { try? await Task.sleep(for: .seconds(due)) }
                if Task.isCancelled { break }
                let shown = image
                await MainActor.run {
                    guard let self, self.model.playing else { return }
                    self.show(shown, frame: i)
                    self.model.playhead = ft
                    self.followSound(ft, keepingUp: keepingUp)
                }
            }
            await MainActor.run {
                guard let self, !Task.isCancelled else { return }
                self.sound?.pause()
                self.model.playing = false
                self.follow()
            }
        }
    }

    /// The sound goes with the picture: restarted where the picture is when it's drifted, and
    /// silent while the picture can't keep up or the sound is off.
    private func followSound(_ t: Double, keepingUp: Bool) {
        guard let source, model.hasSound, model.soundOn, keepingUp else {
            sound?.pause()
            return
        }
        if sound == nil {
            // (plays with the ringer switch off, as a video does)
            try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .moviePlayback)
            try? AVAudioSession.sharedInstance().setActive(true)
            sound = AVPlayer(url: source.url)
        }
        guard let sound else { return }
        let at = CMTimeAdd(source.start, CMTime(seconds: t, preferredTimescale: 60000))
        let drift = abs(CMTimeSubtract(sound.currentTime(), at).seconds)
        if sound.rate == 0 || (drift > 0.1 && Date().timeIntervalSince(soundSynced) > 0.3) {
            soundSynced = Date()
            sound.seek(to: at, toleranceBefore: .zero, toleranceAfter: .zero)
            sound.play()
        }
    }

    func stop() {
        sound?.pause()
        playTask?.cancel()
        playTask = nil
        follow()
        refreshPicture()
    }

    // MARK: export

    var exporting: Bool { exportTask != nil }

    func startExport() {
        guard exportTask == nil else { return }
        let quality = UserDefaults.standard.string(forKey: "exportQuality") ?? "High"
        let saveTo = UserDefaults.standard.string(forKey: "saveTo") ?? "Photos"
        if model.mode == .photo {
            exportPhotos(saveTo: saveTo)
            return
        }
        guard let source, let scan, let scanner else { return }
        // room for the file, and a little over
        let need = Int64(Double(max(Float(2_000_000), source.bitrate * qualityFactor(quality))) / 8 * source.duration * 1.15) + 20_000_000
        let free = Store.freeBytes
        if free < need {
            model.export = .noRoom(need: Int(need / 1_000_000), free: Int(free / 1_000_000))
            return
        }
        if model.playing { model.stop() }
        let total = model.times.count
        model.export = .running(done: 0, total: total, left: nil)
        let times = model.times
        let out = FileManager.default.temporaryDirectory.appendingPathComponent("export-\(id).mp4")
        let name = exportName()
        UIApplication.shared.isIdleTimerDisabled = UserDefaults.standard.object(forKey: "keepAwake") as? Bool ?? true
        exportTask = Task {
            do {
                // who is in the video first, all of it, the way scrubbing finds it: who becomes
                // whom is only known after that
                try await scanner.lookAtAll(progress: { [weak self] d, t in
                    Task { @MainActor in
                        guard let self, self.exportTask != nil else { return }
                        self.model.export = .scouting(done: d, total: t)
                    }
                }, paused: { [weak self] in await MainActor.run { !(self?.active ?? true) } })
                await scanner.hold()
                refresh()
                let planner = model.planner, started = Date()
                let result = try await Exporter.video(source: source, scan: scan, times: times, planner: planner, work: id, quality: quality, output: out,
                                                      progress: { [weak self] d, t in
                                                          Task { @MainActor in
                                                              guard let self, self.exportTask != nil else { return }
                                                              let rate = Double(d) / max(Date().timeIntervalSince(started), 0.1)
                                                              self.model.export = self.active ? .running(done: d, total: t, left: Double(t - d) / max(rate, 0.01)) : .paused(done: d, total: t)
                                                          }
                                                      },
                                                      shown: { [weak self] i, picture in
                                                          Task { @MainActor in
                                                              guard let self, self.exportTask != nil, times.indices.contains(i) else { return }
                                                              self.model.playhead = times[i]
                                                              self.show(picture, frame: i)
                                                          }
                                                      },
                                                      paused: { [weak self] in await MainActor.run { !(self?.active ?? true) } })
                let saved: String
                var kept: URL? = result.file
                if saveTo == "Files" {
                    kept = try Saver.toFiles(result.file, name: name)
                    saved = "Saved to Files"
                } else {
                    try await Saver.videoToPhotos(result.file)
                    saved = "Saved to Photos"
                }
                let mb = max(1, result.bytes / 1_000_000)
                let p = Int(min(result.size.width, result.size.height))
                model.export = .finished(facts: "\(result.frames) frames · \(p)p · \(mb) MB · \(Int(result.seconds.rounded())) s", savedTo: saved, file: kept)
                updatePicture(from: result.file)
                // (the picture follows the export only every moment: it ended on the last frame)
                model.playhead = times[times.count - 1]
            } catch is CancellationError {
                model.export = nil
            } catch {
                var done = 0
                switch model.export {
                case .running(let d, _, _), .paused(let d, _): done = d
                default: break
                }
                model.export = .failed(done: done, total: total)
                model.show(describe(error))
            }
            UIApplication.shared.isIdleTimerDisabled = false
            exportTask = nil
            await scanner.release()
            refresh()
            refreshMade()
            follow()
            refreshPicture()
            model.save()
        }
    }

    func stopExport() {
        exportTask?.cancel()
    }

    private func exportName() -> String {
        let base = ((model.work?.name ?? "") as NSString).deletingPathExtension
        let names = Set(model.doc.people.flatMap { $0.parts.compactMap { $0.off ? nil : model.character($0.s.character)?.name } })
        let who = names.sorted().prefix(3).joined(separator: ", ")
        let head = base.isEmpty ? "FaceJutsu" : base
        return who.isEmpty ? "\(head).mp4" : "\(head) - \(who).mp4"
    }

    private func exportPhotos(saveTo: String) {
        let photos = model.doc.photos.filter { !model.photoPlan($0).isEmpty }
        guard !photos.isEmpty else {
            model.show("Give a face a character first.")
            return
        }
        model.export = .running(done: 0, total: photos.count, left: nil)
        let folder = self.folder
        exportTask = Task {
            var done = 0
            do {
                for p in photos {
                    try Task.checkCancellation()
                    guard let ui = UIImage(contentsOfFile: folder.appendingPathComponent(p.file).path), let cg = ui.cgImage else { continue }
                    let image = try await Exporter.photo(cg, plan: model.photoPlan(p))
                    guard let data = image.jpegData(compressionQuality: 0.92) else { continue }
                    if saveTo == "Files" {
                        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("photo-\(p.id).jpg")
                        try data.write(to: tmp, options: .atomic)
                        _ = try Saver.toFiles(tmp, name: "FaceJutsu \(p.id).jpg")
                    } else {
                        try await Saver.photoToPhotos(data)
                    }
                    done += 1
                    model.export = .running(done: done, total: photos.count, left: nil)
                }
                model.export = .finished(facts: done == 1 ? "1 photo" : "\(done) photos", savedTo: saveTo == "Files" ? "Saved to Files" : "Saved to Photos", file: nil)
            } catch is CancellationError {
                model.export = nil
            } catch {
                model.export = .failed(done: done, total: photos.count)
                model.show(describe(error))
            }
            exportTask = nil
        }
    }

    /// The gallery's picture of the work: a frame of what was exported.
    private func updatePicture(from file: URL) {
        let dest = folder.appendingPathComponent("picture.jpg")
        Task.detached {
            guard let v = try? await VideoSource.open(file), let pb = try? await v.frame(at: v.duration / 3),
                  let image = Pictures.image(pb, rotation: v.rotation, maxSide: 480) else { return }
            try? image.jpegData(compressionQuality: 0.8)?.write(to: dest, options: .atomic)
        }
    }

    // MARK: the app in front or not

    func setActive(_ on: Bool) {
        active = on
        if !on, exportTask != nil, backgroundTask == .invalid {
            // a moment to finish the frame on the way
            backgroundTask = UIApplication.shared.beginBackgroundTask { [weak self] in
                Task { @MainActor in self?.endBackground() }
            }
        }
        if on { endBackground() }
        if case .paused(let d, let t) = model.export, on { model.export = .running(done: d, total: t, left: nil) }
    }

    private func endBackground() {
        if backgroundTask != .invalid {
            UIApplication.shared.endBackgroundTask(backgroundTask)
            backgroundTask = .invalid
        }
    }

    // MARK: closing

    func close() {
        sound?.pause()
        sound = nil
        pictureTask?.cancel()
        frameTask?.cancel()
        madeTask?.cancel()
        playTask?.cancel()
        exportTask?.cancel()
        if let scanner { Task { await scanner.stop() } }
        saveFaces()
    }

    func saveFaces() {
        guard let scan else { return }
        let url = folder.appendingPathComponent("faces.json")
        Task.detached { if let data = try? scan.save() { try? data.write(to: url, options: .atomic) } }
    }
}

/// Frame `i` from its patch, upright for the screen; nil when the work has none for this plan.
private func patched(_ pb: CVPixelBuffer, index i: Int, plan: [PlannedFace], rotation: Int, work: String) -> UIImage? {
    let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
    guard let patch = Patches.get(work: work, key: Patches.key(frame: i, width: w, height: h, rotation: rotation, plan: plan)),
          let out = makePixelBuffer(width: w, height: h) else { return nil }
    // (its colours read as the frame's do, or a swapped frame looks different from a plain one)
    CVBufferPropagateAttachments(pb, out)
    return Patches.apply(patch, frame: pb, into: out) ? Pictures.image(out, rotation: rotation) : nil
}

/// Frame `i` with its faces swapped as planned, upright for the screen: from its patch when the
/// work has one, else swapped and its patch kept.
private func swapped(_ pb: CVPixelBuffer, index i: Int, plan: [PlannedFace], swapper: FaceSwapper, rotation: Int, work: String) -> UIImage? {
    let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
    guard let out = makePixelBuffer(width: w, height: h) else { return nil }
    CVBufferPropagateAttachments(pb, out)
    let key = Patches.key(frame: i, width: w, height: h, rotation: rotation, plan: plan)
    let kept = Patches.get(work: work, key: key)
    let applied = kept.map { Patches.apply($0, frame: pb, into: out) } ?? false
    if !applied {
        do {
            try withPixels(pb) { a, w, h, s in
                try withPixels(out, readOnly: false) { o, _, _, os in
                    _ = try swapper.swapPlanned(input: a, output: o, width: w, height: h, inputStride: s, outputStride: os, rotation: UInt32(rotation), plan: plan)
                }
            }
        } catch {
            return nil
        }
        if let patch = Patches.diff(pb, out) { Patches.put(work: work, key: key, patch) }
    }
    return Pictures.image(out, rotation: rotation)
}
