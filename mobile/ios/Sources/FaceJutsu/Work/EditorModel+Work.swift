import CoreTransferable
import FaceJutsuEngine
import Foundation
import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

/// A video from the photo picker, copied in for the app to keep.
struct PickedMovie: Transferable {
    let url: URL
    let name: String

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { SentTransferredFile($0.url) } importing: { received in
            let dir = FileManager.default.temporaryDirectory.appendingPathComponent("import", isDirectory: true)
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let ext = received.file.pathExtension.isEmpty ? "mov" : received.file.pathExtension
            let dest = dir.appendingPathComponent(UUID().uuidString + "." + ext)
            try FileManager.default.copyItem(at: received.file, to: dest)
            return PickedMovie(url: dest, name: received.file.lastPathComponent)
        }
    }
}

/// frames a face may go missing and stay on its track; frames a person must be seen in
private let gapFrames = 12, minFrames = 3
private let eps = 1e-3

/// Whether `t` is where a person comes back on screen after being away, in their stretches.
func atGap(_ segs: [ClosedRange<Double>], _ t: Double) -> Bool {
    segs.indices.contains { k in k > 0 && t > segs[k - 1].upperBound - eps && t <= segs[k].lowerBound + eps }
}

extension EditorModel {
    var frameIndex: Int { times.isEmpty ? 0 : nearestIndex(times, playhead) }

    // MARK: following the playhead and the edits

    func playheadMoved() {
        // (the export moves it, and brings its own picture)
        guard let live, !live.exporting else { return }
        live.follow()
        live.facesAtPlayhead()
        if !playing { live.refreshPicture() }
    }

    /// The tracks or settings changed: saved soon, and the picture follows.
    func workChanged() {
        scheduleSave()
        live?.refreshMade()
        if !playing { live?.refreshPicture() }
    }

    func togglePlay() {
        if playing { stop() } else { live?.play() }
    }

    func stop() {
        guard playing else { return }
        playing = false
        live?.stop()
    }

    func seek(_ t: Double) {
        playhead = min(max(t, 0), duration)
    }

    // MARK: bringing things in

    /// Brings in a video from the photo picker, showing how far while it comes (from iCloud).
    func importVideo(_ item: PhotosPickerItem) async {
        importing = .some(nil)
        var stopped = false
        let picked: PickedMovie? = await withCheckedContinuation { done in
            let progress = item.loadTransferable(type: PickedMovie.self) { result in
                done.resume(returning: try? result.get())
            }
            stopImport = {
                stopped = true
                progress.cancel()
            }
            Task { @MainActor in
                while !progress.isFinished && !progress.isCancelled {
                    if self.importing != nil { self.importing = .some(progress.fractionCompleted) }
                    try? await Task.sleep(for: .milliseconds(200))
                }
            }
        }
        stopImport = nil
        importing = nil
        if stopped {
            if let picked { try? FileManager.default.removeItem(at: picked.url) }
            return
        }
        guard let picked else {
            show("This video can't be opened on this phone. Try another.")
            return
        }
        await newVideo(picked.url, name: picked.name)
    }

    /// Brings in photos from the photo picker: a new piece of work, or more for the one open.
    func importPhotos(_ items: [PhotosPickerItem], adding: Bool) async {
        importing = .some(nil)
        var stopped = false
        stopImport = { stopped = true }
        var images: [UIImage] = []
        for (n, item) in items.enumerated() {
            if stopped { break }
            if let data = try? await item.loadTransferable(type: Data.self), let image = UIImage(data: data) { images.append(image) }
            importing = .some(Double(n + 1) / Double(items.count))
        }
        stopImport = nil
        importing = nil
        if stopped { return }
        guard !images.isEmpty else {
            show("These photos can't be opened. Try others.")
            return
        }
        if adding, work?.kind == "photos" { await addPhotos(images) } else { await newPhotos(images) }
    }

    private func newVideo(_ url: URL, name: String) async {
        let id = UUID().uuidString
        let ext = url.pathExtension.isEmpty ? "mov" : url.pathExtension
        let dest = Store.workFolder(id).appendingPathComponent("video.\(ext)")
        do { try FileManager.default.moveItem(at: url, to: dest) } catch {
            Store.remove(id)
            show("This video can't be opened on this phone. Try another.")
            return
        }
        guard let source = try? await VideoSource.open(dest) else {
            Store.remove(id)
            show("This video can't be opened on this phone. Try another.")
            return
        }
        let now = Date()
        var w = WorkFile(id: id, kind: "video", name: name, created: now, updated: now, video: dest.lastPathComponent)
        w.duration = source.duration
        w.width = Int(source.shown.width)
        w.height = Int(source.shown.height)
        Store.save(w)
        await openVideo(w, source)
    }

    private func newPhotos(_ images: [UIImage]) async {
        let id = UUID().uuidString
        let now = Date()
        let w = WorkFile(id: id, kind: "photos", created: now, updated: now)
        Store.save(w)
        await openPhotos(w)
        await addPhotos(images)
    }

    /// Opens saved work.
    func open(_ id: String) async {
        guard let w = Store.load(id) else {
            show("This work can't be opened.")
            return
        }
        if w.kind == "video" {
            guard let file = w.video, let source = try? await VideoSource.open(Store.workFolder(id).appendingPathComponent(file)) else {
                show("This video can't be opened on this phone. Try another.")
                return
            }
            await openVideo(w, source)
        } else {
            await openPhotos(w)
        }
    }

    private func resetEditor(_ w: WorkFile) {
        live?.close()
        work = w
        doc.people = w.people
        doc.photos = w.photos
        doc.gone = Set(w.gone)
        doc.fresh = w.fresh ?? SegmentSettings()
        nextId = max(nextId, w.nextId)
        past = []
        future = []
        clearSelection()
        focus = nil
        tool = nil
        before = false
        export = nil
        zoom = 1
        stageImage = nil
        stageFrame = -1
        swapping = false
        thumbs = []
        segments = [:]
        faceBoxes = [:]
        lastBoxes = [:]
        states = []
        made = []
        times = []
        scouting = nil
        faceThumbs = [:]
        photoImages = [:]
        photoAfter = [:]
        photoIndex = 0
        live = LiveWork(model: self, id: w.id)
    }

    private func openVideo(_ w: WorkFile, _ source: VideoSource) async {
        resetEditor(w)
        mode = .video
        clipName = w.name.isEmpty ? "Video" : w.name
        duration = source.duration
        videoSize = source.shown
        hasSound = source.hasAudio
        let folder = Store.workFolder(w.id)
        for f in (try? FileManager.default.contentsOfDirectory(atPath: folder.path)) ?? [] where f.hasPrefix("p") && f.hasSuffix(".jpg") {
            if let n = Int(f.dropFirst().dropLast(4)), let image = UIImage(contentsOfFile: folder.appendingPathComponent(f).path) { faceThumbs[n] = image }
        }
        guard let live else { return }
        live.source = source
        guard let table = try? await Task.detached(priority: .userInitiated, operation: { try source.frameTable() }).value, live === self.live else {
            if live === self.live { show("This video can't be opened on this phone. Try another."); close() }
            return
        }
        times = table.times
        states = Array(repeating: 0, count: times.count)
        playhead = min(w.playhead, duration)
        live.refreshPicture()
        guard let swapper = await Engine.shared.ready(), live === self.live else {
            if live === self.live { show(Engine.shared.failed ?? "The models couldn't be loaded.") }
            return
        }
        live.syncIdentities(doc.characters)
        let saved = try? Data(contentsOf: folder.appendingPathComponent("faces.json"))
        var scan = saved.flatMap { try? VideoScan.restore(swapper: swapper, saved: $0) }
        if scan?.times().count != times.count { scan = nil }
        live.startScan(source: source, scan: scan ?? VideoScan(swapper: swapper, times: times, keys: table.keys.map { UInt32($0) }))
        // the gallery's picture, and the strip of frames
        let picture = folder.appendingPathComponent("picture.jpg")
        if !FileManager.default.fileExists(atPath: picture.path) {
            Task.detached {
                if let pb = try? await source.frame(at: source.duration / 3), let image = Pictures.image(pb, rotation: source.rotation, maxSide: 480) {
                    try? image.jpegData(compressionQuality: 0.8)?.write(to: picture, options: .atomic)
                }
            }
        }
        stripEvery = max(0.5, duration / 150)
        let strip = await source.strip(every: stripEvery, height: 80)
        if live === self.live { thumbs = strip }
    }

    private func openPhotos(_ w: WorkFile) async {
        resetEditor(w)
        mode = .photo
        let folder = Store.workFolder(w.id)
        for p in doc.photos { photoImages[p.id] = await Self.displayImage(folder.appendingPathComponent(p.file)) }
        if let live { live.syncIdentities(doc.characters) }
        await findPhotoFaces()
        live?.refreshPicture()
    }

    private static func displayImage(_ url: URL) async -> UIImage? {
        await Task.detached {
            guard let image = UIImage(contentsOfFile: url.path) else { return nil }
            return image.preparingThumbnail(of: CGSize(width: 2048, height: 2048 * image.size.height / max(image.size.width, 1))) ?? image
        }.value
    }

    /// Back to the gallery; the work is saved.
    func close() {
        if playing { stop() }
        save()
        live?.close()
        live = nil
        work = nil
        stageImage = nil
        gallery = Store.gallery()
    }

    func removeWork(_ id: String) {
        Store.remove(id)
        Patches.remove(work: id)
        gallery = Store.gallery()
    }

    func scheduleSave() {
        saveTask?.cancel()
        saveTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(1))
            if !Task.isCancelled { save() }
        }
    }

    func save() {
        guard var w = work else { return }
        w.updated = Date()
        w.people = doc.people
        w.photos = doc.photos
        w.gone = Array(doc.gone)
        w.fresh = doc.fresh
        w.nextId = nextId
        w.playhead = playhead
        work = w
        Store.save(w)
        live?.saveFaces()
    }

    /// Back to the tracks as found, with nothing given.
    func startOver() {
        if mode == .video {
            edit("starting over") { d in
                d.people = []
                d.gone = []
                d.fresh = SegmentSettings()
            }
            live?.refresh()
        } else {
            edit("starting over") { d in
                for i in d.photos.indices {
                    for j in d.photos[i].faces.indices { d.photos[i].faces[j].s = SegmentSettings() }
                }
            }
        }
        clearSelection()
    }

    // MARK: the scan's people as tracks

    /// Takes in the people the scan has found, as the desktop does (refreshTracks): a person found
    /// to be someone with a track goes on it; someone new, once seen in enough frames (or on a
    /// skimmed keyframe), gets a track of their own; each track's time on screen follows, and each
    /// time they come back on screen starts a part of its own.
    func applyScan(_ found: [ScanPerson]) {
        guard !times.isEmpty else { return }
        let root = Dictionary(found.map { (Int($0.person), Int($0.root)) }, uniquingKeysWith: { a, _ in a })
        let byId = Dictionary(found.map { (Int($0.person), $0) }, uniquingKeysWith: { a, _ in a })
        var d = doc
        // tracks found to be one person become the first of them, each stretch as it was
        var frames: [[UInt32]]?
        var k = 0
        while k < d.people.count {
            let roots = Set(d.people[k].ids.map { root[$0] ?? $0 })
            let same = d.people.indices.filter { j in j > k && d.people[j].ids.contains { roots.contains(root[$0] ?? $0) } }
            if same.isEmpty {
                k += 1
                continue
            }
            let who = [d.people[k]] + same.map { d.people[$0] }
            if frames == nil { frames = live?.scan?.framePeople() ?? [] }
            if let parts = joinedParts(who, frames: frames ?? []) { d.people[k].parts = parts }
            // a character given to one of them is the whole person's, later stretches included
            let given = Set(who.flatMap(\.parts).filter { !$0.off }.compactMap(\.s.character))
            if given.count == 1, let c = given.first, !who[0].parts.contains(where: { $0.s.character != nil }) {
                var next = d.people[k].undiscovered ?? d.fresh
                next.character = c
                d.people[k].undiscovered = next
            }
            d.people[k].ids = Array(Set(who.flatMap(\.ids))).sorted()
            let left = who.flatMap { $0.leftOut ?? [] }
            d.people[k].leftOut = left.isEmpty ? nil : left
            for j in same.reversed() { d.people.remove(at: j) }
            // (its people may be one with another track's too)
        }
        var trackOf: [Int: Int] = [:]
        for (t, p) in d.people.enumerated() {
            for id in p.ids { trackOf[root[id] ?? id] = t }
        }
        // the people found since
        var fresh: [Int: [Int]] = [:]
        for p in found {
            let id = Int(p.person), r = Int(p.root)
            if d.gone.contains(id) || d.gone.contains(r) { continue }
            if let t = trackOf[r] {
                if !d.people[t].ids.contains(id) { d.people[t].ids.append(id) }
            } else {
                fresh[r, default: []].append(id)
            }
        }
        func shown(_ ids: [Int]) -> Bool {
            ids.reduce(0) { $0 + Int(byId[$1]?.frames ?? 0) } >= minFrames || ids.contains { byId[$0]?.loose ?? false }
        }
        func at(_ i: Int) -> Double { i < times.count ? times[i] : duration }
        func segmentsOf(_ ids: [Int]) -> [ClosedRange<Double>] {
            var runs: [(Int, Int)] = []
            for id in ids {
                guard let r = byId[id]?.runs else { continue }
                for n in stride(from: 0, to: r.count - 1, by: 2) { runs.append((Int(r[n]), Int(r[n + 1]))) }
            }
            runs.sort { $0.0 < $1.0 }
            var out: [(Int, Int)] = []
            for (i, j) in runs {
                if let last = out.last, i <= last.1 + gapFrames { out[out.count - 1].1 = max(last.1, j) } else { out.append((i, j)) }
            }
            return out.map { at($0.0)...max(at($0.0), at($0.1)) }
        }
        var segs: [Int: [ClosedRange<Double>]] = [:]
        for i in d.people.indices {
            let s = segmentsOf(d.people[i].ids)
            segs[d.people[i].id] = s
            d.people[i].parts = cutAtGaps(d.people[i].parts, s, d.people[i].undiscovered ?? d.fresh)
        }
        let newcomers = fresh.filter { shown($0.value) }.sorted { (segmentsOf($0.value).first?.lowerBound ?? 0) < (segmentsOf($1.value).first?.lowerBound ?? 0) }
        var number = (d.people.compactMap { Int($0.label.replacingOccurrences(of: "Person ", with: "")) }.max() ?? 0)
        for (r, ids) in newcomers {
            number += 1
            let s = segmentsOf([r] + ids)
            let id = newId()
            segs[id] = s
            d.people.append(Person(id: id, ids: Array(Set([r] + ids)).sorted(), label: "Person \(number)", color: (number - 1) % Ink.faces.count,
                                   parts: cutAtGaps([Part(id: newId(), t0: 0, s: d.fresh)], s, d.fresh), undiscovered: d.fresh))
        }
        if segs != segments { segments = segs }
        if d != doc {
            doc = d
            scheduleSave()
        }
    }

    /// Cuts a track where its person comes back after being off screen, so each stretch on screen
    /// is a part of its own, with the settings it had. Such a cut goes again once the stretches
    /// meet, when its settings are still the ones before it.
    private func cutAtGaps(_ list: [Part], _ segs: [ClosedRange<Double>], _ found: SegmentSettings) -> [Part] {
        var parts: [Part] = []
        for q in list {
            if q.auto, let prev = parts.last, !atGap(segs, q.t0), q.s == prev.s, q.off == prev.off { continue }
            parts.append(q)
        }
        for k in segs.indices where k > 0 {
            let a = segs[k].lowerBound, b = segs[k - 1].upperBound
            if parts.indices.contains(where: { $0 > 0 && parts[$0].t0 > b - eps && parts[$0].t0 <= a + eps }) { continue }
            var at = 0
            for (i, q) in parts.enumerated() where q.t0 <= a + eps { at = i }
            let q = parts[at]
            parts.insert(Part(id: newId(), t0: a, s: found, off: q.off, auto: true), at: at + 1)
        }
        return parts
    }

    // MARK: characters

    /// Makes a character of the largest face in a picture.
    func addCharacter(_ image: UIImage) async {
        guard let cg = uprightImage(image), let pb = pixelBuffer(from: cg) else {
            show("That picture can't be opened. Try another.")
            return
        }
        guard let swapper = await Engine.shared.ready() else {
            show(Engine.shared.failed ?? "The models couldn't be loaded.")
            return
        }
        let found: SourceFace? = await Task.detached(priority: .userInitiated) {
            CVPixelBufferLockBaseAddress(pb, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
            guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
            let stride = CVPixelBufferGetBytesPerRow(pb), h = CVPixelBufferGetHeight(pb), w = CVPixelBufferGetWidth(pb)
            let data = Data(bytes: base, count: stride * h)
            return try? swapper.sourceFace(pixels: data, width: UInt32(w), height: UInt32(h), stride: UInt32(stride), bgra: true)
        }.value
        guard let found else {
            show("No face in that picture. Try another.")
            return
        }
        let id = newId()
        let portrait = Pictures.portrait(cg, box: found.face.bbox) ?? image
        Store.saveCharacterPicture(id, portrait)
        try? swapper.setIdentity(id: UInt32(id), embedding: found.embedding)
        let number = (doc.characters.compactMap { Int($0.name.replacingOccurrences(of: "Character ", with: "")) }.max() ?? 0) + 1
        edit("adding a character") { d in
            d.characters.append(SwapCharacter(id: id, name: "Character \(number)", image: portrait, embedding: found.embedding))
        }
    }

    func removeCharacter(_ id: Int) {
        let name = character(id)?.name ?? ""
        edit("removing \(name)") { d in d.characters.removeAll { $0.id == id } }
    }

    func renameCharacter(_ id: Int, to name: String) {
        let clean = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clean.isEmpty else { return }
        edit("renaming a character") { d in
            guard let i = d.characters.firstIndex(where: { $0.id == id }) else { return }
            d.characters[i].name = String(clean.prefix(24))
        }
    }

    // MARK: photos

    private func addPhotos(_ images: [UIImage]) async {
        guard let w = work else { return }
        let folder = Store.workFolder(w.id)
        var added: [Photo] = []
        for image in images {
            let id = newId()
            let file = "photo-\(id).jpg"
            let url = folder.appendingPathComponent(file)
            let size: CGSize? = await Task.detached {
                guard let cg = uprightImage(image), let data = UIImage(cgImage: cg).jpegData(compressionQuality: 0.95) else { return nil }
                try? data.write(to: url, options: .atomic)
                return CGSize(width: cg.width, height: cg.height)
            }.value
            guard let size else { continue }
            added.append(Photo(id: id, file: file, width: Int(size.width), height: Int(size.height), faces: []))
            photoImages[id] = await Self.displayImage(url)
        }
        guard !added.isEmpty else { return }
        doc.photos.append(contentsOf: added)
        if !FileManager.default.fileExists(atPath: folder.appendingPathComponent("picture.jpg").path), let first = photoImages[added[0].id],
           let small = first.preparingThumbnail(of: CGSize(width: 480, height: 480 * first.size.height / max(first.size.width, 1))) {
            try? small.jpegData(compressionQuality: 0.8)?.write(to: folder.appendingPathComponent("picture.jpg"), options: .atomic)
        }
        save()
        await findPhotoFaces()
    }

    /// Finds the faces of the photos not looked at yet.
    private func findPhotoFaces() async {
        guard let w = work, let swapper = await Engine.shared.ready() else { return }
        let folder = Store.workFolder(w.id)
        for p in doc.photos where !p.found {
            let url = folder.appendingPathComponent(p.file)
            let faces: [FoundFace]? = await Task.detached(priority: .userInitiated) {
                guard let image = UIImage(contentsOfFile: url.path), let cg = image.cgImage, let pb = pixelBuffer(from: cg) else { return nil }
                return try? withPixels(pb) { a, w, h, s in try swapper.detect(input: a, width: w, height: h, stride: s, rotation: 0) }
            }.value
            guard work?.id == w.id, let i = doc.photos.firstIndex(where: { $0.id == p.id }) else { return }
            let W = Double(p.width), H = Double(p.height)
            doc.photos[i].faces = (faces ?? []).filter { $0.bbox.count == 4 && $0.kps.count == 10 }.enumerated().map { n, f in
                PhotoFace(id: newId(), label: "Face \(n + 1)", color: n % Ink.faces.count,
                          box: [f.bbox[0] / W, f.bbox[1] / H, (f.bbox[2] - f.bbox[0]) / W, (f.bbox[3] - f.bbox[1]) / H],
                          bbox: f.bbox, kps: f.kps, score: f.score, s: SegmentSettings())
            }
            doc.photos[i].found = true
            if photoIndex == i && (faces ?? []).isEmpty { show("No faces found in this photo.") }
        }
        save()
    }

    /// Removes the photo showing; the last one closes the work.
    func removePhoto() {
        guard doc.photos.indices.contains(photoIndex) else { return }
        let index = photoIndex
        edit("removing a photo") { d in d.photos.remove(at: index) }
        photoIndex = min(photoIndex, max(0, doc.photos.count - 1))
        pickedFaces = []
        focus = nil
        if doc.photos.isEmpty {
            if let id = work?.id {
                close()
                removeWork(id)
            }
        }
    }

    /// A photo's faces to swap, and with whom.
    func photoPlan(_ p: Photo) -> [PlannedFace] {
        p.faces.compactMap { f in
            guard let c = f.s.character, character(c) != nil else { return nil }
            return ModelFiles.shared.usable(f.s).planned(FoundFace(bbox: f.bbox, score: f.score, kps: f.kps), identity: c)
        }
    }

    // MARK: export

    var exportRunning: Bool {
        switch export {
        case .scouting, .running, .paused: true
        default: false
        }
    }

    var canExport: Bool {
        mode == .video ? !planner.isEmpty : doc.photos.contains { !photoPlan($0).isEmpty }
    }

    func startExport() {
        guard !exportRunning else { return }
        withAnimation(Motion.ease(Motion.move)) {
            deselect()
            zoom = 1
        }
        live?.startExport()
    }

    func stopExport() { live?.stopExport() }

    func closeExport() {
        withAnimation(Motion.ease(Motion.quick)) { export = nil }
    }
}
