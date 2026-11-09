import FaceJutsuEngine
import Observation
import SwiftUI

// The editor's state, as the desktop's (web/app/main.js): each person found in a video is a track;
// their time is cut into parts, each with its own character and settings; what's drawn of a part is
// where the person is on screen in it. Tools apply to the selected parts, or to everyone's when
// nothing is selected. Edits go through `edit`, so undo and redo see them.

enum EditorMode: String, CaseIterable {
    case video, photo
}

enum Tool: String, CaseIterable, Identifiable {
    case characters, swap, enhance, mask
    var id: String { rawValue }
}

/// What a part shows as on the timeline: a stretch of it where the person is on screen.
struct Piece: Identifiable {
    var id: String { "\(part.id)-\(t0)" }
    let part: Part
    let t0: Double
    let t1: Double
}

/// Who each face becomes, and how: a snapshot of the tracks for the work done off the main thread.
/// Its settings are part of it, so a cached frame swapped with others is told apart.
struct Planner: Sendable, Equatable {
    /// who an analysis person is (root) -> their track's parts, in order: (start, character,
    /// settings)
    var parts: [Int: [Pair]] = [:]
    /// who is left as they are when: stretches [t0, t1) by analysis person (root)
    var leftOut: [Int: [LeftOut]] = [:]

    struct Pair: Sendable, Equatable {
        let t0: Double
        let character: Int?
        let s: SegmentSettings
    }

    /// The part of `root`'s track at `t` seconds.
    private func part(_ root: Int, at t: Double) -> Pair? {
        if leftOut[root]?.contains(where: { t >= $0.t0 && t < $0.t1 }) == true { return nil }
        return parts[root]?.last { $0.t0 <= t + 1e-6 }
    }

    /// The character the face of `root` becomes at `t` seconds, if any.
    func character(_ root: Int, at t: Double) -> Int? {
        part(root, at: t)?.character ?? nil
    }

    /// The faces of a frame to swap, with whom and how.
    func plan(_ faces: [ScanFace], at t: Double) -> [PlannedFace] {
        faces.compactMap { f in
            guard let p = part(Int(f.person), at: t), let c = p.character else { return nil }
            return p.s.planned(f.face, identity: c, ownDetail: f.detail)
        }
    }

    var isEmpty: Bool { !parts.values.contains { $0.contains { $0.character != nil } } }
}

/// Where an export is at.
enum ExportState: Equatable {
    /// finding who is in the video, before anything is swapped
    case scouting(done: Int, total: Int)
    case running(done: Int, total: Int, left: Double?)
    case paused(done: Int, total: Int)
    case finished(facts: String, savedTo: String, file: URL?)
    case failed(done: Int, total: Int)
    case noRoom(need: Int, free: Int)
}

@MainActor
@Observable
final class EditorModel {
    var doc = Doc()
    var mode: EditorMode = .video

    // MARK: the work open (none: the gallery)
    var work: WorkFile?
    var gallery: [WorkSummary] = Store.gallery()
    /// bringing a video or photos in: how far (nil: not known)
    var importing: Double??
    /// stops what's being brought in
    @ObservationIgnored var stopImport: (() -> Void)?
    /// what the photo picker is open for
    var picking: Picking?

    enum Picking: Equatable { case video, photos, morePhotos, character }

    // MARK: the video
    var duration: Double = 1
    /// the picture's size as shown
    var videoSize = CGSize(width: 16, height: 9)
    var times: [Double] = []
    /// per frame: 0 not looked at, 1 its keyframe skimmed, 2 looked at
    var states: [UInt8] = []
    /// how much around the playhead has been looked at (nil: done, or not scanning)
    var scouting: Double?
    var clipName = ""
    /// the picture on the stage, and the frame it's of
    var stageImage: UIImage?
    var stageFrame = -1
    /// a swap of the paused frame is on the way
    var swapping = false
    /// per frame: already swapped as the tracks now say
    var made: [Bool] = []
    /// the video has sound, and it's wanted while playing
    var hasSound = false
    var soundOn = UserDefaults.standard.object(forKey: "sound") as? Bool ?? true {
        didSet { UserDefaults.standard.set(soundOn, forKey: "sound") }
    }
    /// small pictures across the clip, `stripEvery` seconds apart
    var thumbs: [UIImage] = []
    var stripEvery = 1.0
    /// each analysis person's face picture
    var faceThumbs: [Int: UIImage] = [:]
    /// each track's time on screen
    var segments: [Int: [ClosedRange<Double>]] = [:]
    /// the faces at the playhead, by track: where they are, as parts of the picture
    var faceBoxes: [Int: CGRect] = [:]
    /// where each track's face was last seen
    var lastBoxes: [Int: (t: Double, box: CGRect)] = [:]

    /// showing the original instead of the swap
    var before = false
    /// the track (or photo face) the picture is framed on
    var focus: Int?
    /// goes up each time an edit changes what a face on screen becomes (its original or a
    /// character), for the puff of smoke on those faces (`poofBoxes`) and the wink
    var poofTick = 0
    @ObservationIgnored private(set) var poofBoxes: [CGRect] = []
    /// which way the photos last moved
    var photoDirection: CGFloat = 0
    /// what the last edit was
    private(set) var lastEditKey = ""
    var playhead: Double = 0 { didSet { if playhead != oldValue { playheadMoved() } } }
    var playing = false
    var export: ExportState?

    // MARK: the photos
    var photoIndex = 0
    var photoImages: [Int: UIImage] = [:]
    var photoAfter: [Int: UIImage] = [:]
    /// the selected faces of the photo, by id
    var pickedFaces: Set<Int> = [] { didSet { keepFocus() } }

    // MARK: selection and tools
    /// the selected parts, by id
    var picked: Set<Int> = [] {
        didSet {
            if picked != oldValue { settingsRow = false }
            keepFocus()
        }
    }
    /// with a selection, the row shows the tools that set it up (Back from the clip tools) until
    /// the selection changes
    var settingsRow = false
    /// the tracks selected as a whole, by id
    var wholeTracks: Set<Int> = []
    var tool: Tool?
    /// the stage's zoom, 1 = fitted
    var zoom: CGFloat = 1
    /// a short message over the editor
    var toast: String?

    @ObservationIgnored var past: [Doc] = []
    @ObservationIgnored var future: [Doc] = []
    @ObservationIgnored private var lastEdit: (key: String, at: Date)?
    @ObservationIgnored var nextId = 1000
    /// the work going on in the background for the open work
    @ObservationIgnored var live: LiveWork?
    @ObservationIgnored var saveTask: Task<Void, Never>?

    var canUndo: Bool { !past.isEmpty }
    var canRedo: Bool { !future.isEmpty }

    init() {
        doc.characters = Store.loadCharacters()
        Store.tidyCharacters()
        nextId = max(nextId, (doc.characters.map(\.id).max() ?? 0) + 1)
    }

    // MARK: history

    /// Changes the document as one undo step. Changes with the same key within a moment (a slider's
    /// drag) make one step.
    func edit(_ key: String, _ change: (inout Doc) -> Void) {
        var next = doc
        change(&next)
        guard next != doc else { return }
        let now = Date()
        if let last = lastEdit, last.key == key, key.hasPrefix("drag "), now.timeIntervalSince(last.at) < 0.8 {
            // the same drag: no new step
        } else {
            past.append(doc)
            if past.count > 100 { past.removeFirst() }
        }
        future.removeAll()
        lastEdit = (key, now)
        lastEditKey = key
        setDoc(next)
    }

    func undo() {
        guard let prev = past.popLast() else { return }
        future.append(doc)
        lastEdit = nil
        lastEditKey = ""
        setDoc(prev)
        dropStaleSelection()
    }

    func redo() {
        guard let next = future.popLast() else { return }
        past.append(doc)
        lastEdit = nil
        lastEditKey = ""
        setDoc(next)
        dropStaleSelection()
    }

    /// The document changed: what depends on it follows.
    private func setDoc(_ next: Doc) {
        let charactersChanged = next.characters != doc.characters || next.characters.map(\.id) != doc.characters.map(\.id)
        let was = facesOnScreen(doc)
        doc = next
        let now = facesOnScreen(doc)
        let changed = Set(was.keys).union(now.keys).filter { was[$0]?.character != now[$0]?.character }
        if !changed.isEmpty {
            poofBoxes = changed.compactMap { now[$0]?.box ?? was[$0]?.box }
            poofTick &+= 1
        }
        if charactersChanged {
            Store.saveCharacters(doc.characters)
            live?.syncIdentities(doc.characters)
        }
        workChanged()
    }

    /// The faces on screen now (by track, or by photo face), where they are and the character each
    /// becomes in `d` (nil: left as it is).
    private func facesOnScreen(_ d: Doc) -> [Int: (box: CGRect, character: Int?)] {
        let known = Set(d.characters.map(\.id))
        let given = { (s: SegmentSettings) in s.character.flatMap { known.contains($0) ? $0 : nil } }
        if mode == .photo {
            guard d.photos.indices.contains(photoIndex) else { return [:] }
            return Dictionary(d.photos[photoIndex].faces.map { ($0.id, (box(face: $0), given($0.s))) }, uniquingKeysWith: { a, _ in a })
        }
        var out: [Int: (box: CGRect, character: Int?)] = [:]
        for p in d.people {
            guard let b = faceBoxes[p.id] else { continue }
            let part = p.parts.last { $0.t0 <= playhead + 1e-6 } ?? p.parts.first
            out[p.id] = (b, part.flatMap { $0.off ? nil : given($0.s) })
        }
        return out
    }

    private func dropStaleSelection() {
        let ids = Set(doc.people.flatMap { $0.parts.filter { !$0.off }.map(\.id) })
        picked = picked.intersection(ids)
        wholeTracks = wholeTracks.intersection(Set(doc.people.map(\.id)))
    }

    func newId() -> Int {
        nextId += 1
        return nextId
    }

    // MARK: tracks and parts

    /// What's drawn of a track: its parts where the person is on screen.
    func pieces(_ p: Person) -> [Piece] {
        let segs = segments[p.id] ?? []
        var out: [Piece] = []
        for (i, part) in p.parts.enumerated() {
            let end = i + 1 < p.parts.count ? p.parts[i + 1].t0 : .infinity
            for s in segs where s.upperBound > part.t0 && s.lowerBound < end {
                out.append(Piece(part: part, t0: max(s.lowerBound, part.t0), t1: min(s.upperBound, end)))
            }
        }
        return out
    }

    /// The part of a track at `t`.
    func part(of p: Person, at t: Double) -> Part? {
        p.parts.last { $0.t0 <= t + 1e-6 } ?? p.parts.first
    }

    // MARK: selection

    var hasSelection: Bool { mode == .video ? !picked.isEmpty : !pickedFaces.isEmpty }

    func clearSelection() {
        picked = []
        wholeTracks = []
        pickedFaces = []
    }

    /// Adds a part to the selection, or takes it out.
    func toggle(part: Int) {
        if picked.contains(part) { picked.remove(part) } else { picked.insert(part) }
        for p in doc.people where !p.parts.filter({ !$0.off }).allSatisfy({ picked.contains($0.id) }) { wholeTracks.remove(p.id) }
        tool = nil
    }

    /// Selects a whole track, or takes it out again.
    func toggle(person id: Int) {
        guard let p = doc.people.first(where: { $0.id == id }) else { return }
        let ids = p.parts.filter { !$0.off }.map(\.id)
        if wholeTracks.contains(id) {
            wholeTracks.remove(id)
            picked.subtract(ids)
        } else {
            wholeTracks.insert(id)
            picked.formUnion(ids)
        }
        tool = nil
    }

    /// Selects every part, or none when they all are.
    func toggleAll() {
        let all = Set(doc.people.flatMap { $0.parts.filter { !$0.off }.map(\.id) })
        if !all.isEmpty && picked == all { clearSelection() } else {
            picked = all
            wholeTracks = Set(doc.people.map(\.id))
        }
        tool = nil
    }

    func toggle(face id: Int) {
        if pickedFaces.contains(id) { pickedFaces.remove(id) } else { pickedFaces.insert(id) }
    }

    /// Gives a person a new name; an empty one keeps the old.
    func rename(person id: Int, to name: String) {
        let clean = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clean.isEmpty else { return }
        edit("renaming") { d in
            guard let i = d.people.firstIndex(where: { $0.id == id }) else { return }
            d.people[i].label = clean
        }
    }

    // MARK: the scope: what the tools change

    /// The parts the tools change: the selected ones, or everyone's.
    private var scopeParts: Set<Int> {
        picked.isEmpty ? Set(doc.people.flatMap { $0.parts.filter { !$0.off }.map(\.id) }) : picked
    }

    private var currentPhoto: Photo? { doc.photos.indices.contains(photoIndex) ? doc.photos[photoIndex] : nil }

    private var scopeFaces: Set<Int> {
        guard let photo = currentPhoto else { return [] }
        return pickedFaces.isEmpty ? Set(photo.faces.map(\.id)) : pickedFaces
    }

    /// The settings of everything in the scope: the first one's.
    private var scopeList: [SegmentSettings] {
        if mode == .video {
            let scope = scopeParts
            return doc.people.flatMap { $0.parts.filter { scope.contains($0.id) }.map(\.s) }
        }
        let scope = scopeFaces
        return currentPhoto?.faces.filter { scope.contains($0.id) }.map(\.s) ?? []
    }

    var scopeSettings: SegmentSettings { scopeList.first ?? (mode == .video ? doc.fresh : SegmentSettings()) }

    /// The character the scope has: nil when none or mixed, and whether it's mixed.
    var scopeCharacter: (id: Int?, mixed: Bool) {
        let ids = scopeList.map(\.character)
        if ids.isEmpty && mode == .video { return (doc.fresh.character, false) }
        if Set(ids.map { $0 ?? -1 }).count > 1 { return (nil, true) }
        return (ids.first ?? nil, false)
    }

    /// Changes the settings of everything in the scope.
    /// Each segment has its own settings. Applied to everything (nothing selected, or every
    /// track), it goes to all that's still to be found too; to a track, to what's still to be
    /// found of it; to single segments, only to them.
    private func changeScope(_ key: String, _ change: @escaping (inout SegmentSettings) -> Void) {
        if mode == .video {
            let scope = scopeParts, tracks = wholeTracks
            let everything = picked.isEmpty || (!tracks.isEmpty && Set(doc.people.map(\.id)) == tracks)
            edit(key) { d in
                if everything { change(&d.fresh) }
                for i in d.people.indices {
                    for j in d.people[i].parts.indices where scope.contains(d.people[i].parts[j].id) { change(&d.people[i].parts[j].s) }
                    if everything || tracks.contains(d.people[i].id) {
                        var next = d.people[i].undiscovered ?? d.fresh
                        change(&next)
                        d.people[i].undiscovered = next
                    }
                }
            }
        } else {
            let scope = scopeFaces, index = photoIndex
            edit(key) { d in
                guard d.photos.indices.contains(index) else { return }
                for j in d.photos[index].faces.indices where scope.contains(d.photos[index].faces[j].id) { change(&d.photos[index].faces[j].s) }
            }
        }
    }

    func character(_ id: Int?) -> SwapCharacter? { doc.characters.first { $0.id == id } }

    /// Gives a character to the scope; giving the one it already has takes it off.
    func give(_ id: Int) {
        let take = scopeCharacter.id == id && !scopeCharacter.mixed
        let name = character(id)?.name ?? ""
        changeScope(take ? "taking \(name) off" : "giving \(name)") { $0.character = take ? nil : id }
    }

    /// A binding to a setting of the scope, as one undo step per change (a slider's drag is one).
    func binding<V: Equatable>(_ path: WritableKeyPath<SegmentSettings, V>, drag: Bool = false) -> Binding<V> {
        Binding(get: { self.scopeSettings[keyPath: path] },
                set: { v in self.changeScope(drag ? "drag \(path)" : "setting") { $0[keyPath: path] = v } })
    }

    // MARK: the clip tools

    private func selectedPieces(_ p: Person) -> [Piece] { pieces(p).filter { !$0.part.off && picked.contains($0.part.id) } }

    /// a selected part the playhead is inside, on screen
    var canSplit: Bool {
        doc.people.contains { p in selectedPieces(p).contains { playhead > $0.t0 + 0.05 && playhead < $0.t1 - 0.05 } }
    }

    /// two selected parts of one track, one after the other, cut inside a stretch on screen (where
    /// the person comes back after being away stays a cut: no split could make it again)
    var canJoin: Bool {
        doc.people.contains { p in
            zip(p.parts, p.parts.dropFirst()).contains { picked.contains($0.id) && picked.contains($1.id) && !$0.off && !$1.off && !atGap(segments[p.id] ?? [], $1.t0) }
        }
    }

    /// two or more tracks selected
    var canMerge: Bool { doc.people.filter { p in p.parts.contains { picked.contains($0.id) } }.count >= 2 }

    /// Cuts each selected part at the playhead, where it's on screen. The parts after the cut are
    /// selected.
    func split() {
        let t = playhead
        var made: Set<Int> = []
        let sel = picked
        let cut = doc.people.filter { p in selectedPieces(p).contains { t > $0.t0 + 0.05 && t < $0.t1 - 0.05 } }.map(\.id)
        edit("split") { d in
            for i in d.people.indices where cut.contains(d.people[i].id) {
                guard let j = d.people[i].parts.lastIndex(where: { $0.t0 <= t }), sel.contains(d.people[i].parts[j].id) else { continue }
                let q = d.people[i].parts[j]
                let n = Part(id: newId(), t0: t, s: q.s, off: q.off)
                d.people[i].parts.insert(n, at: j + 1)
                made.insert(n.id)
            }
        }
        if made.isEmpty {
            show("Move the playhead inside a segment to split it.")
        } else {
            picked = made
            wholeTracks = []
        }
    }

    /// Joins each selected part to the selected one before it, which keeps its settings.
    func join() {
        let sel = picked
        var keep: Set<Int> = []
        edit("join") { d in
            for i in d.people.indices {
                var parts: [Part] = []
                let segs = segments[d.people[i].id] ?? []
                for q in d.people[i].parts {
                    if let last = parts.last, sel.contains(last.id), sel.contains(q.id), !last.off, !q.off, !atGap(segs, q.t0) { continue }
                    parts.append(q)
                }
                d.people[i].parts = parts
                keep.formUnion(parts.filter { sel.contains($0.id) }.map(\.id))
            }
        }
        picked = keep
    }

    /// Merges the selected tracks into the topmost, for a person found as several. Not for people
    /// in the same frame with neither deleted there: then the playhead goes to where they meet. A
    /// face deleted where the other track is on screen stays as it is.
    func merge() {
        let who = doc.people.filter { p in p.parts.contains { picked.contains($0.id) } }
        guard who.count >= 2, let top = who.first, let scan = live?.scan, let roots = live?.roots else { return }
        var owner: [Int: Person] = [:]
        for p in who {
            for id in p.ids { owner[roots[id] ?? id] = p }
        }
        func partAt(_ p: Person, _ t: Double) -> Part { part(of: p, at: t) ?? p.parts[0] }
        // frame by frame, the part each stretch has on its own track: a new one starts where
        // that changes, so every stretch keeps its settings
        var from: [(t0: Double, q: Part)] = []
        for (i, t) in times.enumerated() {
            let here = scan.faces(frame: UInt32(i)).compactMap { owner[Int($0.person)] }
            guard let first = here.first else { continue }
            var showing: [Person] = []
            for p in here where !partAt(p, t).off && !showing.contains(where: { $0.id == p.id }) { showing.append(p) }
            if showing.count > 1 {
                seek(t)
                show("\(showing[0].label) and \(showing[1].label) are both on screen at \(Self.timecode(t)), so they can't be one track.")
                return
            }
            let q = partAt(showing.first ?? first, t)
            if from.last?.q.id != q.id { from.append((t, q)) }
        }
        var parts: [Part] = []
        var used = Set<Int>()
        for f in from {
            // a part back after another track's stretch is a new one, with the same settings
            parts.append(Part(id: used.contains(f.q.id) ? newId() : f.q.id, t0: parts.isEmpty ? 0 : f.t0, s: f.q.s, off: f.q.off))
            used.insert(f.q.id)
        }
        if parts.isEmpty { parts = top.parts }
        var left = who.flatMap { $0.leftOut ?? [] }
        for p in who {
            for (j, q) in p.parts.enumerated() where q.off {
                left.append(LeftOut(ids: p.ids, t0: q.t0, t1: j + 1 < p.parts.count ? p.parts[j + 1].t0 : duration + 1))
            }
        }
        let ids = Set(who.map(\.id))
        var merged = top
        merged.ids = who.flatMap(\.ids)
        merged.parts = parts
        merged.leftOut = left.isEmpty ? nil : left
        edit("merging tracks") { d in
            d.people = d.people.compactMap { p in p.id == top.id ? merged : (ids.contains(p.id) ? nil : p) }
        }
        // (the merged track's stretches on screen)
        live?.refresh()
        // nothing stays selected: a tap adds to the selection, so what's next would include this
        clearSelection()
    }

    /// The parts of tracks `who`, found to be one person, as one track's: each stretch with the
    /// settings it had, and one with the same settings as the one before it part of that one. A
    /// character these tracks are given goes to the stretches of the ones never given one. nil when
    /// none of them is on screen yet. `frames`: the analysis people of each frame.
    func joinedParts(_ who: [Person], frames: [[UInt32]]) -> [Part]? {
        var owner: [Int: Person] = [:]
        for p in who {
            for id in p.ids { owner[id] = p }
        }
        let given = Set(who.flatMap(\.parts).filter { !$0.off }.compactMap(\.s.character))
        let character = given.count == 1 ? given.first : nil
        var from: [(t0: Double, q: Part, s: SegmentSettings)] = []
        for (i, ids) in frames.enumerated() where i < times.count {
            guard let p = ids.lazy.compactMap({ owner[Int($0)] }).first, let q = part(of: p, at: times[i]) else { continue }
            if from.last?.q.id == q.id { continue }
            var s = q.s
            if let character, p.parts.allSatisfy({ $0.s.character == nil }) { s.character = character }
            from.append((times[i], q, s))
        }
        var parts: [Part] = []
        var used = Set<Int>()
        for f in from {
            if let last = parts.last, last.s == f.s, last.off == f.q.off { continue }
            // a part back after another track's stretch is a new one, with the same settings
            parts.append(Part(id: used.contains(f.q.id) ? newId() : f.q.id, t0: parts.isEmpty ? 0 : f.t0, s: f.s, off: f.q.off))
            used.insert(f.q.id)
        }
        return parts.isEmpty ? nil : parts
    }

    /// Deletes what's selected: a whole track goes (and doesn't come back), of the others only the
    /// selected parts (their faces are left as they are).
    func delete() {
        let sel = picked, whole = wholeTracks
        edit("delete") { d in
            for p in d.people where whole.contains(p.id) { d.gone.formUnion(p.ids) }
            d.people.removeAll { whole.contains($0.id) }
            for i in d.people.indices {
                for j in d.people[i].parts.indices where sel.contains(d.people[i].parts[j].id) {
                    d.people[i].parts[j].off = true
                }
            }
        }
        clearSelection()
    }

    // MARK: who becomes whom

    /// The tracks as the background work reads them.
    var planner: Planner {
        var out = Planner()
        guard let roots = live?.roots else { return out }
        for p in doc.people {
            let pairs = p.parts.map { Planner.Pair(t0: $0.t0, character: $0.off ? nil : $0.s.character.flatMap { id in character(id) == nil ? nil : id }, s: ModelFiles.shared.usable($0.s)) }
            for id in p.ids { out.parts[roots[id] ?? id] = pairs }
            for l in p.leftOut ?? [] {
                for id in l.ids { out.leftOut[roots[id] ?? id, default: []].append(l) }
            }
        }
        return out
    }

    // MARK: the picture

    private var photo: Photo? { currentPhoto }

    var pictureAspect: CGFloat {
        if mode == .photo, let p = photo { return CGFloat(p.width) / CGFloat(max(p.height, 1)) }
        return videoSize.width / max(videoSize.height, 1)
    }

    var photoStageImage: UIImage? {
        guard let p = photo else { return nil }
        return before ? photoImages[p.id] : (photoAfter[p.id] ?? photoImages[p.id])
    }

    /// Cuts straight between the original and the swap.
    func toggleBefore() {
        before.toggle()
        live?.refreshPicture()
    }

    func selectPhoto(_ index: Int) {
        guard index != photoIndex, doc.photos.indices.contains(index) else { return }
        photoDirection = index > photoIndex ? 1 : -1
        photoIndex = index
        pickedFaces = []
        focus = nil
    }

    // MARK: boxes and the crop

    private func box(face: PhotoFace) -> CGRect {
        CGRect(x: face.box[0], y: face.box[1], width: face.box[2], height: face.box[3])
    }

    private var selectedOnScreen: [Int] {
        if mode == .photo { return Array(pickedFaces) }
        return doc.people.filter { p in faceBoxes[p.id] != nil && p.parts.contains { picked.contains($0.id) } }.map(\.id)
    }

    struct FaceBox: Identifiable {
        let id: Int
        let rect: CGRect
        let color: Int
    }

    /// The boxes on the picture: whatever's selected and on screen.
    var visibleBoxes: [FaceBox] {
        if mode == .photo {
            return (photo?.faces ?? []).filter { pickedFaces.contains($0.id) }.map { FaceBox(id: $0.id, rect: box(face: $0), color: $0.color) }
        }
        return selectedOnScreen.compactMap { id in
            guard let p = doc.people.first(where: { $0.id == id }), let b = faceBoxes[id] else { return nil }
            return FaceBox(id: id, rect: b, color: p.color)
        }
    }

    struct Camera: Equatable {
        var x: CGFloat = 0.5, y: CGFloat = 0.5, z: CGFloat = 1
    }

    /// Where the crop looks: while they're on screen, at them; while they're gone, where they last
    /// were.
    private var heldBox: CGRect? {
        guard let focus else { return nil }
        if mode == .photo { return photo?.faces.first { $0.id == focus }.map { box(face: $0) } }
        return faceBoxes[focus] ?? lastBoxes[focus]?.box
    }

    /// The crop's framing, for a picture fitted to `fit` on a stage of `stage`. The picture never
    /// leaves a gap when it's big enough to fill the stage.
    func camera(fit: CGSize, stage: CGSize) -> Camera {
        guard let b = heldBox else { return Camera() }
        let z = min(max(min(stage.width / (b.width * fit.width * 1.12), stage.height / (b.height * fit.height * 1.3)), 1.3), 5)
        let hx = fit.width * z > stage.width ? stage.width / (2 * fit.width * z) : 0.5
        let hy = fit.height * z > stage.height ? stage.height / (2 * fit.height * z) : 0.5
        return Camera(x: min(max(b.midX, hx), 1 - hx), y: min(max(b.midY, hy), 1 - hy), z: z)
    }

    func enterFocus(_ id: Int) { focus = id }
    func exitFocus() { focus = nil }

    var focusAway: Bool {
        guard let focus, mode == .video else { return false }
        return faceBoxes[focus] == nil
    }

    /// when the person the crop is on comes back
    var nextBack: Double? {
        guard let focus, let p = doc.people.first(where: { $0.id == focus }) else { return nil }
        return pieces(p).filter { !$0.part.off && $0.t0 > playhead }.map(\.t0).min()
    }

    var focusName: String {
        if mode == .photo { return photo?.faces.first { $0.id == focus }?.label ?? "" }
        return doc.people.first { $0.id == focus }?.label ?? ""
    }

    /// A track's face picture.
    func picture(_ p: Person) -> UIImage? {
        p.ids.lazy.compactMap { self.faceThumbs[$0] }.first
    }

    /// The face on the crop's chip.
    var focusCrop: (image: UIImage?, box: CGRect) {
        if mode == .photo, let face = photo?.faces.first(where: { $0.id == focus }) { return faceCrop(face) }
        let p = doc.people.first { $0.id == focus }
        return (p.flatMap { picture($0) }, CGRect(x: 0, y: 0, width: 1, height: 1))
    }

    func faceCrop(_ face: PhotoFace) -> (image: UIImage?, box: CGRect) {
        guard let p = photo else { return (nil, CGRect(x: 0, y: 0, width: 1, height: 1)) }
        // a little wider than the face, square in the picture
        let b = box(face: face), a = pictureAspect
        let side = max(b.width, b.height / a) * 1.3
        let w = side, h = side * a
        return (photoImages[p.id], CGRect(x: b.midX - w / 2, y: b.midY - h / 2, width: w, height: h))
    }

    /// The crop goes with the selection: to whoever else is selected, or away.
    private func keepFocus() {
        guard let f = focus else { return }
        let keep = mode == .photo ? pickedFaces.contains(f) : doc.people.contains { $0.id == f && $0.parts.contains { picked.contains($0.id) } }
        if !keep { focus = selectedOnScreen.first }
    }

    /// Clears the selection, in video or photos, and goes back to the whole picture.
    func deselect() {
        tool = nil
        clearSelection()
        focus = nil
    }

    static var modelsMegabytes: Int { Store.megabytes(Engine.modelsFolder) }

    // MARK: messages

    func show(_ message: String) {
        withAnimation(Motion.ease(Motion.quick)) { toast = message }
        let shown = message
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(3.5))
            if toast == shown { withAnimation(Motion.ease(Motion.quick)) { toast = nil } }
        }
    }

    nonisolated static func timecode(_ t: Double, hundredths: Bool = false) -> String {
        let m = Int(t) / 60, s = t - Double(m * 60)
        return hundredths ? String(format: "%d:%05.2f", m, s) : String(format: "%d:%02d", m, Int(s))
    }
}
