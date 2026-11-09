import FaceJutsuEngine
import Foundation
import UIKit

// What the app keeps, as the desktop keeps it in data/: the characters, and each piece of work (a
// video or a set of photos, its edits and the faces found in it), in the app's own folder. Finished
// videos and photos go to Photos, or to Documents (Files: On My iPhone > FaceJutsu).

/// What a stretch of a person's time (or a face in a photo) gets: the character and the tools'
/// settings.
struct SegmentSettings: Codable, Equatable {
    /// the character it becomes (nil: left as it is)
    var character: Int?
    var detail = "Fast"
    var quality = "Off"
    var style = "Subtle"
    var strength = 0.6
    var edges = 0.2
    var objects = false
    var coverage = 0.0

    /// A face to swap with these settings, as the engine takes it.
    func planned(_ face: FoundFace, identity: Int, ownDetail: Double? = nil) -> PlannedFace {
        let detail: UInt32 = switch detail { case "Fast": 1; case "Max": 3; default: 2 }
        let enhance: UInt32 = switch quality { case "Off": 0; case "Best": 512; default: 256 }
        return PlannedFace(face: face, identity: UInt32(identity), detail: detail, enhance: enhance, subtle: style != "Strong",
                           strength: Float(strength), edges: Float(edges), objects: objects, coverage: Float(coverage),
                           ownDetail: quality != "Off" && style != "Strong" ? ownDetail : nil)
    }
}

/// A stretch of a person's time, from `t0` to the next part's (or the end): what happens to their
/// face then. As on the desktop, a person's parts cover the whole video; what's drawn is where they
/// are on screen.
struct Part: Identifiable, Equatable, Codable {
    let id: Int
    var t0: Double
    var s: SegmentSettings
    /// deleted: the face is left as it is, and the stretch shows hatched
    var off = false
    /// made where the person came back on screen, rather than by a split
    var auto = false
}

/// Someone found in the video: a track. `ids` are the people of the analysis it's made of.
struct Person: Identifiable, Equatable, Codable {
    let id: Int
    var ids: [Int]
    var label: String
    var color: Int
    var parts: [Part]
    /// from tracks merged into this one: stretches their faces were deleted in, left as they are
    /// whatever the merged track does then
    var leftOut: [LeftOut]? = nil
    /// what its segments found from here on get: set by applying to the track, or to everything
    var undiscovered: SegmentSettings? = nil
}

/// A stretch [t0, t1) in which the faces of analysis people `ids` are left as they are.
struct LeftOut: Equatable, Codable, Sendable {
    let ids: [Int]
    let t0: Double, t1: Double
}

/// A face found in a photo, with what it gets.
struct PhotoFace: Identifiable, Equatable, Codable {
    let id: Int
    var label: String
    var color: Int
    /// where it is, as parts of the upright photo: x, y, width, height
    var box: [Double]
    /// its box and keypoints in the upright photo's pixels, for the swap
    var bbox: [Double]
    var kps: [Double]
    var score: Double
    var s: SegmentSettings
}

struct Photo: Identifiable, Equatable, Codable {
    let id: Int
    var file: String
    var width: Int
    var height: Int
    var faces: [PhotoFace]
    /// its faces have been looked for
    var found = false
}

/// A face to swap in, picked by the user.
struct SwapCharacter: Identifiable, Equatable {
    let id: Int
    var name: String
    var image: UIImage?
    var embedding: [Float]

    static func == (a: SwapCharacter, b: SwapCharacter) -> Bool { a.id == b.id && a.name == b.name }
}

/// Everything undo and redo cover.
struct Doc: Equatable {
    var characters: [SwapCharacter] = []
    var people: [Person] = []
    var photos: [Photo] = []
    /// people of the analysis whose tracks were deleted: they don't come back
    var gone: Set<Int> = []
    /// what tracks found from here on get: set by applying to everything
    var fresh = SegmentSettings()
}

/// A piece of work as saved.
struct WorkFile: Codable {
    var id: String
    var kind: String
    /// the video's own name, as it came
    var name: String = ""
    var created: Date
    var updated: Date
    /// the video's file in the work's folder
    var video: String?
    var duration: Double = 0
    /// the picture's size as shown
    var width: Int = 0
    var height: Int = 0
    var people: [Person] = []
    var photos: [Photo] = []
    var gone: [Int] = []
    var fresh: SegmentSettings? = nil
    var nextId: Int = 1000
    var playhead: Double = 0
}

/// A piece of work in the gallery.
struct WorkSummary: Identifiable, Equatable {
    let id: String
    let video: Bool
    let updated: Date
    let duration: Double
    let photos: Int
    let aspect: CGFloat
    let picture: URL

    var tag: String {
        video ? EditorModel.timecode(duration) : (photos == 1 ? "1 photo" : "\(photos) photos")
    }
}

private struct CharacterRecord: Codable {
    var id: Int
    var name: String
    var embedding: [Float]
}

enum Store {
    static let root: URL = {
        let url = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("FaceJutsu", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }()

    static var works: URL { folder("Work") }
    static var characterFolder: URL { folder("Characters") }
    /// what Save to Files writes: the app's Documents, which Files shows
    static var documents: URL { FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0] }

    private static func folder(_ name: String) -> URL {
        let url = root.appendingPathComponent(name, isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    static func workFolder(_ id: String) -> URL {
        let url = works.appendingPathComponent(id, isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.dateEncodingStrategy = .iso8601
        return e
    }()

    private static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }()

    // MARK: work

    static func save(_ work: WorkFile) {
        guard let data = try? encoder.encode(work) else { return }
        try? data.write(to: workFolder(work.id).appendingPathComponent("work.json"), options: .atomic)
    }

    static func load(_ id: String) -> WorkFile? {
        let url = works.appendingPathComponent(id).appendingPathComponent("work.json")
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? decoder.decode(WorkFile.self, from: data)
    }

    static func remove(_ id: String) {
        try? FileManager.default.removeItem(at: works.appendingPathComponent(id))
    }

    /// The saved work, newest first.
    static func gallery() -> [WorkSummary] {
        let ids = (try? FileManager.default.contentsOfDirectory(atPath: works.path)) ?? []
        return ids.compactMap { id -> WorkSummary? in
            guard let w = load(id) else { return nil }
            let aspect = CGFloat(max(w.width, 1)) / CGFloat(max(w.height, 1))
            return WorkSummary(id: id, video: w.kind == "video", updated: w.updated, duration: w.duration, photos: w.photos.count,
                               aspect: min(max(aspect, 0.5), 2.2), picture: workFolder(id).appendingPathComponent("picture.jpg"))
        }
        .sorted { $0.updated > $1.updated }
    }

    // MARK: characters

    static func loadCharacters() -> [SwapCharacter] {
        let url = characterFolder.appendingPathComponent("characters.json")
        guard let data = try? Data(contentsOf: url), let list = try? decoder.decode([CharacterRecord].self, from: data) else { return [] }
        return list.map { r in
            SwapCharacter(id: r.id, name: r.name, image: UIImage(contentsOfFile: characterFolder.appendingPathComponent("c\(r.id).jpg").path), embedding: r.embedding)
        }
    }

    /// Writes the list of characters; a new character's picture is written when it's made.
    static func saveCharacters(_ list: [SwapCharacter]) {
        let records = list.map { CharacterRecord(id: $0.id, name: $0.name, embedding: $0.embedding) }
        guard let data = try? encoder.encode(records) else { return }
        try? data.write(to: characterFolder.appendingPathComponent("characters.json"), options: .atomic)
        // pictures of characters no longer in the list (kept while undo can still bring them back
        // in this run, so only once the app starts again)
    }

    static func saveCharacterPicture(_ id: Int, _ image: UIImage) {
        try? image.jpegData(compressionQuality: 0.85)?.write(to: characterFolder.appendingPathComponent("c\(id).jpg"), options: .atomic)
    }

    /// Removes the pictures of characters that aren't in the list any more.
    static func tidyCharacters() {
        let keep = Set(loadCharacters().map { "c\($0.id).jpg" })
        let files = (try? FileManager.default.contentsOfDirectory(atPath: characterFolder.path)) ?? []
        for f in files where f.hasPrefix("c") && f.hasSuffix(".jpg") && !keep.contains(f) {
            try? FileManager.default.removeItem(at: characterFolder.appendingPathComponent(f))
        }
    }

    // MARK: sizes

    static func megabytes(_ url: URL) -> Int {
        guard let e = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var bytes = 0
        for case let f as URL in e { bytes += (try? f.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0 }
        return bytes / 1_000_000
    }

    /// what the phone has free for the app to write, in bytes
    static var freeBytes: Int64 {
        let v = try? root.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return v?.volumeAvailableCapacityForImportantUsage ?? .max
    }
}
