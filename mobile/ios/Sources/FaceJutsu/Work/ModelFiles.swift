import Foundation
import UIKit

/// A model the app can use.
struct ModelFile: Identifiable, Sendable {
    let name: String
    let title: String
    let optional: Bool
    var id: String { name }
}

/// Where a model stands on this phone.
enum ModelState: Equatable {
    case missing, waiting, ready, failed
    case getting(Double)
}

/// The models on this phone, and getting them: downloaded from the addresses in the settings file
/// the desktop ships, or copied in from Files. A file by a model's name is always a whole one.
@MainActor
@Observable
final class ModelFiles {
    static let shared = ModelFiles()

    nonisolated static let files = [
        ModelFile(name: "inswapper-128.int8.safetensors", title: "inswapper_128", optional: false),
        ModelFile(name: "arcface-r50.int8.safetensors", title: "ArcFace R50", optional: false),
        ModelFile(name: "scrfd-10g.bf16.safetensors", title: "SCRFD-10G", optional: false),
        ModelFile(name: "gpen-bfr-256.int8.safetensors", title: "GPEN 256", optional: true),
        ModelFile(name: "gpen-bfr-512.int8.safetensors", title: "GPEN 512", optional: true),
        ModelFile(name: "xseg.int8.safetensors", title: "XSeg", optional: true),
    ]

    private nonisolated static let resources = Bundle.main.bundleURL.appendingPathComponent("FaceJutsu_FaceJutsu.bundle")
    private nonisolated static let bundled = resources.appendingPathComponent("Models", isDirectory: true)
    private nonisolated static let downloaded = Store.root.appendingPathComponent("Models", isDirectory: true)

    /// Where the models are: a build that carries them uses its own.
    nonisolated static var folder: URL {
        let carried = files.filter { !$0.optional }.allSatisfy { FileManager.default.fileExists(atPath: bundled.appendingPathComponent($0.name).path) }
        return carried ? bundled : downloaded
    }

    /// each model's address and size in bytes, from the files the desktop ships
    private let urls: [String: URL]
    private let sizes: [String: Int64]

    private(set) var states: [String: ModelState] = [:]
    private(set) var busy = false
    /// the whole download, 0 to 1 (nil when not downloading)
    private(set) var overall: Double?
    /// what went wrong
    private(set) var note: String?
    private(set) var failed = false
    /// the start stays on the models until what's being got has come, or they go on without it
    private var staying = false
    /// whether the app is in front: downloads don't count as failed while it's away
    private var active = true

    private init() {
        urls = Self.readUrls()
        sizes = Self.readSizes()
        refresh()
    }

    func size(_ f: ModelFile) -> Int64? { sizes[f.name] }

    func refresh() {
        let folder = Self.folder
        for f in Self.files where !busy || states[f.name] == nil {
            let there = FileManager.default.fileExists(atPath: folder.appendingPathComponent(f.name).path)
            let failedBefore = states[f.name] == .failed
            states[f.name] = there ? .ready : failedBefore ? .failed : .missing
        }
    }

    func has(_ name: String) -> Bool { states[name] == .ready }

    /// The models that swapping can't do without are all here.
    var ready: Bool { Self.files.filter { !$0.optional }.allSatisfy { has($0.name) } }
    var allHere: Bool { Self.files.allSatisfy { has($0.name) } }
    /// The start shows the models rather than the work.
    var gated: Bool { !ready || staying }
    /// They can go on: the models needed are here, but others didn't come.
    var canStart: Bool { ready && staying && !busy }

    func start() { staying = false }

    /// What a stretch's settings can do with the models here: what lacks its model is left out.
    func usable(_ s: SegmentSettings) -> SegmentSettings {
        var s = s
        if !enhancerHere(s) { s.quality = "Off" }
        if !objectsHere(s) { s.objects = false }
        return s
    }

    /// False when these settings ask for an enhancer that isn't here.
    func enhancerHere(_ s: SegmentSettings) -> Bool {
        switch s.quality {
        case "Off": true
        case "Best": has("gpen-bfr-512.int8.safetensors")
        default: has("gpen-bfr-256.int8.safetensors")
        }
    }

    /// False when these settings ask for Objects and its model isn't here.
    func objectsHere(_ s: SegmentSettings) -> Bool { !s.objects || has("xseg.int8.safetensors") }

    func setActive(_ now: Bool) {
        active = now
        if now { refresh() }
    }

    // MARK: downloading

    func downloadAll() {
        guard !busy else { return }
        let list = Self.files.filter { !has($0.name) && urls[$0.name] != nil }
        guard !list.isEmpty else { return }
        begin()
        for f in list { states[f.name] = .waiting }
        let total = list.reduce(Int64(0)) { $0 + (sizes[$1.name] ?? 1) }
        overall = 0
        Task {
            var finished: Int64 = 0
            var why: [String] = []
            for f in list {
                let bytes = sizes[f.name] ?? 1
                let base = finished
                states[f.name] = .getting(0)
                do {
                    try await download(f) { [weak self] part in
                        self?.states[f.name] = .getting(part)
                        self?.overall = (Double(base) + part * Double(bytes)) / Double(max(total, 1))
                    }
                    states[f.name] = .ready
                } catch {
                    states[f.name] = .failed
                    why.append("\(f.name): \(describe(error))")
                }
                finished += bytes
            }
            end(why)
        }
    }

    private func begin() {
        if !ready { staying = true }
        busy = true
        failed = false
        note = nil
        UIApplication.shared.isIdleTimerDisabled = true
    }

    private func end(_ why: [String]) {
        busy = false
        overall = nil
        failed = !why.isEmpty
        note = why.isEmpty ? nil : why.joined(separator: " · ")
        UIApplication.shared.isIdleTimerDisabled = false
        refresh()
        if allHere { staying = false }
    }

    /// One file, carrying on from where it broke off when it can. A connection lost while the app
    /// is away doesn't count: it goes on when the app is back.
    private func download(_ f: ModelFile, progress: @escaping @MainActor (Double) -> Void) async throws {
        guard let url = urls[f.name] else { throw VideoError("no address") }
        let dir = Self.downloaded
        try Self.makeFolder(dir)
        var tries = 0
        while true {
            do {
                try await Fetch.get(url, as: dir.appendingPathComponent(f.name), resume: dir.appendingPathComponent(f.name + ".resume"), size: sizes[f.name], progress: progress)
                return
            } catch {
                if error is VideoError { throw error }
                let away = !active
                while !active { try? await Task.sleep(for: .milliseconds(300)) }
                if !away { tries += 1 }
                if tries >= 4 { throw error }
                try? await Task.sleep(for: .seconds(1))
            }
        }
    }

    // MARK: from Files

    func bringIn(_ picked: [URL]) {
        guard !busy, !picked.isEmpty else { return }
        begin()
        let sizes = sizes
        Task {
            let dir = Self.downloaded
            let why: [String] = await Task.detached(priority: .userInitiated) {
                var why: [String] = []
                for url in picked {
                    let name = url.lastPathComponent
                    let reading = url.startAccessingSecurityScopedResource()
                    defer { if reading { url.stopAccessingSecurityScopedResource() } }
                    guard Self.files.contains(where: { $0.name == name }) else {
                        why.append("\(name) isn't one of the models.")
                        continue
                    }
                    do {
                        try Self.makeFolder(dir)
                        let part = dir.appendingPathComponent(UUID().uuidString + ".part")
                        defer { try? FileManager.default.removeItem(at: part) }
                        try FileManager.default.copyItem(at: url, to: part)
                        if let want = sizes[name], Self.bytes(part) != want {
                            why.append("\(name) isn't the right size.")
                            continue
                        }
                        try Self.install(part, as: dir.appendingPathComponent(name))
                    } catch {
                        why.append("\(name): \(describe(error))")
                    }
                }
                return why
            }.value
            end(why)
        }
    }

    // MARK: the files

    fileprivate nonisolated static func bytes(_ url: URL) -> Int64 {
        ((try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? NSNumber)?.int64Value ?? -1
    }

    fileprivate nonisolated static func install(_ from: URL, as to: URL) throws {
        try? FileManager.default.removeItem(at: to)
        try FileManager.default.moveItem(at: from, to: to)
    }

    private nonisolated static func makeFolder(_ dir: URL) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        // (big, and they can be got again)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        var url = dir
        try url.setResourceValues(values)
    }

    /// The addresses under `[model_urls]` in the settings file.
    private static func readUrls() -> [String: URL] {
        guard let text = try? String(contentsOf: resources.appendingPathComponent("Config/config.toml"), encoding: .utf8) else { return [:] }
        var out: [String: URL] = [:]
        var inside = false
        for line in text.split(whereSeparator: \.isNewline) {
            let line = line.trimmingCharacters(in: .whitespaces)
            if line.hasPrefix("[") {
                inside = line == "[model_urls]"
            } else if inside, !line.hasPrefix("#") {
                let bits = line.components(separatedBy: "\"")
                if bits.count >= 4, let url = URL(string: bits[3]), url.scheme == "https" { out[bits[1]] = url }
            }
        }
        return out
    }

    /// Each file's size in the manifest.
    private static func readSizes() -> [String: Int64] {
        guard let data = try? Data(contentsOf: resources.appendingPathComponent("Config/manifest.json")),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let components = json["components"] as? [String: [String: Any]] else { return [:] }
        var out: [String: Int64] = [:]
        for c in components.values {
            for b in (c["builds"] as? [String: [String: Any]] ?? [:]).values {
                if let path = b["path"] as? String, let size = (b["size"] as? NSNumber)?.int64Value { out[path] = size }
            }
        }
        return out
    }
}

/// One download: into a temporary file, checked and moved into place when it's whole. What broke
/// off is kept in `resume` to carry on from.
private final class Fetch: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let target: URL
    private let resume: URL
    private let size: Int64?
    private let progress: @MainActor (Double) -> Void
    private var finish: CheckedContinuation<Void, Error>?
    private var problem: Error?
    private var shown = -1

    private init(target: URL, resume: URL, size: Int64?, progress: @escaping @MainActor (Double) -> Void) {
        self.target = target
        self.resume = resume
        self.size = size
        self.progress = progress
    }

    static func get(_ url: URL, as target: URL, resume: URL, size: Int64?, progress: @escaping @MainActor (Double) -> Void) async throws {
        let fetch = Fetch(target: target, resume: resume, size: size, progress: progress)
        let session = URLSession(configuration: .default, delegate: fetch, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        try await withCheckedThrowingContinuation { (c: CheckedContinuation<Void, Error>) in
            fetch.finish = c
            if let data = try? Data(contentsOf: resume) {
                session.downloadTask(withResumeData: data).resume()
            } else {
                session.downloadTask(with: url).resume()
            }
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytes: Int64, totalBytesWritten done: Int64, totalBytesExpectedToWrite expected: Int64) {
        let total = size ?? expected
        guard total > 0 else { return }
        let part = min(Double(done) / Double(total), 1)
        let percent = Int(part * 100)
        guard percent != shown else { return }
        shown = percent
        let progress = progress
        Task { @MainActor in progress(part) }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        // (the file is gone once this returns)
        if let code = (downloadTask.response as? HTTPURLResponse)?.statusCode, !(200..<300).contains(code) {
            problem = VideoError("the address answered \(code)")
            return
        }
        let got = ModelFiles.bytes(location)
        if let size, got != size {
            problem = VideoError("got \(got) bytes, not the \(size) expected: the address may hold another build of it")
            return
        }
        do { try ModelFiles.install(location, as: target) } catch { problem = VideoError(describe(error)) }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error {
            // what it's got so far, to carry on from; none means starting again
            if let data = (error as NSError).userInfo[NSURLSessionDownloadTaskResumeData] as? Data {
                try? data.write(to: resume)
            } else {
                try? FileManager.default.removeItem(at: resume)
            }
            finish?.resume(throwing: error)
        } else {
            try? FileManager.default.removeItem(at: resume)
            if let problem { finish?.resume(throwing: problem) } else { finish?.resume() }
        }
        finish = nil
    }
}
