import SwiftUI
import UniformTypeIdentifiers

/// The models, one to a row: name, size, and how far along.
struct ModelList: View {
    private let store = ModelFiles.shared

    var body: some View {
        VStack(spacing: 0) {
            ForEach(ModelFiles.files) { f in
                row(f)
                if case .getting(let part) = store.states[f.name] ?? .missing {
                    ProgressBar(value: part).padding(.bottom, 4)
                }
            }
        }
    }

    private func row(_ f: ModelFile) -> some View {
        HStack(spacing: 10) {
            Text(f.title).font(.ui(14, 500)).foregroundStyle(Ink.fg).lineLimit(1)
            if f.optional { Text("Optional").font(.ui(11)).foregroundStyle(Ink.faint) }
            Spacer(minLength: 0)
            if let bytes = store.size(f) {
                Text("\(Int((Double(bytes) / 1_000_000).rounded())) MB").font(.mono(12, 500)).foregroundStyle(Ink.faint).monospacedDigit()
            }
            state(store.states[f.name] ?? .missing).frame(width: 62, alignment: .trailing)
        }
        .frame(height: 30)
    }

    @ViewBuilder
    private func state(_ s: ModelState) -> some View {
        switch s {
        case .missing: EmptyView()
        case .waiting: Text("Waiting").font(.ui(12, 500)).foregroundStyle(Ink.muted)
        case .ready: Text("Ready").font(.ui(12, 500)).foregroundStyle(Ink.good)
        case .failed: Text("Failed").font(.ui(12, 500)).foregroundStyle(Ink.bad)
        case .getting(let part): Text("\(Int(part * 100))%").font(.mono(12, 500)).foregroundStyle(Ink.fg)
        }
    }
}

/// Download all and From Files, and what went wrong.
struct ModelKeys: View {
    private let store = ModelFiles.shared
    @State private var picking = false

    private var title: String {
        if let done = store.overall { return "Downloading… \(Int(done * 100))%" }
        return store.failed ? "Try again" : "Download all"
    }

    var body: some View {
        VStack(spacing: 8) {
            if let note = store.note {
                Text(note).font(.ui(12)).foregroundStyle(Ink.bad).multilineTextAlignment(.center)
            }
            HStack(spacing: 8) {
                Button(title) { store.downloadAll() }
                    .buttonStyle(BigKeyStyle(red: true, height: 44))
                    .disabled(store.busy || store.allHere)
                    .opacity(store.allHere ? 0.35 : 1)
                Button("From Files") { picking = true }
                    .buttonStyle(BigKeyStyle(height: 44))
                    .disabled(store.busy)
                    .opacity(store.busy ? 0.35 : 1)
            }
            .padding(.bottom, Metric.edge)
            if store.canStart {
                Button("Start") { store.start() }.buttonStyle(TextButtonStyle())
            }
        }
        .fileImporter(isPresented: $picking, allowedContentTypes: [.data], allowsMultipleSelection: true) { result in
            if case .success(let urls) = result { store.bringIn(urls) }
        }
    }
}

/// The models, from Settings.
struct ModelsPage: View {
    let onBack: () -> Void

    var body: some View {
        PageScaffold("Models", onBack: onBack) {
            VStack(spacing: 16) {
                ModelList()
                ModelKeys()
            }
            .padding(Metric.gutter)
        }
    }
}
