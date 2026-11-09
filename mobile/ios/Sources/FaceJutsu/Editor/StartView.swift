import SwiftUI
import UIKit

/// Where the app starts: the saved work, newest first, as pictures in rows of one height that fill
/// the width (each at its own shape, as the photo strip keeps them), and the two keys that bring in a
/// video or photos. The first time, with nothing saved, the welcome instead.
struct StartView: View {
    let model: EditorModel
    @State private var editing = false
    /// the work whose remove seal asks "Sure?"
    @State private var armed: String?
    @State private var pictures: [String: UIImage] = [:]

    private static let welcome = "Swap faces in your photos and videos. Fast, private, and all on your phone."

    var body: some View {
        VStack(spacing: 0) {
            if ModelFiles.shared.gated {
                hero("Almost there. Grab the models and start swapping.").frame(maxWidth: .infinity, maxHeight: .infinity)
                VStack(spacing: 12) {
                    ModelList()
                    ModelKeys()
                }
                .padding(.horizontal, Metric.gutter)
                .padding(.top, 10)
                .padding(.bottom, 8)
            } else {
                if model.gallery.isEmpty {
                    hero(Self.welcome).frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    gallery
                }
                dock
            }
        }
        .onChange(of: model.gallery) { _, now in if now.isEmpty { editing = false } }
    }

    // MARK: the welcome

    private func hero(_ lead: String) -> some View {
        VStack(spacing: 14) {
            Mark(size: 72, paper: Ink.bg).foregroundStyle(Ink.fg)
            (Text("Face swap no ") + Text("jutsu.").foregroundColor(Ink.accent))
                .font(.display(36, 800))
                .tracking(-0.035 * 36)
                .foregroundStyle(Ink.fg)
                .multilineTextAlignment(.center)
            Text(lead)
                .font(.ui(15))
                .foregroundStyle(Ink.muted)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 300)
        }
        .padding(.horizontal, 28)
    }

    // MARK: the saved work

    private var gallery: some View {
        GeometryReader { geo in
            let width = geo.size.width - Metric.gutter * 2
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 4, pinnedViews: [.sectionHeaders]) {
                    Section {
                        ForEach(Array(rows(width: width).enumerated()), id: \.offset) { _, row in
                            HStack(spacing: 4) {
                                ForEach(row.items) { item in tile(item, width: row.height * item.aspect, height: row.height) }
                            }
                        }
                    } header: { header }
                }
                .padding(.horizontal, Metric.gutter)
                .padding(.bottom, 8)
            }
            .scrollIndicators(.hidden)
        }
    }

    private var header: some View {
        HStack(spacing: 8) {
            Text(editing ? "Edit your work" : "Your work").overline(11)
            Spacer(minLength: 8)
            if editing {
                Button("Done") {
                    armed = nil
                    withAnimation(Motion.ease(Motion.quick)) { editing = false }
                }
                .buttonStyle(TextButtonStyle())
                .frame(height: 32)
            } else {
                Text("\(model.gallery.count)").font(.ui(11)).foregroundStyle(Ink.faint)
                Button { withAnimation(Motion.ease(Motion.quick)) { editing = true } } label: { IconView(.pen, size: 16, weight: 2) }
                    .buttonStyle(IconButtonStyle(color: Ink.muted, height: 32))
                    .padding(.trailing, -Metric.gutter + 4)
                    .accessibilityLabel("Edit")
            }
        }
        .frame(height: 32)
        .background(Ink.bg)
    }

    private struct Row {
        var items: [WorkSummary]
        var height: CGFloat
    }

    /// Rows of one height that fill the width; the last, if short, at the usual height.
    private func rows(width: CGFloat) -> [Row] {
        let target: CGFloat = 118, gap: CGFloat = 4
        var out: [Row] = []
        var row: [WorkSummary] = []
        var sum: CGFloat = 0
        for w in model.gallery {
            row.append(w)
            sum += w.aspect
            if sum * target + gap * CGFloat(row.count - 1) >= width {
                out.append(Row(items: row, height: (width - gap * CGFloat(row.count - 1)) / sum))
                row = []
                sum = 0
            }
        }
        if !row.isEmpty { out.append(Row(items: row, height: target)) }
        return out
    }

    private func tile(_ w: WorkSummary, width: CGFloat, height: CGFloat) -> some View {
        Button {
            guard !editing else { return }
            Task { await model.open(w.id) }
        } label: {
            ZStack {
                if let image = pictures[w.id] {
                    Image(uiImage: image).resizable().scaledToFill()
                } else {
                    Ink.panel2
                }
            }
            .frame(width: width, height: height)
            .clipShape(cornerShape)
            .overlay(alignment: .bottomLeading) {
                Text(w.tag)
                    .font(.mono(10, 600))
                    .foregroundStyle(Color(hex: 0xf2ede6))
                    .padding(.horizontal, 4)
                    .frame(height: 16)
                    .background { markShape.fill(Color(hex: 0x121014, alpha: 0.78)) }
                    .padding(4)
            }
            .overlay(alignment: .topTrailing) { if editing { removeSeal(w) } }
            .contentShape(Rectangle())
        }
        .buttonStyle(TilePress())
        .task(id: w.picture) {
            guard pictures[w.id] == nil else { return }
            let url = w.picture
            pictures[w.id] = await Task.detached { UIImage(contentsOfFile: url.path) }.value
        }
        .accessibilityLabel(w.video ? "Video, \(w.tag)" : "Photos, \(w.tag)")
    }

    /// The first tap asks; the second removes the work.
    private func removeSeal(_ w: WorkSummary) -> some View {
        let sure = armed == w.id
        return Button {
            if sure {
                armed = nil
                withAnimation(Motion.ease(Motion.move)) { model.removeWork(w.id) }
            } else {
                withAnimation(Motion.ease(Motion.quick)) { armed = w.id }
                Task { @MainActor in
                    try? await Task.sleep(for: .milliseconds(2400))
                    if armed == w.id { withAnimation(Motion.ease(Motion.quick)) { armed = nil } }
                }
            }
        } label: {
            Group {
                if sure { Text("Sure?").font(.ui(10, 600)).padding(.horizontal, 5) } else { IconView(.close, size: 11, weight: 3) }
            }
            .foregroundStyle(Ink.accentInk)
            .frame(minWidth: 18, minHeight: 18)
            .background { markShape.fill(Ink.accent) }
            .rotationEffect(.degrees(sure ? 0 : -4))
            .padding(4)
            .frame(minWidth: Metric.hit, minHeight: Metric.hit, alignment: .topTrailing)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .transition(.opacity)
        .accessibilityLabel(sure ? "Remove, sure" : "Remove")
    }

    // MARK: the keys

    private var dock: some View {
        VStack(spacing: 12) {
            if let progress = model.importing {
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Text("Getting it ready").font(.ui(14, 600)).foregroundStyle(Ink.fg)
                        Spacer()
                        if let p = progress { Text("\(Int(p * 100))%").font(.mono(12, 500)).foregroundStyle(Ink.fg) }
                    }
                    ProgressBar(value: progress ?? 0)
                    Button("Cancel") { model.stopImport?() }
                        .buttonStyle(TextButtonStyle())
                        .frame(maxWidth: .infinity)
                }
                .padding(.top, 8)
                .frame(height: 96 + Metric.edge, alignment: .top)
            } else {
                HStack(spacing: 8) {
                    Button { model.picking = .video } label: {
                        VStack(spacing: 10) { IconView(.film, size: 30, weight: 1.5); Text("Video") }
                    }
                    .buttonStyle(BigKeyStyle(red: true))
                    Button { model.picking = .photos } label: {
                        VStack(spacing: 10) { IconView(.image, size: 30, weight: 1.5); Text("Photos") }
                    }
                    .buttonStyle(BigKeyStyle())
                }
                .padding(.bottom, Metric.edge)
            }
            Text("Only swap the faces of people who have agreed to it.")
                .font(.ui(11))
                .foregroundStyle(Ink.faint)
                .multilineTextAlignment(.center)
        }
        .padding(.horizontal, Metric.gutter)
        .padding(.top, 10)
        .padding(.bottom, 8)
    }
}

private struct TilePress: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label.pressDip(configuration.isPressed, scale: 0.96)
    }
}

/// A thin red bar that fills from the left.
struct ProgressBar: View {
    let value: Double

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 2).fill(Ink.line2)
                RoundedRectangle(cornerRadius: 2).fill(Ink.accent).frame(width: geo.size.width * min(max(value, 0), 1))
            }
        }
        .frame(height: 4)
        .animation(.linear(duration: 0.12), value: value)
    }
}
