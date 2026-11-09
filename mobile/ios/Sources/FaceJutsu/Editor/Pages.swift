import SwiftUI
import UIKit

/// A full page over the editor.
enum Page: Hashable {
    case settings, models, licenses, responsibly
}

/// The frame of a full page: a back key, the title and an optional action on one bar, and the
/// content under it. A swipe in from the left edge goes back.
struct PageScaffold<Content: View, Action: View>: View {
    let title: String
    let onBack: () -> Void
    @ViewBuilder let action: Action
    @ViewBuilder let content: Content

    init(_ title: String, onBack: @escaping () -> Void, @ViewBuilder action: () -> Action = { EmptyView() },
         @ViewBuilder content: () -> Content) {
        self.title = title
        self.onBack = onBack
        self.action = action()
        self.content = content()
    }

    var body: some View {
        VStack(spacing: 0) {
            ZStack {
                Text(title).font(.display(17, 700)).foregroundStyle(Ink.fg)
                HStack(spacing: 0) {
                    Button(action: onBack) { IconView(.chevronLeft) }
                        .buttonStyle(IconButtonStyle(color: Ink.fg))
                        .accessibilityLabel("Back")
                    Spacer()
                    action
                }
                .padding(.horizontal, 4)
            }
            .frame(height: Metric.hit)
            content.frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        }
        .background(Ink.bg.ignoresSafeArea())
        .simultaneousGesture(
            DragGesture(minimumDistance: 20).onEnded { g in
                if g.startLocation.x < 24 && g.translation.width > 60 { onBack() }
            }
        )
    }
}

/// A section's small heading.
struct SectionLabel: View {
    let text: String

    var body: some View {
        Text(text)
            .overline(11, color: Ink.faint)
            .padding(.horizontal, Metric.gutter)
            .padding(.top, 18)
            .padding(.bottom, 8)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// A row that opens something.
struct LinkRow: View {
    let title: String
    var value: String? = nil
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack {
                Text(title).font(.ui(15, 500)).foregroundStyle(Ink.fg)
                Spacer()
                if let value { Text(value).font(.mono(12, 500)).foregroundStyle(Ink.faint).monospacedDigit() }
                IconView(.chevronRight, size: 16, weight: 1.75).foregroundStyle(Ink.faint)
            }
            .padding(.horizontal, Metric.gutter)
            .frame(height: Metric.hit)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

/// A word on a row, with something on the end.
private struct InfoRow<End: View>: View {
    let title: String
    @ViewBuilder let end: End

    var body: some View {
        HStack(spacing: 12) {
            Text(title).font(.ui(15, 500)).foregroundStyle(Ink.fg)
            Spacer(minLength: 8)
            end
        }
        .padding(.horizontal, Metric.gutter)
        .frame(minHeight: Metric.hit)
    }
}

/// An action that can't be taken back: the first tap asks, the second does it.
private struct ConfirmAction: View {
    let title: String
    var danger = false
    var enabled = true
    let action: () -> Void
    @State private var armed = false
    @State private var done = false
    @State private var nudges = 0
    @State private var disarm: Task<Void, Never>?

    var body: some View {
        Button {
            if armed {
                disarm?.cancel()
                armed = false
                done = true
                action()
            } else {
                armed = true
                nudges += 1
                disarm = Task { @MainActor in
                    try? await Task.sleep(for: .milliseconds(2400))
                    if !Task.isCancelled { armed = false }
                }
            }
        } label: {
            Text(done ? "Done" : armed ? "Sure?" : title)
                .frame(minWidth: 58, alignment: .trailing)
                .iconAct(.slide, on: nudges)
        }
        .buttonStyle(TextButtonStyle(danger: danger || armed))
        .disabled(done || !enabled)
        .opacity(done || !enabled ? 0.4 : 1)
    }
}

/// The app's settings.
struct SettingsPage: View {
    let model: EditorModel
    let onBack: () -> Void
    let open: (Page) -> Void
    @AppStorage("appearance") private var appearance = Appearance.system
    @AppStorage("exportQuality") private var quality = "High"
    @AppStorage("saveTo") private var saveTo = "Photos"
    @AppStorage("keepAwake") private var keepAwake = true
    @AppStorage("haptics") private var haptics = true
    @State private var themeFrame: CGRect = .zero
    // (counted off the main thread: there can be thousands of swapped frames)
    @State private var savedWork = 0
    @State private var swappedFrames = 0

    private func size(_ megabytes: Int) -> some View {
        Text("\(megabytes) MB").font(.mono(12, 500)).foregroundStyle(Ink.faint).monospacedDigit()
    }

    // The new theme spreads out from the key that was tapped.
    private var themeBinding: Binding<Appearance> {
        Binding(get: { appearance }, set: { next in
            guard next != appearance else { return }
            let i = CGFloat(Appearance.allCases.firstIndex(of: next) ?? 0)
            let n = CGFloat(Appearance.allCases.count)
            let at = CGPoint(x: themeFrame.minX + themeFrame.width * (i + 0.5) / n, y: themeFrame.midY)
            ThemeReveal.run(next, at: at)
            appearance = next
        })
    }

    var body: some View {
        PageScaffold("Settings", onBack: onBack) {
            ScrollView {
                VStack(spacing: 0) {
                    Group {
                        SectionLabel(text: "Export").rise(0, dx: 14, dy: 0)
                        OptionRow(label: "Quality", height: 44) {
                            SegKeys(options: ["Good", "High", "Very high"].map { ($0, $0) }, selection: $quality)
                        }
                        .padding(.horizontal, Metric.gutter).rise(1, dx: 14, dy: 0)
                        OptionRow(label: "Save to", height: 44) {
                            SegKeys(options: ["Photos", "Files"].map { ($0, $0) }, selection: $saveTo)
                        }
                        .padding(.horizontal, Metric.gutter).rise(2, dx: 14, dy: 0)
                        InfoRow(title: "Keep the screen on") { SealSwitch(isOn: $keepAwake, label: "Keep the screen on").padding(.trailing, -3) }
                            .rise(3, dx: 14, dy: 0)
                        SectionLabel(text: "Appearance").rise(4, dx: 14, dy: 0)
                        OptionRow(label: "Theme", height: 44) {
                            SegKeys(options: Appearance.allCases.map { ($0, $0.label) }, selection: themeBinding)
                                .background {
                                    GeometryReader { g in
                                        Color.clear
                                            .onAppear { themeFrame = g.frame(in: .global) }
                                            .onChange(of: g.frame(in: .global)) { _, f in themeFrame = f }
                                    }
                                }
                        }
                        .padding(.horizontal, Metric.gutter).rise(5, dx: 14, dy: 0)
                        InfoRow(title: "Haptics") { SealSwitch(isOn: $haptics, label: "Haptics").padding(.trailing, -3) }
                            .rise(6, dx: 14, dy: 0)
                    }
                    Group {
                        SectionLabel(text: model.work?.kind == "photos" ? "These photos" : "This video").rise(7, dx: 14, dy: 0)
                        InfoRow(title: "Start over") { ConfirmAction(title: "Reset", danger: true, enabled: model.work != nil) { model.startOver() } }
                            .rise(8, dx: 14, dy: 0)
                        SectionLabel(text: "Storage")
                        LinkRow(title: "Models", value: "\(EditorModel.modelsMegabytes) MB") { open(.models) }
                        InfoRow(title: "Swapped frames") {
                            size(swappedFrames)
                            ConfirmAction(title: "Remove", enabled: swappedFrames > 0) {
                                Patches.remove()
                                swappedFrames = 0
                                model.live?.refreshMade()
                            }
                        }
                        InfoRow(title: "Saved work") {
                            size(savedWork)
                            ConfirmAction(title: "Forget", enabled: savedWork > 0 || !model.gallery.isEmpty) {
                                if model.work != nil { model.close() }
                                for w in model.gallery { model.removeWork(w.id) }
                                savedWork = 0
                                swappedFrames = 0
                            }
                        }
                    }
                    Group {
                        SectionLabel(text: "About")
                        InfoRow(title: "Version") {
                            Text(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "")
                                .font(.mono(12, 500)).foregroundStyle(Ink.faint)
                        }
                        LinkRow(title: "Licenses") { open(.licenses) }
                        LinkRow(title: "Use it responsibly") { open(.responsibly) }
                    }
                }
                .padding(.bottom, 48)
            }
        }
        .task {
            let sizes = await Task.detached(priority: .utility) { (Store.megabytes(Store.works), Store.megabytes(Patches.folder)) }.value
            (savedWork, swappedFrames) = sizes
        }
    }
}

/// A page of plain words.
struct TextPage: View {
    let title: String
    let paragraphs: [String]
    let onBack: () -> Void

    var body: some View {
        PageScaffold(title, onBack: onBack) {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    ForEach(paragraphs, id: \.self) { text in
                        Text(text).font(.ui(15)).foregroundStyle(Ink.fg).fixedSize(horizontal: false, vertical: true)
                    }
                }
                .padding(Metric.gutter)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }
}

/// Dark, light, or as the system is.
enum Appearance: String, CaseIterable {
    case system, dark, light

    var label: String {
        switch self {
        case .system: "System"
        case .dark: "Dark"
        case .light: "Light"
        }
    }

    /// Sets the whole app's windows to this appearance, or back to the system's.
    @MainActor
    func apply() {
        let style: UIUserInterfaceStyle
        switch self {
        case .system: style = .unspecified
        case .dark: style = .dark
        case .light: style = .light
        }
        for scene in UIApplication.shared.connectedScenes {
            guard let windows = (scene as? UIWindowScene)?.windows else { continue }
            for window in windows { window.overrideUserInterfaceStyle = style }
        }
    }
}

/// The new theme spreading from a point: a picture of the old look stays on top, with a hole growing
/// out of that point until it's gone.
enum ThemeReveal {
    @MainActor
    static func run(_ next: Appearance, at point: CGPoint) {
        let window = UIApplication.shared.connectedScenes.compactMap { ($0 as? UIWindowScene)?.windows.first }.first
        guard let window, !UIAccessibility.isReduceMotionEnabled, let old = window.snapshotView(afterScreenUpdates: false) else {
            next.apply()
            return
        }
        old.isUserInteractionEnabled = false
        window.addSubview(old)
        next.apply()
        let bounds = window.bounds
        let radius = hypot(max(point.x, bounds.width - point.x), max(point.y, bounds.height - point.y)) + 8
        func hole(_ r: CGFloat) -> CGPath {
            let p = UIBezierPath(rect: bounds)
            p.append(UIBezierPath(arcCenter: point, radius: r, startAngle: 0, endAngle: .pi * 2, clockwise: true))
            return p.cgPath
        }
        let mask = CAShapeLayer()
        mask.fillRule = .evenOdd
        mask.path = hole(radius)
        old.layer.mask = mask
        CATransaction.begin()
        CATransaction.setCompletionBlock { old.removeFromSuperview() }
        let grow = CABasicAnimation(keyPath: "path")
        grow.fromValue = hole(0.1)
        grow.toValue = hole(radius)
        grow.duration = Motion.flourish
        grow.timingFunction = CAMediaTimingFunction(controlPoints: 0.16, 1, 0.3, 1)
        mask.add(grow, forKey: "path")
        CATransaction.commit()
    }
}
