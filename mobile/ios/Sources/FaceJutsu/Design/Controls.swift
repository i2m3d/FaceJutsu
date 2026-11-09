import SwiftUI

// The controls: letterpress keys on an edge that sink onto it when pressed (down at once, back in a
// moment), one paper key that travels between choices, a switch, a red-filled slider and seals. Only
// something you press or have chosen gets a shape; the rest is ink on the background.

/// The corner of everything seen as an object.
let cornerShape = RoundedRectangle(cornerRadius: Metric.corner)
/// The same corner on small marks.
let markShape = RoundedRectangle(cornerRadius: Metric.cornerSmall)

/// Something on a hard edge: the edge stays put and the face sinks onto it when pressed.
private struct OnEdge: ViewModifier {
    let edge: Color
    let height: CGFloat
    let pressed: Bool

    func body(content: Content) -> some View {
        let sink = height - 1
        content
            .offset(y: pressed ? sink : 0)
            .background {
                cornerShape.offset(y: height).subtracting(cornerShape.offset(y: pressed ? sink : 0)).fill(edge)
            }
            .animation(pressed ? nil : Motion.ease(Motion.quick), value: pressed)
    }
}

extension View {
    fileprivate func onEdge(_ edge: Color, height: CGFloat, pressed: Bool) -> some View {
        modifier(OnEdge(edge: edge, height: height, pressed: pressed))
    }

    /// A light tap when a control goes down.
    fileprivate func pressHaptic(_ pressed: Bool) -> some View {
        sensoryFeedback(.impact(weight: .light, intensity: 0.7), trigger: pressed) { _, now in now && Feel.on }
    }

    /// Dips at once when pressed, and springs back.
    func pressDip(_ pressed: Bool, scale: CGFloat = 0.93) -> some View {
        scaleEffect(pressed ? scale : 1)
            .animation(pressed ? nil : Motion.ease(Motion.quick), value: pressed)
    }
}

/// The red key: the one primary action on a screen.
struct PrimaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.ui(13, 600))
            .foregroundStyle(Ink.accentInk)
            .padding(.leading, 8)
            .padding(.trailing, 11)
            .frame(height: 30)
            .background { cornerShape.fill(Ink.accent.shadow(.inner(color: .white.opacity(0.18), radius: 0, x: 0, y: 1))) }
            .contentShape(cornerShape)
            .onEdge(Ink.accentEdge, height: Metric.edge, pressed: configuration.isPressed)
            .frame(minHeight: Metric.hit)
            .contentShape(Rectangle())
            .pressHaptic(configuration.isPressed)
    }
}

/// A large key for the start's choices: red for the first, paper for the other, its icon over its
/// word, on the same letterpress edge as the red key.
struct BigKeyStyle: ButtonStyle {
    var red = false
    var height: CGFloat = 96

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.ui(15, 600))
            .foregroundStyle(red ? Ink.accentInk : Ink.keyFg)
            .frame(maxWidth: .infinity)
            .frame(height: height)
            .background { cornerShape.fill(red ? Ink.accent : Ink.keyBg) }
            .contentShape(cornerShape)
            .onEdge(red ? Ink.accentEdge : Ink.keyEdge, height: Metric.edge, pressed: configuration.isPressed)
            .pressHaptic(configuration.isPressed)
    }
}

/// A small paper key, for the second choice beside a quiet one.
struct PaperButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.ui(13, 600))
            .foregroundStyle(Ink.keyFg)
            .padding(.horizontal, 12)
            .frame(height: 34)
            .background { cornerShape.fill(Ink.keyBg) }
            .contentShape(cornerShape)
            .onEdge(Ink.keyEdge, height: Metric.edgeSmall, pressed: configuration.isPressed)
            .frame(minHeight: Metric.hit)
            .contentShape(Rectangle())
            .pressHaptic(configuration.isPressed)
    }
}

/// The quiet action: words alone, red when it can't be taken back.
struct TextButtonStyle: ButtonStyle {
    var danger = false

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.ui(13, 600))
            .foregroundStyle(danger ? Ink.bad : Ink.fg)
            .frame(minHeight: Metric.hit)
            .contentShape(Rectangle())
            .pressDip(configuration.isPressed)
    }
}

/// A bare icon in a full-size target.
struct IconButtonStyle: ButtonStyle {
    var color: Color = Ink.muted
    var height: CGFloat = Metric.hit

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundStyle(color)
            .frame(width: Metric.hit, height: height)
            .background { cornerShape.fill(configuration.isPressed ? Ink.press : .clear) }
            .contentShape(Rectangle())
            .pressDip(configuration.isPressed)
    }
}

/// A paper key on its edge, as play is. The target is larger than what's drawn.
struct KeyButtonStyle: ButtonStyle {
    var width: CGFloat = 36
    var height: CGFloat = 26

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundStyle(Ink.keyFg)
            .frame(width: width, height: height)
            .background { cornerShape.fill(Ink.keyBg) }
            .onEdge(Ink.keyEdge, height: Metric.edgeSmall, pressed: configuration.isPressed)
            .frame(width: max(width, Metric.hit), height: max(height, 36))
            .contentShape(Rectangle())
            .pressHaptic(configuration.isPressed)
    }
}

// MARK: - Seals

/// A seal: a small red stamp, slightly askew, with a number on it.
struct Seal: View {
    var text: String = ""
    var size: CGFloat = 16

    var body: some View {
        Text(text)
            .font(.ui(size * 0.56, 700))
            .foregroundStyle(Ink.accentInk)
            .frame(width: size, height: size)
            .background { markShape.fill(Ink.accent) }
            .rotationEffect(.degrees(-4))
    }
}

/// The ✓ seal on something chosen.
struct CheckSeal: View {
    var size: CGFloat = 14

    var body: some View {
        IconView(.check, size: size * 0.8, weight: 3)
            .foregroundStyle(Ink.accentInk)
            .frame(width: size, height: size)
            .background { markShape.fill(Ink.accent) }
            .rotationEffect(.degrees(-6))
    }
}

// MARK: - Choosing

/// Lays out a row as CSS's `flex: 1 1 auto` does: each item gets its own width plus an equal share of
/// what's left over, so a longer label gets a wider key. When they don't fit, all shrink alike.
private struct FlexRow: Layout {
    var spacing: CGFloat = 2

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let ideal = subviews.map { $0.sizeThatFits(.unspecified) }
        let natural = ideal.map(\.width).reduce(0, +) + spacing * CGFloat(max(0, subviews.count - 1))
        return CGSize(width: proposal.width ?? natural, height: proposal.height ?? (ideal.map(\.height).max() ?? 0))
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let ideal = subviews.map { $0.sizeThatFits(.unspecified).width }
        let gaps = spacing * CGFloat(max(0, subviews.count - 1))
        let natural = ideal.reduce(0, +)
        let room = bounds.width - gaps
        let widths: [CGFloat] = room >= natural
            ? ideal.map { $0 + (room - natural) / CGFloat(subviews.count) }
            : ideal.map { $0 * room / max(natural, 1) }
        var x = bounds.minX
        for (view, w) in zip(subviews, widths) {
            view.place(at: CGPoint(x: x, y: bounds.minY), proposal: ProposedViewSize(width: w, height: bounds.height))
            x += w + spacing
        }
    }
}

/// One option out of a few: words in a row, the chosen one on a raised paper key that travels to the
/// next choice. Each key is as wide as its label, and they share the rest of the row.
struct SegKeys<Value: Hashable>: View {
    let options: [(Value, String)]
    @Binding var selection: Value
    var height: CGFloat = 32

    @State private var rects: [AnyHashable: CGRect] = [:]
    @State private var from: Value?
    @State private var tick = 0

    var body: some View {
        FlexRow {
            ForEach(options, id: \.0) { value, label in
                let on = value == selection
                Button {
                    guard value != selection else { return }
                    from = selection
                    tick &+= 1
                    selection = value
                } label: {
                    HStack(spacing: 5) {
                        // the seal's room is kept when it's hidden, so keys don't change width
                        Rectangle().fill(Ink.accent).frame(width: 6, height: 6).rotationEffect(.degrees(-6))
                            .opacity(on ? 1 : 0)
                            .stamped(on: on)
                        Text(label)
                            .font(.ui(13, on ? 700 : 500))
                            .lineLimit(1)
                            .minimumScaleFactor(0.8)
                    }
                    .padding(.trailing, 6)
                    .padding(.leading, 4)
                    .foregroundStyle(on ? Ink.keyFg : Ink.muted)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .reportKey(value, in: "seg")
                .accessibilityAddTraits(on ? .isSelected : [])
            }
        }
        .coordinateSpace(name: "seg")
        .onPreferenceChange(KeyRects.self) { rects = $0 }
        .background {
            TravellingKey(from: from.flatMap { rects[$0] }, to: rects[selection], trigger: tick)
        }
        .frame(height: height)
        .sensoryFeedback(.selection, trigger: selection) { _, _ in Feel.on }
    }
}

/// On or off: a slot that fills with red from the left as its square knob slides over.
struct SealSwitch: View {
    @Binding var isOn: Bool
    var label: String = ""
    @GestureState private var held = false

    var body: some View {
        Button {
            withAnimation(Motion.ease(Motion.quick)) { isOn.toggle() }
        } label: {
            ZStack(alignment: isOn ? .trailing : .leading) {
                cornerShape.fill(Ink.panel2)
                Color.clear.overlay(alignment: .leading) {
                    GeometryReader { geo in
                        Rectangle().fill(Ink.accent).frame(width: isOn ? geo.size.width : 0)
                    }
                }
                .clipShape(cornerShape)
                markShape
                    .fill(isOn ? Ink.keyBg : Ink.muted)
                    .frame(width: 14, height: 14)
                    .padding(3)
            }
            .frame(width: 38, height: 20)
            .frame(width: Metric.hit, height: Metric.hit)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .sensoryFeedback(.selection, trigger: isOn) { _, _ in Feel.on }
        .accessibilityLabel(label)
        .accessibilityValue(isOn ? "On" : "Off")
        .accessibilityAddTraits(.isToggle)
    }
}

/// A value on a line: the line fills red from its origin (the left end, or the middle when the range
/// is around zero) to a red square knob with a paper rim. It lifts while held and leans with the drag.
struct SealSlider: View {
    @Binding var value: Double
    var range: ClosedRange<Double> = 0...1
    var step: Double = 0.01
    var label: String = ""
    @State private var dragging = false
    @State private var lean: Double = 0
    @State private var lastX: CGFloat = 0

    private var origin: Double { range.lowerBound < 0 && range.upperBound > 0 ? 0 : range.lowerBound }
    private func ratio(_ v: Double) -> Double { (v - range.lowerBound) / (range.upperBound - range.lowerBound) }

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let r = ratio(value), r0 = ratio(origin)
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 2).fill(Ink.line2).frame(height: 4)
                RoundedRectangle(cornerRadius: 2).fill(Ink.accent)
                    .frame(width: abs(r - r0) * w, height: 4)
                    .offset(x: min(r, r0) * w)
                markShape
                    .fill(Ink.accent)
                    .overlay { markShape.strokeBorder(Ink.keyBg, lineWidth: 2) }
                    .background { markShape.fill(Ink.accentEdge).offset(y: 1) }
                    .frame(width: 16, height: 16)
                    .rotationEffect(.degrees(dragging ? max(-26, min(14, -6 + lean)) : -6))
                    .scaleEffect(dragging ? 1.25 : 1)
                    .offset(x: r * w - 8)
                    .animation(Motion.ease(Motion.quick), value: dragging)
            }
            .frame(width: w, height: geo.size.height)
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { g in
                        if dragging { lean = (g.location.x - lastX) * 1.6 } else { lean = 0 }
                        lastX = g.location.x
                        dragging = true
                        let t = min(max(g.location.x / max(w, 1), 0), 1)
                        let raw = range.lowerBound + t * (range.upperBound - range.lowerBound)
                        value = (raw / step).rounded() * step
                    }
                    .onEnded { _ in dragging = false }
            )
        }
        .frame(height: 38)
        .accessibilityElement()
        .accessibilityLabel(label)
        .accessibilityValue(String(format: "%.2f", value))
        .accessibilityAdjustableAction { dir in
            let delta = (range.upperBound - range.lowerBound) / 20
            value = min(max(value + (dir == .increment ? delta : -delta), range.lowerBound), range.upperBound)
        }
    }
}
