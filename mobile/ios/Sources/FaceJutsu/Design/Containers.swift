import SwiftUI

// How things sit together: on the app's own background, with space between them. Nothing gets a
// frame just to hold it. A paper outline means something is selected.

/// A short message over the bottom of the timeline.
struct Toast: View {
    let text: String

    var body: some View {
        Text(text)
            .font(.ui(13, 500))
            .foregroundStyle(Ink.keyFg)
            .multilineTextAlignment(.leading)
            .padding(.horizontal, Metric.gutter)
            .padding(.vertical, 10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background { cornerShape.fill(Ink.keyBg) }
            .padding(.horizontal, 8)
            .padding(.bottom, 8)
    }
}

/// A tool's settings: the seal, title and summary, and the tool's rows. It takes the timeline's place
/// exactly, so the picture is never covered and nothing jumps between tools. The header ends in the
/// chevron that closes it, or a word (Done) when that's what closing means.
struct ToolPanel<Content: View>: View {
    var seal: String?
    /// a stamped check before the title, for something finished
    var check = false
    let title: String
    var titleColor: Color = Ink.fg
    var summary: String = ""
    var onClose: (() -> Void)?
    /// a key before the close, such as the pen
    var extra: AnyView?
    var closeLabel: String?
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 8) {
                if let seal { Seal(text: seal).stamped(on: title, ring: true) }
                if check { CheckSeal(size: 16).stamped(on: title, ring: true) }
                Text(title).overline(11, color: titleColor)
                Spacer(minLength: 8)
                Text(summary).font(.ui(11)).foregroundStyle(Ink.faint).lineLimit(1)
                if let extra { extra }
                if let onClose {
                    if let closeLabel {
                        Button(closeLabel, action: onClose)
                            .buttonStyle(TextButtonStyle())
                            .padding(.trailing, Metric.gutter)
                            .frame(height: 32)
                    } else {
                        Button(action: onClose) { IconView(.chevronDown, size: 18, weight: 1.75) }
                            .buttonStyle(IconButtonStyle(color: Ink.faint, height: 32))
                            .accessibilityLabel("Close")
                    }
                } else {
                    Color.clear.frame(width: Metric.gutter - 4, height: 1)
                }
            }
            .padding(.leading, Metric.gutter)
            .frame(height: 32)
            VStack(alignment: .leading, spacing: 0) { content }
                .padding(.horizontal, Metric.gutter)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        .frame(height: Metric.timeline, alignment: .top)
        .background(Ink.bg)
    }
}

/// One row of options: a label in a fixed column, the control, and an optional value at the end.
struct OptionRow<Control: View>: View {
    let label: String
    var value: String?
    var height: CGFloat = 38
    @ViewBuilder let control: Control

    var body: some View {
        HStack(spacing: 8) {
            Text(label)
                .font(.ui(12))
                .foregroundStyle(Ink.muted)
                .frame(width: 58, alignment: .leading)
            control
            if let value {
                Text(value)
                    .font(.mono(11, 500))
                    .foregroundStyle(Ink.fg)
                    .frame(width: 38, alignment: .trailing)
            }
        }
        .frame(height: height)
    }
}

// MARK: - The tool row

/// A tool in the tool row.
struct ToolItem: Identifiable {
    let id: String
    let icon: Icon
    var label = ""
    var seal: String?
    var enabled = true
    var move: IconMove = .still
    /// a narrow key with no word, as the one back to the main tools
    var narrow = false
}

/// The row of tools along the bottom, edge to edge, on the app's background. One paper key travels to
/// the open tool. When the set of tools changes, the new ones come in together.
struct ToolRow: View {
    let items: [ToolItem]
    var selected: String?
    let onTap: (String) -> Void

    @State private var rects: [AnyHashable: CGRect] = [:]
    @State private var from: String?
    @State private var tick = 0
    @State private var taps: [String: Int] = [:]

    private var key: String { items.map(\.id).joined(separator: ",") }

    var body: some View {
        HStack(spacing: 2) {
            ForEach(items) { item in
                toolButton(item)
            }
        }
        .id(key)
        .padding(.horizontal, 4)
        .coordinateSpace(name: "tools")
        .onPreferenceChange(KeyRects.self) { rects = $0 }
        .background {
            TravellingKey(from: from.flatMap { rects[$0] }, to: selected.flatMap { rects[$0] }, trigger: tick)
        }
        .frame(height: Metric.toolbar)
        .background(Ink.bg.ignoresSafeArea(edges: .bottom))
        .onChange(of: selected) { old, _ in
            from = old
            tick &+= 1
        }
        .sensoryFeedback(.selection, trigger: selected) { _, _ in Feel.on }
    }

    private func toolLabel(_ item: ToolItem, on: Bool) -> some View {
        VStack(spacing: 5) {
            IconView(item.icon, size: 22, weight: 1.6)
                .iconAct(item.move, on: taps[item.id, default: 0])
                .popWhen(item.enabled)
            if !item.label.isEmpty { Text(item.label).font(.ui(11, 500)).lineLimit(1) }
        }
        .foregroundStyle(on ? Ink.keyFg : Ink.muted)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        // moves inside the button, so the button itself stays put for the key to measure
        .rise(0, dy: 8)
        .overlay(alignment: .top) {
            if let seal = item.seal {
                Seal(text: seal, size: 13)
                    .offset(x: 14, y: 5)
                    .stamped(on: on, ring: on)
            }
        }
        .contentShape(Rectangle())
    }

    private func toolButton(_ item: ToolItem) -> some View {
        let on = item.id == selected
        return Button {
            taps[item.id, default: 0] += 1
            onTap(item.id)
        } label: {
            toolLabel(item, on: on)
        }
        .buttonStyle(ToolPress())
        .frame(width: item.narrow ? Metric.hit : nil)
        .frame(maxWidth: item.narrow ? nil : .infinity)
        .disabled(!item.enabled)
        .opacity(item.enabled ? 1 : 0.35)
        .animation(Motion.ease(Motion.quick), value: item.enabled)
        .reportKey(item.id, in: "tools")
        .accessibilityLabel(item.narrow ? "Back" : item.label)
        .accessibilityAddTraits(on ? .isSelected : [])
    }
}

private struct ToolPress: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label.pressDip(configuration.isPressed)
    }
}

// MARK: - Faces

/// A face cut out of a picture: `box` is where it is, as parts of the picture.
struct FaceCrop: View {
    var image: UIImage?
    var box: CGRect
    var size: CGFloat = 24

    var body: some View {
        let width = size / max(box.width, 0.01), height = size / max(box.height, 0.01)
        Color.clear
            .frame(width: size, height: size)
            .overlay(alignment: .topLeading) {
                if let image {
                    Image(uiImage: image).resizable()
                        .frame(width: width, height: height)
                        .offset(x: -box.minX * width, y: -box.minY * height)
                } else {
                    Ink.panel2
                }
            }
            .clipShape(cornerShape)
    }
}

/// A character: their face at 3:4 and their name. Chosen, it gets the paper outline and a stamped seal.
struct CharacterCard: View {
    var name: String
    var image: UIImage?
    var chosen = false
    /// being edited: the pen before the name, which a tap renames
    var editing = false

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Color.clear
                .aspectRatio(3 / 4, contentMode: .fit)
                .overlay {
                    if let image { Image(uiImage: image).resizable().scaledToFill() } else { Ink.panel2 }
                }
                .clipShape(cornerShape)
                .overlay { if chosen { cornerShape.strokeBorder(Ink.ring, lineWidth: 2) } }
                .overlay(alignment: .topLeading) {
                    if chosen { CheckSeal().padding(3).stamped(on: chosen) }
                }
            HStack(spacing: 3) {
                if editing { IconView(.pen, size: 11, weight: 2).foregroundStyle(Ink.muted) }
                Text(name).font(.ui(11, 600)).foregroundStyle(Ink.fg).lineLimit(1)
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(editing ? "Rename \(name)" : name)
        .accessibilityAddTraits(chosen ? .isSelected : [])
    }
}

/// What adds a character: a plus and the word, in a character's place.
struct AddCard: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Color.clear
                .aspectRatio(3 / 4, contentMode: .fit)
                .overlay { IconView(.plus, size: 20).foregroundStyle(Ink.muted) }
            Text("Add").font(.ui(11, 600)).foregroundStyle(Ink.muted).lineLimit(1)
        }
        .contentShape(Rectangle())
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Add")
    }
}
