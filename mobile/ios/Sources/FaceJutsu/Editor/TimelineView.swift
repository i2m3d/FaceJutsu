import SwiftUI
import UIKit

private enum Row {
    static let ruler: CGFloat = 24, film: CGFloat = 40, person: CGFloat = 30
    /// the side column at its narrowest: a face
    static let side: CGFloat = 38
    static let faceX: CGFloat = 8, faceSize: CGFloat = 24
    /// the pen, between the face and the name
    static let penX: CGFloat = 34, penWidth: CGFloat = 26
    static let nameX: CGFloat = 62
    /// the side column is wide enough to show names from here
    static let namedFrom: CGFloat = 118
    static var top: CGFloat { ruler + film }
}

/// What the timeline's scrolling and pinching keep. The time across is the playhead itself.
@MainActor
@Observable
final class TimelineViewport {
    var scrollY: CGFloat = 0
    /// points a second
    var pps: CGFloat = 56
    /// how far the playhead's cap leans with the scrubbing, in degrees
    var lean: Double = 0
}

/// What a segment was, to see how it changed.
private struct PartState: Equatable {
    let id: Int, person: Int
    let t0: Double, t1: Double
    let off: Bool, selected: Bool
}

/// How edits move: every segment knows where it was. Filled in when the model changes, read as the
/// timeline draws.
@MainActor
private final class StickerMoves {
    var slides: [Int: (t0: Double, t1: Double, start: Date)] = [:]
    var hops: [Int: Date] = [:]
    var fresh: [Int: Date] = [:]
    var wipes: [Int: Date] = [:]
    var ghosts: [(part: PartState, start: Date)] = []
    var slashes: [(t: Double, person: Int, start: Date)] = []
    var rows: [Int: (from: Int, start: Date)] = [:]
    var end = Date.distantPast

    func take(old: [PartState], new: [PartState], oldRows: [Int], newRows: [Int], split: Bool) {
        let now = Date()
        let before = Dictionary(uniqueKeysWithValues: old.map { ($0.id, $0) })
        let after = Set(new.map(\.id))
        var ripple = 0.0
        for part in new {
            guard let was = before[part.id] else {
                if !old.isEmpty {
                    fresh[part.id] = now
                    if split { slashes.append((part.t0, part.person, now)) }
                }
                continue
            }
            if was.t0 != part.t0 || was.t1 != part.t1 { slides[part.id] = (was.t0, was.t1, now) }
            if part.selected && !was.selected {
                hops[part.id] = now.addingTimeInterval(min(ripple, 0.084))
                ripple += 0.014
            }
            if part.off && !was.off { wipes[part.id] = now }
        }
        for part in old where !after.contains(part.id) { ghosts.append((part, now)) }
        for (i, id) in newRows.enumerated() {
            if let was = oldRows.firstIndex(of: id), was != i { rows[id] = (was, now) }
        }
        end = now.addingTimeInterval(Motion.move + 0.1)
    }

    func clearFinished(_ now: Date) {
        guard now > end else { return }
        slides = [:]; hops = [:]; fresh = [:]; wipes = [:]; ghosts = []; slashes = []; rows = [:]
    }
}

private func progress(_ now: Date, _ start: Date, _ duration: Double) -> CGFloat {
    CGFloat(min(max(now.timeIntervalSince(start) / duration, 0), 1))
}

/// The video's timeline. The playhead stays in the middle and the video moves under it, so scrolling
/// is scrubbing. The ruler and the clip stay at the top as the people scroll up and down, and the
/// faces stay at the side: at the start the side is wide enough for the pen and the name, and it
/// narrows to the faces as the video moves under it.
///
/// It's drawn in one Canvas from the model, under a transparent UIScrollView that does the touches,
/// so scrolling is the system's own.
struct TimelineView: View {
    let model: EditorModel

    @State private var viewport = TimelineViewport()
    /// the person whose name is being edited
    @State private var editing: Int?
    @State private var draft = ""
    @FocusState private var typing: Bool
    @State private var moves = StickerMoves()
    @State private var moving = false
    @Environment(\.colorScheme) private var scheme
    @Environment(\.accessibilityReduceMotion) private var calm

    private var states: [PartState] {
        model.doc.people.flatMap { p in
            var seen = Set<Int>()
            return model.pieces(p).compactMap { piece -> PartState? in
                guard seen.insert(piece.part.id).inserted else { return nil }
                return PartState(id: piece.part.id, person: p.id, t0: piece.t0, t1: piece.t1, off: piece.part.off, selected: model.picked.contains(piece.part.id))
            }
        }
    }

    var body: some View {
        GeometryReader { geo in
            let size = geo.size
            let pps = viewport.pps
            let x = CGFloat(model.playhead) * pps
            let y = viewport.scrollY
            let side = max(Row.side, (size.width / 2 - x - 6).rounded())
            let rows = CGFloat(model.doc.people.count) * Row.person
            ZStack(alignment: .topLeading) {
                SwiftUI.TimelineView(.animation(paused: !moving)) { frame in
                    canvas(now: frame.date, pps: pps, x: x, y: y, side: side)
                }
                TimelineScroller(
                    viewport: viewport,
                    model: model,
                    contentSize: CGSize(width: CGFloat(model.duration) * pps + size.width, height: max(size.height, Row.top + rows)),
                    targetX: x,
                    onTap: { tap($0, width: size.width, x: x, y: y, side: side, pps: pps) },
                    onBeginDrag: { typing = false }
                )
                nameField(y: y, side: side)
                playheadCap(width: size.width)
            }
        }
        .frame(height: Metric.timeline)
        .background(Ink.bg)
        .clipped()
        .accessibilityLabel("Timeline")
        .onChange(of: states) { old, new in
            guard !calm else { return }
            moves.take(old: old, new: new, oldRows: order(old), newRows: order(new), split: model.lastEditKey == "split")
            moving = true
            Task { @MainActor in
                try? await Task.sleep(for: .seconds(Motion.move + 0.12))
                if Date() > moves.end {
                    moves.clearFinished(Date())
                    moving = false
                }
            }
        }
        .onChange(of: typing) { _, now in if !now { commitName() } }
        .onChange(of: model.tool) { _, now in if now != nil { typing = false } }
    }

    private func order(_ parts: [PartState]) -> [Int] {
        var seen: [Int] = []
        for p in parts where !seen.contains(p.person) { seen.append(p.person) }
        return seen
    }

    private func playheadCap(width: CGFloat) -> some View {
        markShape.fill(Ink.accent)
            .frame(width: 10, height: 10)
            .keyframeAnimator(initialValue: CGFloat(1), repeating: model.playing) { view, s in
                view.scaleEffect(model.playing ? s : 1)
            } keyframes: { _ in
                KeyframeTrack(\.self) {
                    CubicKeyframe(CGFloat(1.25), duration: 0.7)
                    CubicKeyframe(CGFloat(1), duration: 0.7)
                }
            }
            .rotationEffect(.degrees(-6 + viewport.lean))
            .animation(Motion.ease(Motion.quick), value: viewport.lean)
            .offset(x: width / 2 - 5)
            .allowsHitTesting(false)
    }

    // MARK: editing a name

    @ViewBuilder
    private func nameField(y: CGFloat, side: CGFloat) -> some View {
        if let editing, let i = model.doc.people.firstIndex(where: { $0.id == editing }) {
            let width = max(40, side - Row.nameX - 6)
            ZStack(alignment: .bottomLeading) {
                TextField("Name", text: $draft)
                    .font(.ui(12, 600))
                    .foregroundStyle(Ink.fg)
                    .tint(Ink.accent)
                    .submitLabel(.done)
                    .focused($typing)
                    .onSubmit { typing = false }
                    .onChange(of: draft) { _, v in if v.count > 24 { draft = String(v.prefix(24)) } }
                InkLine().padding(.bottom, 3)
            }
            .frame(width: width, height: Row.person)
            .background(Ink.bg)
            .position(x: Row.nameX + width / 2, y: Row.top + CGFloat(i) * Row.person - y + Row.person / 2)
            .task(id: editing) { typing = true }
        }
    }

    private func commitName() {
        guard let id = editing else { return }
        editing = nil
        model.rename(person: id, to: draft)
    }

    // MARK: taps

    private func tap(_ p: CGPoint, width: CGFloat, x: CGFloat, y: CGFloat, side: CGFloat, pps: CGFloat) {
        if editing != nil { typing = false; return }
        guard p.y >= Row.ruler else { return }
        let start = width / 2 - x
        if p.y < Row.top {
            // the speaker, where a track has its pen, turns the sound on and off
            if model.hasSound && side >= Row.namedFrom && p.x >= Row.penX && p.x < Row.penX + Row.penWidth {
                model.soundOn.toggle()
                return
            }
            if p.x < side || (p.x >= start && p.x <= start + CGFloat(model.duration) * pps) { model.toggleAll() }
            return
        }
        let i = Int(floor((p.y - Row.top + y) / Row.person))
        guard model.doc.people.indices.contains(i) else { return }
        let person = model.doc.people[i]
        if p.x < side {
            // only the pen renames; the face and the name select the person
            if side >= Row.namedFrom && p.x >= Row.penX && p.x < Row.penX + Row.penWidth {
                draft = person.label
                editing = person.id
            } else {
                model.toggle(person: person.id)
            }
            return
        }
        let t = Double((p.x - start) / pps)
        if let piece = model.pieces(person).first(where: { t >= $0.t0 && t <= $0.t1 }), !piece.part.off {
            model.toggle(part: piece.part.id)
        }
    }

    // MARK: drawing

    private func canvas(now: Date, pps: CGFloat, x: CGFloat, y: CGFloat, side: CGFloat) -> some View {
        // read here, so that a change redraws
        let people = model.doc.people, characters = model.doc.characters
        let picked = model.picked, whole = model.wholeTracks, thumbs = model.thumbs, every = model.stripEvery
        let duration = CGFloat(model.duration), playhead = model.playhead
        let aspect = model.videoSize.width / max(model.videoSize.height, 1)
        let live = people.flatMap { $0.parts.filter { !$0.off }.map(\.id) }
        let pieces = Dictionary(people.map { ($0.id, model.pieces($0)) }, uniquingKeysWith: { a, _ in a })
        let pictures = Dictionary(people.map { ($0.id, model.picture($0)) }, uniquingKeysWith: { a, _ in a })
        let frameStates = model.states, times = model.times, clipName = model.clipName, made = model.made
        let sound: Bool? = model.hasSound ? model.soundOn : nil
        let status = model.scouting.map { "Scouting faces · \(Int($0 * 100))%" }
        let allPicked = !live.isEmpty && Set(live) == picked
        let editing = editing, moves = moves
        let nameOpacity = min(1, max(0, (side - 92) / 40))
        _ = scheme

        return Canvas { ctx, size in
            let w = size.width
            let origin = w / 2 - x
            func screen(_ t: CGFloat) -> CGFloat { origin + t * pps }

            // the ruler: a label fades as the playhead passes over it
            let step = CGFloat([1, 2, 5, 10, 30, 60, 120, 300, 600, 1800].first { CGFloat($0) * pps >= 80 } ?? 3600)
            var t: CGFloat = 0
            while t <= duration + 0.001 {
                let px = screen(t)
                if px > -40 && px < w + 40 {
                    let label = Text(EditorModel.timecode(Double(t))).font(.mono(10, 500)).foregroundColor(Ink.faint)
                    ctx.drawLayer { l in
                        l.opacity = min(1, max(0, (abs(px - w / 2) - 12) / 10))
                        l.draw(l.resolve(label), at: CGPoint(x: px, y: Row.ruler / 2), anchor: .center)
                    }
                }
                let mid = t + step / 2
                if mid <= duration {
                    ctx.fill(Path(ellipseIn: CGRect(x: screen(mid) - 1.5, y: Row.ruler / 2 - 1.5, width: 3, height: 3)), with: .color(Ink.line2))
                }
                t += step
            }

            // what's been looked at, as the desktop's rail: dim, keyframes, every frame, and green
            // where it's already swapped
            if !frameStates.isEmpty, times.count == frameStates.count {
                let y = Row.ruler - 4, x0 = max(0, origin), x1 = min(w, origin + duration * pps)
                if x1 > x0 {
                    ctx.drawLayer { l in
                        l.clip(to: Path(roundedRect: CGRect(x: origin, y: y, width: duration * pps, height: 3), cornerRadius: 1.5))
                        var x = x0, runStart = x0, runState: UInt8 = 255
                        func flush(_ end: CGFloat) {
                            guard runState != 255, end > runStart else { return }
                            let color = runState == 3 ? Ink.good : Ink.fg.opacity([0.1, 0.28, 0.55][Int(min(runState, 2))])
                            l.fill(Path(CGRect(x: runStart, y: y, width: end - runStart, height: 3)), with: .color(color))
                        }
                        while x < x1 {
                            let k = nearestIndex(times, Double((x + 1 - origin) / pps))
                            let st: UInt8 = made.indices.contains(k) && made[k] ? 3 : frameStates[k]
                            if st != runState {
                                flush(x)
                                runStart = x
                                runState = st
                            }
                            x += 2
                        }
                        flush(x1)
                    }
                }
            }

            // the clip's frames, as one strip
            let filmRect = CGRect(x: origin, y: Row.ruler + 2, width: duration * pps, height: Row.film - 4)
            let tw = max(8, filmRect.height * aspect)
            ctx.drawLayer { l in
                l.clip(to: Path(roundedRect: filmRect, cornerRadius: Metric.corner))
                var i = max(0, Int(floor((0 - origin) / tw)))
                while CGFloat(i) * tw + origin < w, CGFloat(i) * tw < duration * pps {
                    if !thumbs.isEmpty {
                        let k = min(thumbs.count - 1, max(0, Int((CGFloat(i) + 0.5) * tw / pps / CGFloat(every))))
                        l.draw(l.resolve(Image(uiImage: thumbs[k])), in: CGRect(x: origin + CGFloat(i) * tw, y: filmRect.minY, width: tw, height: filmRect.height))
                    }
                    i += 1
                }
            }
            if allPicked { ctx.stroke(Path(roundedRect: filmRect.insetBy(dx: 1, dy: 1), cornerRadius: Metric.corner), with: .color(Ink.ring), lineWidth: 2) }

            // the people
            ctx.drawLayer { l in
                l.clip(to: Path(CGRect(x: 0, y: Row.top, width: w, height: size.height - Row.top)))
                for (i, person) in people.enumerated() {
                    var top = Row.top + CGFloat(i) * Row.person - y
                    if let row = moves.rows[person.id] {
                        top += CGFloat(row.from - i) * Row.person * (1 - Motion.eased(progress(now, row.start, Motion.move)))
                    }
                    guard top + Row.person > Row.top, top < size.height else { continue }
                    let color = Ink.faces[person.color % Ink.faces.count]
                    let mine = pieces[person.id] ?? []
                    for piece in mine {
                        let part = piece.part, t0 = piece.t0, t1 = piece.t1
                        let inside = playheadAmount(playhead, t0: t0, t1: t1)
                        Self.drawSticker(&l, rect: CGRect(x: screen(CGFloat(t0)) + 1, y: top + 2, width: max(4, CGFloat(t1 - t0) * pps - 2), height: Row.person - 4),
                                         part: part, length: t1 - t0, character: characters.first { $0.id == part.s.character },
                                         color: color, selected: picked.contains(part.id), side: side, width: w, live: part.off ? 0 : inside, now: now, moves: moves)
                    }
                    for slash in moves.slashes where slash.person == person.id {
                        let p = progress(now, slash.start, Motion.quick)
                        if p < 1 {
                            l.fill(Path(CGRect(x: screen(CGFloat(slash.t)) - 1, y: top, width: 2, height: Row.person)), with: .color(Ink.ring.opacity(Double(1 - p))))
                        }
                    }
                    for ghost in moves.ghosts where ghost.part.person == person.id {
                        let p = progress(now, ghost.start, Motion.quick)
                        if p < 1 {
                            l.fill(Path(roundedRect: CGRect(x: screen(CGFloat(ghost.part.t0)) + 1, y: top + 2, width: CGFloat(ghost.part.t1 - ghost.part.t0) * pps - 2, height: Row.person - 4), cornerRadius: Metric.corner),
                                   with: .color(color.opacity(Double(1 - p))))
                        }
                    }
                    let here = mine.filter { !$0.part.off }.map { playheadAmount(playhead, t0: $0.t0, t1: $0.t1) }.max() ?? 0
                    Self.drawSide(&l, top: top, height: Row.person, side: side, picture: pictures[person.id] ?? nil,
                                  box: nil, name: editing == person.id ? "" : person.label, editable: true,
                                  writing: editing == person.id, nameOpacity: nameOpacity, faceOpacity: 0.5 + 0.5 * here, selected: whole.contains(person.id))
                }
            }

            // the clip's own side, over the frames
            Self.drawSide(&ctx, top: Row.ruler, height: Row.film, side: side, picture: thumbs.first, box: nil, name: clipName,
                          editable: false, writing: false, nameOpacity: nameOpacity, faceOpacity: 1, selected: allPicked, status: status, sound: sound)

            // the playhead (its cap is drawn over this)
            ctx.fill(Path(roundedRect: CGRect(x: w / 2 - 1, y: 0, width: 2, height: size.height), cornerRadius: 1), with: .color(Ink.accent))
        }
    }

    /// 1 inside a segment, easing to 0 just outside it, so what depends on it changes smoothly.
    private func playheadAmount(_ t: Double, t0: Double, t1: Double) -> CGFloat {
        let d = min(t - t0, t1 - t)
        return d >= 0 ? 1 : CGFloat(max(0, 1 + d / 0.15))
    }

    /// A picture filling a rectangle, cut to it. With a box, that part of the picture fills it.
    private static func drawPicture(_ ctx: inout GraphicsContext, _ image: UIImage, in rect: CGRect, box: CGRect? = nil, corner: CGFloat, opacity: CGFloat = 1) {
        let draw: CGRect
        if let box {
            let w = rect.width / box.width, h = rect.height / box.height
            draw = CGRect(x: rect.minX - box.minX * w, y: rect.minY - box.minY * h, width: w, height: h)
        } else {
            let a = image.size.width / max(image.size.height, 1)
            draw = a > rect.width / rect.height
                ? CGRect(x: rect.midX - rect.height * a / 2, y: rect.minY, width: rect.height * a, height: rect.height)
                : CGRect(x: rect.minX, y: rect.midY - rect.width / a / 2, width: rect.width, height: rect.width / a)
        }
        ctx.drawLayer { l in
            l.opacity = opacity
            l.clip(to: Path(roundedRect: rect, cornerRadius: corner))
            l.draw(l.resolve(Image(uiImage: image)), in: draw)
        }
    }

    /// A stretch of a person's time: a sticker in their colour, the character's face and name staying
    /// in view as it scrolls under the side. Deleted, it's hatched.
    private static func drawSticker(_ ctx: inout GraphicsContext, rect base: CGRect, part: Part, length: Double, character: SwapCharacter?,
                                    color: Color, selected: Bool, side: CGFloat, width: CGFloat, live: CGFloat, now: Date, moves: StickerMoves) {
        guard base.maxX > 0, base.minX < width else { return }
        var rect = base
        var alpha: CGFloat = 1
        if let start = moves.fresh[part.id] {
            let p = Motion.eased(progress(now, start, Motion.quick))
            alpha = p
            rect.origin.x -= 6 * (1 - p)
            rect.size.width *= 0.86 + 0.14 * p
        }
        var ringWidth: CGFloat = 2, chip: CGFloat = 1
        if selected, let start = moves.hops[part.id] {
            let p = progress(now, start, Motion.quick), beat = sin(p * .pi)
            rect.origin.y -= 2 * beat
            rect = rect.insetBy(dx: 0, dy: -0.04 * rect.height * beat)
            ringWidth = 2 * Motion.eased(progress(now, start, Motion.quick))
            chip = Motion.eased(progress(now, start, Motion.quick))
        }
        let shape = Path(roundedRect: rect, cornerRadius: Metric.corner)
        ctx.drawLayer { outer in
            outer.opacity = alpha
            if part.off {
                let wipe = moves.wipes[part.id].map { Motion.eased(progress(now, $0, Motion.move)) } ?? 1
                outer.drawLayer { l in
                    l.clip(to: Path(CGRect(x: rect.minX, y: rect.minY, width: rect.width * wipe, height: rect.height)))
                    l.clip(to: shape)
                    l.fill(shape, with: .color(Ink.panel))
                    var hatch = Path()
                    var x = rect.minX - rect.height
                    while x < rect.maxX {
                        hatch.move(to: CGPoint(x: x, y: rect.maxY))
                        hatch.addLine(to: CGPoint(x: x + rect.height, y: rect.minY))
                        x += 8.5
                    }
                    l.stroke(hatch, with: .color(Ink.panel2), lineWidth: 4)
                }
                if wipe < 1 {
                    outer.drawLayer { l in
                        l.clip(to: Path(CGRect(x: rect.minX + rect.width * wipe, y: rect.minY, width: rect.width * (1 - wipe), height: rect.height)))
                        l.fill(shape, with: .color(color))
                    }
                }
                return
            }
            outer.fill(shape, with: .color(color))
            if live > 0 { outer.fill(shape, with: .color(.white.opacity(0.1 * Double(live)))) }
            if selected {
                outer.stroke(Path(roundedRect: rect.insetBy(dx: 1, dy: 1), cornerRadius: Metric.corner - 1), with: .color(Ink.ring), lineWidth: ringWidth)
            }
            guard character != nil || selected else { return }
            outer.drawLayer { l in
                l.clip(to: shape)
                let av: CGFloat = 20
                let name = character.map { l.resolve(Text($0.name).font(.ui(11, 700)).foregroundColor(Ink.stickerInk)) }
                let nameW = name?.measure(in: CGSize(width: 200, height: 40)).width ?? 0
                let time = l.resolve(Text(String(format: "%.1fs", length)).font(.mono(10, 600)).foregroundColor(Color(hex: 0xf2ede6)))
                let timeW = selected ? time.measure(in: CGSize(width: 80, height: 40)).width + 6 : 0
                let total = (character != nil ? av + 4 + nameW : 0) + (selected ? (character != nil ? 4 : 0) + timeW : 0)
                // stays clear of the side while there's room in the sticker
                var x = max(rect.minX + 5, min(side + 5, rect.maxX - 5 - total))
                let mid = rect.midY
                if let character, let image = character.image {
                    drawPicture(&l, image, in: CGRect(x: x, y: mid - av / 2, width: av, height: av), corner: Metric.cornerSmall)
                }
                if let name {
                    l.draw(name, at: CGPoint(x: x + av + 4, y: mid), anchor: .leading)
                    x += av + 4 + nameW + 4
                }
                if selected && chip > 0 {
                    l.drawLayer { c in
                        c.opacity = chip
                        c.fill(Path(roundedRect: CGRect(x: x, y: mid - 8, width: timeW, height: 16), cornerRadius: Metric.cornerSmall),
                               with: .color(Color(hex: 0x121014, alpha: 0.78)))
                        c.draw(time, at: CGPoint(x: x + timeW / 2, y: mid), anchor: .center)
                    }
                }
            }
        }
    }

    private static let penBody = SVGPath.parse("M8.6 18.6 18.7 8.5a1.9 1.9 0 0 0 0-2.7l-.5-.5a1.9 1.9 0 0 0-2.7 0L5.4 15.4")
    private static let penCollar = SVGPath.parse("M13.8 7l3.2 3.2")
    private static let penTip = SVGPath.parse("M3.7 20.3l1.3-5.3 4 4z")
    private static let speaker = SVGPath.parse("M4 9.5h3.5L12 6v12l-4.5-3.5H4z")
    private static let speakerOn = SVGPath.parse("M15.5 9a4 4 0 0 1 0 6M18 6.5a7.5 7.5 0 0 1 0 11")
    private static let speakerOff = SVGPath.parse("M16 9.5l5 5M21 9.5l-5 5")

    /// The side of a row: its face and, with room, the pen and name. The face dims while that person
    /// isn't on screen.
    private static func drawSide(_ ctx: inout GraphicsContext, top: CGFloat, height: CGFloat, side: CGFloat, picture: UIImage?, box: CGRect?,
                                 name: String, editable: Bool, writing: Bool, nameOpacity: CGFloat, faceOpacity: CGFloat, selected: Bool, status: String? = nil, sound: Bool? = nil) {
        let rect = CGRect(x: 0, y: top, width: side, height: height)
        ctx.fill(Path(rect), with: .color(Ink.bg))
        if let picture {
            drawPicture(&ctx, picture, in: CGRect(x: Row.faceX, y: rect.midY - Row.faceSize / 2, width: Row.faceSize, height: Row.faceSize),
                        box: box, corner: Metric.corner, opacity: faceOpacity)
        }
        if nameOpacity > 0 {
            ctx.drawLayer { l in
                l.clip(to: Path(CGRect(x: Row.penX, y: top, width: max(0, side - Row.penX - 4), height: height)))
                l.opacity = nameOpacity
                let textX = editable || sound != nil ? Row.nameX : Row.penX
                if let sound {
                    let k: CGFloat = 16 / 24
                    let t = CGAffineTransform(translationX: Row.penX + (Row.penWidth - 16) / 2, y: rect.midY - 8).scaledBy(x: k, y: k)
                    let stroke = StrokeStyle(lineWidth: 2 * k, lineCap: .round, lineJoin: .round)
                    l.stroke(speaker.applying(t), with: .color(Ink.muted), style: stroke)
                    l.stroke((sound ? speakerOn : speakerOff).applying(t), with: .color(sound ? Ink.muted : Ink.accent), style: stroke)
                }
                if editable {
                    let k: CGFloat = 14 / 24
                    let t = CGAffineTransform(translationX: Row.penX + (Row.penWidth - 14) / 2, y: rect.midY - 7).scaledBy(x: k, y: k)
                    let stroke = StrokeStyle(lineWidth: 2 * k, lineCap: .round, lineJoin: .round)
                    let ink = writing ? Ink.fg : Ink.muted
                    l.stroke(penBody.applying(t), with: .color(ink), style: stroke)
                    l.stroke(penCollar.applying(t), with: .color(ink), style: stroke)
                    l.fill(penTip.applying(t), with: .color(Ink.accent))
                }
                if !name.isEmpty {
                    let y = status == nil ? rect.midY : rect.midY - 6
                    l.draw(l.resolve(Text(name).font(.ui(12, 600)).foregroundColor(Ink.fg)), at: CGPoint(x: textX, y: y), anchor: .leading)
                }
                if let status {
                    l.draw(l.resolve(Text(status).font(.ui(10, 500)).foregroundColor(Ink.faint)), at: CGPoint(x: textX, y: rect.midY + 7), anchor: .leading)
                }
            }
        }
        if selected {
            ctx.stroke(Path(roundedRect: rect.insetBy(dx: 1, dy: 1), cornerRadius: Metric.corner), with: .color(Ink.ring), lineWidth: 2)
        }
    }
}

/// A red line of ink that draws itself under what's being typed.
private struct InkLine: View {
    @State private var drawn = false

    var body: some View {
        RoundedRectangle(cornerRadius: 1).fill(Ink.accent)
            .frame(height: 2)
            .scaleEffect(x: drawn ? 1 : 0, anchor: .leading)
            .onAppear { withAnimation(Motion.ease(Motion.move)) { drawn = true } }
            .allowsHitTesting(false)
    }
}

/// A transparent system scroll view that takes the timeline's touches: its offset across is the
/// playhead, its offset down scrolls the people, taps come back as points, and a pinch zooms time
/// around the playhead.
private struct TimelineScroller: UIViewRepresentable {
    let viewport: TimelineViewport
    let model: EditorModel
    let contentSize: CGSize
    let targetX: CGFloat
    let onTap: (CGPoint) -> Void
    let onBeginDrag: () -> Void

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIView(context: Context) -> UIScrollView {
        let s = UIScrollView()
        s.backgroundColor = .clear
        s.delegate = context.coordinator
        s.isDirectionalLockEnabled = true
        s.showsHorizontalScrollIndicator = false
        s.showsVerticalScrollIndicator = false
        s.bounces = false
        s.contentInsetAdjustmentBehavior = .never
        // two fingers are a pinch
        s.panGestureRecognizer.maximumNumberOfTouches = 1
        let tap = UITapGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.tapped(_:)))
        tap.cancelsTouchesInView = false
        s.addGestureRecognizer(tap)
        s.addGestureRecognizer(UIPinchGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.pinched(_:))))
        return s
    }

    func updateUIView(_ s: UIScrollView, context: Context) {
        let c = context.coordinator
        c.parent = self
        if s.contentSize != contentSize { s.contentSize = contentSize }
        // follows the playhead, unless a finger or a glide is what's moving it
        guard !s.isTracking, !s.isDecelerating, abs(s.contentOffset.x - targetX) > 0.5 else { return }
        c.quiet = true
        s.contentOffset.x = targetX
        c.quiet = false
        c.lastX = targetX
    }

    @MainActor
    final class Coordinator: NSObject, UIScrollViewDelegate {
        var parent: TimelineScroller
        /// the playhead moved the scroll view, so there's nothing to report back
        var quiet = false
        var lastX: CGFloat = 0
        private var pinchFrom: CGFloat = 56
        private var settle: Task<Void, Never>?

        init(_ parent: TimelineScroller) { self.parent = parent }

        func scrollViewWillBeginDragging(_ s: UIScrollView) {
            if parent.model.playing { parent.model.stop() }
            parent.onBeginDrag()
        }

        func scrollViewDidScroll(_ s: UIScrollView) {
            guard !quiet else { return }
            if parent.viewport.scrollY != s.contentOffset.y { parent.viewport.scrollY = s.contentOffset.y }
            guard s.isTracking || s.isDecelerating else { return }
            parent.model.seek(Double(s.contentOffset.x / parent.viewport.pps))
            let dx = s.contentOffset.x - lastX
            lastX = s.contentOffset.x
            parent.viewport.lean = Double(max(-18, min(18, -dx * 0.7)))
            settle?.cancel()
            settle = Task { @MainActor [weak self] in
                try? await Task.sleep(for: .milliseconds(90))
                if !Task.isCancelled { self?.parent.viewport.lean = 0 }
            }
        }

        @objc func tapped(_ g: UITapGestureRecognizer) {
            guard let s = g.view as? UIScrollView else { return }
            let p = g.location(in: s)
            parent.onTap(CGPoint(x: p.x - s.contentOffset.x, y: p.y - s.contentOffset.y))
        }

        @objc func pinched(_ g: UIPinchGestureRecognizer) {
            guard let s = g.view as? UIScrollView else { return }
            switch g.state {
            case .began:
                pinchFrom = parent.viewport.pps
            case .changed:
                // out to the whole video between the faces at the sides
                let whole = (s.bounds.width - 2 * Row.side) / max(CGFloat(parent.model.duration), 0.1)
                let next = min(max(pinchFrom * g.scale, min(24, whole)), 240)
                let t = CGFloat(parent.model.playhead)
                parent.viewport.pps = next
                quiet = true
                s.contentSize.width = CGFloat(parent.model.duration) * next + s.bounds.width
                s.contentOffset.x = t * next
                quiet = false
            default:
                break
            }
        }
    }
}
