import SwiftUI

// The app's line icons and its mark, drawn from the same SVG paths as the desktop, so they keep one
// stroke weight on a 24-point grid.

/// An SVG path (`d`) read into a Path: M L H V C S Q T A Z, absolute and relative.
enum SVGPath {
    static func parse(_ d: String) -> Path {
        var p = Path()
        var s = Scanner(d)
        var cmd: Character = "M"
        var cur = CGPoint.zero, start = CGPoint.zero
        var lastCtrl: CGPoint?   // the last cubic's second control point, for S
        var lastQuad: CGPoint?   // the last quadratic's control point, for T
        while true {
            s.skipSeparators()
            guard let c = s.peek() else { break }
            if c.isLetter {
                cmd = c
                s.advance()
                if c == "Z" || c == "z" {
                    p.closeSubpath()
                    cur = start
                    lastCtrl = nil; lastQuad = nil
                    continue
                }
            }
            let rel = cmd.isLowercase
            func pt() -> CGPoint? {
                guard let x = s.number(), let y = s.number() else { return nil }
                return rel ? CGPoint(x: cur.x + x, y: cur.y + y) : CGPoint(x: x, y: y)
            }
            switch cmd.uppercased().first! {
            case "M":
                guard let q = pt() else { return p }
                p.move(to: q); cur = q; start = q
                cmd = rel ? "l" : "L"   // further pairs are lines
                lastCtrl = nil; lastQuad = nil
            case "L":
                guard let q = pt() else { return p }
                p.addLine(to: q); cur = q
                lastCtrl = nil; lastQuad = nil
            case "H":
                guard let x = s.number() else { return p }
                cur = CGPoint(x: rel ? cur.x + x : x, y: cur.y)
                p.addLine(to: cur)
                lastCtrl = nil; lastQuad = nil
            case "V":
                guard let y = s.number() else { return p }
                cur = CGPoint(x: cur.x, y: rel ? cur.y + y : y)
                p.addLine(to: cur)
                lastCtrl = nil; lastQuad = nil
            case "C":
                guard let c1 = pt(), let c2 = pt(), let q = pt() else { return p }
                p.addCurve(to: q, control1: c1, control2: c2)
                cur = q; lastCtrl = c2; lastQuad = nil
            case "S":
                let c1 = lastCtrl.map { CGPoint(x: 2 * cur.x - $0.x, y: 2 * cur.y - $0.y) } ?? cur
                guard let c2 = pt(), let q = pt() else { return p }
                p.addCurve(to: q, control1: c1, control2: c2)
                cur = q; lastCtrl = c2; lastQuad = nil
            case "Q":
                guard let c1 = pt(), let q = pt() else { return p }
                p.addQuadCurve(to: q, control: c1)
                cur = q; lastQuad = c1; lastCtrl = nil
            case "T":
                let c1 = lastQuad.map { CGPoint(x: 2 * cur.x - $0.x, y: 2 * cur.y - $0.y) } ?? cur
                guard let q = pt() else { return p }
                p.addQuadCurve(to: q, control: c1)
                cur = q; lastQuad = c1; lastCtrl = nil
            case "A":
                guard let rx = s.number(), let ry = s.number(), let rot = s.number(),
                      let large = s.flag(), let sweep = s.flag(), let q = pt() else { return p }
                arc(&p, from: cur, to: q, rx: rx, ry: ry, rotation: rot, large: large, sweep: sweep)
                cur = q; lastCtrl = nil; lastQuad = nil
            default:
                return p
            }
        }
        return p
    }

    /// An SVG arc (endpoint form) as cubic curves, at most a quarter turn each.
    private static func arc(_ p: inout Path, from a: CGPoint, to b: CGPoint, rx rx0: CGFloat, ry ry0: CGFloat,
                            rotation: CGFloat, large: Bool, sweep: Bool) {
        if a == b { return }
        var rx = abs(rx0), ry = abs(ry0)
        if rx == 0 || ry == 0 { p.addLine(to: b); return }
        let phi = rotation * .pi / 180
        let cosP = cos(phi), sinP = sin(phi)
        let dx = (a.x - b.x) / 2, dy = (a.y - b.y) / 2
        let x1 = cosP * dx + sinP * dy, y1 = -sinP * dx + cosP * dy
        let lambda = (x1 * x1) / (rx * rx) + (y1 * y1) / (ry * ry)
        if lambda > 1 { rx *= lambda.squareRoot(); ry *= lambda.squareRoot() }
        let num = rx * rx * ry * ry - rx * rx * y1 * y1 - ry * ry * x1 * x1
        let den = rx * rx * y1 * y1 + ry * ry * x1 * x1
        var k = (max(0, num / den)).squareRoot()
        if large == sweep { k = -k }
        let cx1 = k * rx * y1 / ry, cy1 = -k * ry * x1 / rx
        let cx = cosP * cx1 - sinP * cy1 + (a.x + b.x) / 2
        let cy = sinP * cx1 + cosP * cy1 + (a.y + b.y) / 2
        func angle(_ ux: CGFloat, _ uy: CGFloat, _ vx: CGFloat, _ vy: CGFloat) -> CGFloat {
            let sign: CGFloat = ux * vy - uy * vx < 0 ? -1 : 1
            let dot = ux * vx + uy * vy
            let len = (ux * ux + uy * uy).squareRoot() * (vx * vx + vy * vy).squareRoot()
            return sign * acos(min(1, max(-1, dot / len)))
        }
        let t1 = angle(1, 0, (x1 - cx1) / rx, (y1 - cy1) / ry)
        var dt = angle((x1 - cx1) / rx, (y1 - cy1) / ry, (-x1 - cx1) / rx, (-y1 - cy1) / ry)
        if !sweep && dt > 0 { dt -= 2 * .pi }
        if sweep && dt < 0 { dt += 2 * .pi }
        let n = Int(ceil(abs(dt) / (.pi / 2)))
        let step = dt / CGFloat(n)
        let alpha = 4 / 3 * tan(step / 4)
        func point(_ t: CGFloat) -> CGPoint {
            let x = rx * cos(t), y = ry * sin(t)
            return CGPoint(x: cosP * x - sinP * y + cx, y: sinP * x + cosP * y + cy)
        }
        func deriv(_ t: CGFloat) -> CGPoint {
            let x = -rx * sin(t), y = ry * cos(t)
            return CGPoint(x: cosP * x - sinP * y, y: sinP * x + cosP * y)
        }
        var t = t1
        for i in 0..<n {
            let t2 = t + step
            let p1 = point(t), p2 = i == n - 1 ? b : point(t2)
            let d1 = deriv(t), d2 = deriv(t2)
            p.addCurve(to: p2,
                       control1: CGPoint(x: p1.x + alpha * d1.x, y: p1.y + alpha * d1.y),
                       control2: CGPoint(x: p2.x - alpha * d2.x, y: p2.y - alpha * d2.y))
            t = t2
        }
    }

    /// Reads numbers and flags out of path data.
    private struct Scanner {
        let chars: [Character]
        var i = 0

        init(_ s: String) { chars = Array(s) }

        func peek() -> Character? { i < chars.count ? chars[i] : nil }
        mutating func advance() { i += 1 }

        mutating func skipSeparators() {
            while i < chars.count, chars[i] == " " || chars[i] == "," || chars[i] == "\n" || chars[i] == "\t" { i += 1 }
        }

        mutating func flag() -> Bool? {
            skipSeparators()
            guard let c = peek(), c == "0" || c == "1" else { return nil }
            i += 1
            return c == "1"
        }

        mutating func number() -> CGFloat? {
            skipSeparators()
            var j = i
            var text = ""
            if j < chars.count, chars[j] == "-" || chars[j] == "+" { text.append(chars[j]); j += 1 }
            var dot = false, digits = false
            while j < chars.count {
                let c = chars[j]
                if c.isNumber { digits = true; text.append(c); j += 1 }
                else if c == "." && !dot { dot = true; text.append(c); j += 1 }
                else if (c == "e" || c == "E") && digits {
                    text.append(c); j += 1
                    if j < chars.count, chars[j] == "-" || chars[j] == "+" { text.append(chars[j]); j += 1 }
                } else { break }
            }
            guard digits, let v = Double(text) else { return nil }
            i = j
            return CGFloat(v)
        }
    }
}

/// A path drawn in a square view box, scaled to fit the frame and centred.
struct SVGShape: Shape {
    let path: Path
    let box: CGFloat

    init(_ d: String, box: CGFloat = 24) {
        path = SVGPath.parse(d)
        self.box = box
    }

    func path(in rect: CGRect) -> Path {
        let k = min(rect.width, rect.height) / box
        let t = CGAffineTransform(translationX: rect.midX - box * k / 2, y: rect.midY - box * k / 2).scaledBy(x: k, y: k)
        return path.applying(t)
    }
}

/// The line icons. Each is drawn on a 24-point grid with one stroke weight; a few are filled.
enum Icon: String, CaseIterable {
    case undo, redo, gear, play, pause, mute, fit, minus, plus, export, person, swap, enhance, mask, close
    case split, join, merge, trash, check, chevronDown, chevronLeft, chevronRight, pen, compare, deselect, back
    case film, image, share, warn, stop

    fileprivate var strokes: [String] {
        switch self {
        case .undo: ["M9 14 4 9l5-5", "M4 9h10.5a5.5 5.5 0 0 1 0 11H11"]
        case .redo: ["m15 14 5-5-5-5", "M20 9H9.5a5.5 5.5 0 0 0 0 11H13"]
        case .gear: ["M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Z",
                     "M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1Z"]
        case .play, .pause: []
        case .mute: ["M4 9.5h3.5L12 6v12l-4.5-3.5H4z", "M15.5 9a4 4 0 0 1 0 6M18 6.5a7.5 7.5 0 0 1 0 11"]
        case .fit: ["M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"]
        case .minus: ["M6 12h12"]
        case .plus: ["M12 6v12M6 12h12"]
        case .export: ["M12 4v11M7 10.5l5 5 5-5M5 19.5h14"]
        case .person: ["M15.75 8a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0Z", "M4.75 20.25a7.25 7.25 0 0 1 14.5 0"]
        case .swap: ["M4 8.5h13.5M14 5l3.5 3.5L14 12M20 15.5H6.5M10 12l-3.5 3.5L10 19"]
        case .enhance: ["M9.5 4.5c.45 3.7 1.8 5.05 5.5 5.5-3.7.45-5.05 1.8-5.5 5.5-.45-3.7-1.8-5.05-5.5-5.5 3.7-.45 5.05-1.8 5.5-5.5Z",
                        "M18.5 15v4M16.5 17h4"]
        case .mask: ["M4.5 9.25c0-2.75 2.1-4.5 4.75-4.5h5.5c2.65 0 4.75 1.75 4.75 4.5v3.5c0 3.9-3.25 6.75-7.5 6.75S4.5 16.65 4.5 12.75z",
                     "M8.25 11.25h2.5M13.25 11.25h2.5"]
        case .close: ["M6.5 6.5l11 11M17.5 6.5l-11 11"]
        case .split: ["M9 6a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z", "M9 18a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z",
                      "M20 4 8.1 15.9M14.5 14.5 20 20M8.1 8.1 12 12"]
        case .join: ["M4 12h6M14 12h6M8 8l4 4-4 4M16 8l-4 4 4 4"]
        case .merge: ["M4 6h3c4 0 5 6 9 6h4M4 18h3c4 0 5-6 9-6M17 9l3 3-3 3"]
        case .trash: ["M4 7h16M10 11v6M14 11v6M6 7l1 12a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1l1-12M9 7V4h6v3"]
        case .check: ["M5 12.5l4.5 4.5L19 7.5"]
        case .chevronDown: ["M7 10l5 5 5-5"]
        case .chevronLeft: ["M15 6l-6 6 6 6"]
        case .chevronRight: ["M9 6l6 6-6 6"]
        case .pen: ["M8.6 18.6 18.7 8.5a1.9 1.9 0 0 0 0-2.7l-.5-.5a1.9 1.9 0 0 0-2.7 0L5.4 15.4", "M13.8 7l3.2 3.2"]
        case .compare: ["M6 5h12a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2Z", "M12 3v18"]
        case .deselect: ["M9.6 9.6l4.8 4.8M14.4 9.6l-4.8 4.8"]
        case .back: ["M12 6l-6 6 6 6M18 6l-6 6 6 6"]
        case .film: ["M5.5 5h13a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2Z", "M8 5v14M16 5v14M3.5 9.5H8M3.5 14.5H8M16 9.5h4.5M16 14.5h4.5"]
        case .image: ["M5.5 5h13a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2Z", "M10.6 10a1.6 1.6 0 1 1-3.2 0 1.6 1.6 0 0 1 3.2 0Z", "m4 17 5-4.5 3.5 3 3-2.5 4.5 4"]
        case .share: ["M12 15V4M8 7.5 12 3.5l4 4", "M6 11v8a1.5 1.5 0 0 0 1.5 1.5h9A1.5 1.5 0 0 0 18 19v-8"]
        case .warn: ["M12 4 2.8 19.5h18.4z", "M12 10v4.5"]
        case .stop: []
        }
    }

    fileprivate var fills: [String] {
        switch self {
        case .play: ["M8 5.5v13l10-6.5z"]
        case .pause: ["M8 5.5h2.5a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1H8a1 1 0 0 1-1-1v-11a1 1 0 0 1 1-1Z",
                      "M14.5 5.5H17a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1h-2.5a1 1 0 0 1-1-1v-11a1 1 0 0 1 1-1Z"]
        case .compare: ["M6 5h6v14H6a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2Z"]
        case .warn: ["M13 17a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z"]
        case .stop: ["M6.5 5h11A1.5 1.5 0 0 1 19 6.5v11a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 5 17.5v-11A1.5 1.5 0 0 1 6.5 5Z"]
        default: []
        }
    }

    /// what's filled red
    fileprivate var accents: [String] { self == .pen ? ["M3.7 20.3l1.3-5.3 4 4z"] : [] }

    /// what's drawn dashed
    fileprivate var dashes: [String] {
        self == .deselect ? ["M6.5 4.5h11a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2h-11a2 2 0 0 1-2-2v-11a2 2 0 0 1 2-2Z"] : []
    }
}

/// An icon at a size, in the current foreground colour. The stroke is given on the 24-point grid,
/// as the SVGs give it, and scales with the icon.
struct IconView: View {
    let icon: Icon
    var size: CGFloat = 20
    var weight: CGFloat = 1.9

    init(_ icon: Icon, size: CGFloat = 20, weight: CGFloat = 1.9) {
        self.icon = icon
        self.size = size
        self.weight = weight
    }

    var body: some View {
        ZStack {
            // one stroke for the whole icon, so where its lines cross stays one line
            if !icon.strokes.isEmpty {
                SVGShape(icon.strokes.joined(separator: " "))
                    .stroke(style: StrokeStyle(lineWidth: weight * size / 24, lineCap: .round, lineJoin: .round))
            }
            ForEach(Array(icon.fills.enumerated()), id: \.offset) { _, d in
                SVGShape(d).fill()
            }
            ForEach(Array(icon.accents.enumerated()), id: \.offset) { _, d in
                SVGShape(d).fill(Ink.accent)
            }
            ForEach(Array(icon.dashes.enumerated()), id: \.offset) { _, d in
                let k = size / 24
                SVGShape(d).stroke(style: StrokeStyle(lineWidth: weight * k, lineCap: .round, lineJoin: .round, dash: [3 * k, 3.2 * k]))
            }
        }
        .frame(width: size, height: size)
        .compositingGroup()
        .accessibilityHidden(true)
    }
}

/// The mark: the wink mask with two red flicks. Its eye slit takes the surface it sits on. A new
/// `wink` value makes it wink.
struct Mark: View {
    var size: CGFloat = 24
    var paper: Color = Ink.bg
    var wink: Int = 0

    var body: some View {
        KeyframeAnimator(initialValue: CGFloat(1), trigger: wink) { p in
            let beat = sin(p * .pi)
            ZStack {
                ZStack {
                    SVGShape("M49 19C54 15 57.5 10.5 60.5 5C60.5 12 57.5 18 52.5 22.5Z", box: 64).fill(Ink.accent)
                    SVGShape("M52.5 24.5C57 24 60.5 22 63 18.5C62.5 24 59 27.5 54 28.5Z", box: 64).fill(Ink.accent)
                }
                .rotationEffect(.degrees(-16 * beat), anchor: UnitPoint(x: 0.77, y: 0.36))
                SVGShape("M56 35a25 25 0 1 1-50 0 25 25 0 0 1 50 0Z", box: 64).fill()
                SVGShape("M17.25 26h27.5a7.75 7.75 0 0 1 0 15.5h-27.5a7.75 7.75 0 0 1 0-15.5Z", box: 64).fill(paper)
                SVGShape("M26.2 33.8a3.7 3.7 0 1 1-7.4 0 3.7 3.7 0 0 1 7.4 0Z", box: 64).fill()
                    .scaleEffect(y: 1 - 0.9 * beat, anchor: UnitPoint(x: 22.5 / 64, y: 33.8 / 64))
                SVGShape("M34 35.2Q39 29.4 44 35.2", box: 64)
                    .stroke(style: StrokeStyle(lineWidth: 3.6 * size / 64, lineCap: .round))
            }
        } keyframes: { _ in
            KeyframeTrack(\.self) {
                MoveKeyframe(CGFloat(0))
                LinearKeyframe(CGFloat(1), duration: Motion.flourish)
            }
        }
        .frame(width: size, height: size)
        .accessibilityHidden(true)
    }
}

/// The wordmark: lowercase, the j's dot a red seal.
struct Wordmark: View {
    var size: CGFloat = 17

    var body: some View {
        // The dot sits 0.125em into the j and 0.86em above the baseline, as on the desktop; the
        // Text's frame starts 0.93em (Bricolage's ascent) above the baseline.
        HStack(alignment: .firstTextBaseline, spacing: 0) {
            Text("face")
            Text("\u{0237}")
                .font(Font(Fonts.uiFont(.dotlessJ, size, 800) as CTFont))
                .overlay(alignment: .topLeading) {
                    RoundedRectangle(cornerRadius: size * 0.02)
                        .fill(Ink.accent)
                        .frame(width: size * 0.17, height: size * 0.17)
                        .rotationEffect(.degrees(-4))
                        .offset(x: size * 0.125, y: size * 0.07)
                }
            Text("utsu")
        }
        .font(.display(size, 800))
        .tracking(-0.04 * size)
        .fixedSize()
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("FaceJutsu")
    }
}
