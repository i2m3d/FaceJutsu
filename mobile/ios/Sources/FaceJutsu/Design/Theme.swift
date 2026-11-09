import CoreText
import SwiftUI
import UIKit

// Ink & Seal, the desktop app's design system (web/app/style.css), for the phone: the same colours,
// type and motion, with sizes in points taken from the phone layouts. Views use these tokens, not
// their own values.

extension UIColor {
    convenience init(hex: UInt32, alpha: CGFloat = 1) {
        self.init(
            red: CGFloat((hex >> 16) & 0xff) / 255,
            green: CGFloat((hex >> 8) & 0xff) / 255,
            blue: CGFloat(hex & 0xff) / 255,
            alpha: alpha
        )
    }
}

extension Color {
    init(hex: UInt32, alpha: Double = 1) {
        self.init(UIColor(hex: hex, alpha: alpha))
    }
}

/// The colours. Each follows the system's dark or light appearance, as the desktop's
/// prefers-color-scheme does.
enum Ink {
    private static func pair(_ dark: UInt32, _ light: UInt32, alpha: (CGFloat, CGFloat) = (1, 1)) -> Color {
        Color(UIColor { traits in
            traits.userInterfaceStyle == .light
                ? UIColor(hex: light, alpha: alpha.1) : UIColor(hex: dark, alpha: alpha.0)
        })
    }

    /// the backdrop behind the panels
    static let bg = pair(0x0c0b0d, 0xe7e1d6)
    /// the well: the stage, lanes, control tracks
    /// panels and cards
    static let panel = pair(0x17151a, 0xfbf8f3)
    /// a selected header, empty pictures
    static let panel2 = pair(0x221f25, 0xefe9df)
    /// ink hairlines
    static let line = pair(0x332f36, 0xddd5c9)
    static let line2 = pair(0x4a4550, 0xc3baad)
    static let fg = pair(0xf2ede6, 0x1c191e)
    static let muted = pair(0xaba49f, 0x5c565b)
    static let faint = pair(0x8c8585, 0x6d6668)

    /// hanko red: the primary action, progress, the playhead and the seals
    static let accent = Color(hex: 0xd93a22)
    /// the red button's letterpress edge
    static let accentEdge = Color(hex: 0x9b2614)
    static let accentInk = Color.white

    /// a chosen option: a raised paper key
    static let keyBg = pair(0xf2ede6, 0xffffff)
    static let keyFg = pair(0x121014, 0x1c191e)
    static let keyEdge = pair(0xb9b0a3, 0xcfc6b8)
    /// selected: a paper outline
    static let ring = pair(0xf2ede6, 0x1c191e)

    static let good = pair(0x8fbf9f, 0x3f7a52)
    static let warn = pair(0xd6b56e, 0x8a6a1f)
    static let bad = pair(0xec8e83, 0xb3261e)

    /// a pressed control's tint
    static let press = pair(0xf2ede6, 0x1c191e, alpha: (0.12, 0.12))
    /// the ink edge of a face sticker, the same in both themes
    static let stickerInk = Color(hex: 0x121014)
    /// the people's colours, in the order they're found
    static let faces: [Color] = [0xe8b04b, 0x5fb9a6, 0xe97b84, 0x86a9e8, 0xb3d26a].map { Color(hex: $0) }
}

/// Sizes in points.
enum Metric {
    /// the corner of everything seen as an object: buttons, pictures, the film strip, segments, keys
    static let corner: CGFloat = 6
    /// the same corner on small marks: seals, slider knobs
    static let cornerSmall: CGFloat = 3
    /// the edge the red key sits on, and the one under smaller keys
    static let edge: CGFloat = 3, edgeSmall: CGFloat = 2
    /// the smallest a touch target gets
    static let hit: CGFloat = 44
    static let gutter: CGFloat = 12
    static let toolbar: CGFloat = 54
    /// the timeline, and the tool panels that take its place
    static let timeline: CGFloat = 154
}

/// Motion: one curve and three lengths, so everything in the app moves the same way. A press shows at
/// once, and only the before/after cut has no motion at all.
enum Motion {
    /// what changes where it is: a press coming back, a fade, a switch, an icon's move
    static let quick = 0.2
    /// what travels: the paper key, pages, segments, the crop
    static let move = 0.28
    /// the flourishes: the ring of ink, the wink, the poof, a new theme spreading
    static let flourish = 0.6

    /// cubic-bezier(0.3, 0, 0.2, 1)
    static func ease(_ duration: Double = move) -> Animation { .timingCurve(0.3, 0, 0.2, 1, duration: duration) }

    /// The same curve, for what's drawn by hand: how far along at `p`, both 0 to 1.
    static func eased(_ p: CGFloat) -> CGFloat {
        if p <= 0 { return 0 }
        if p >= 1 { return 1 }
        func at(_ t: CGFloat, _ a: CGFloat, _ b: CGFloat) -> CGFloat {
            let u = 1 - t
            return 3 * u * u * t * a + 3 * u * t * t * b + t * t * t
        }
        var low: CGFloat = 0, high: CGFloat = 1
        for _ in 0..<16 {
            let mid = (low + high) / 2
            if at(mid, 0.3, 0.2) < p { low = mid } else { high = mid }
        }
        return at((low + high) / 2, 0, 1)
    }
}

// MARK: - Type

/// The three faces: Instrument Sans for the interface, Bricolage Grotesque for display and
/// overlines, JetBrains Mono for numbers and times. Each is one variable font, bundled with the app.
enum Typeface: String, CaseIterable {
    case ui = "InstrumentSans-Regular"
    case display = "BricolageGrotesque96ptExtraBold-Regular"
    case mono = "JetBrainsMono-Regular"
    /// only the wordmark's dotless j, whose dot is a seal
    case dotlessJ = "BricolageGrotesque96ptExtraBold-Medium"

    fileprivate var file: String {
        switch self {
        case .ui: "instrument-sans-var"
        case .display: "bricolage-grotesque-var"
        case .mono: "jetbrains-mono-var"
        case .dotlessJ: "bricolage-grotesque-dotless-j"
        }
    }

    fileprivate var hasOpticalSize: Bool { self == .display || self == .dotlessJ }
}

enum Fonts {
    private static var registered = false
    private static var cache: [String: UIFont] = [:]
    private static let lock = NSLock()

    /// Registers the bundled fonts with the process. Called once, at launch.
    static func register() {
        lock.lock(); defer { lock.unlock() }
        if registered { return }
        registered = true
        let dir = Bundle.main.bundleURL.appendingPathComponent("FaceJutsu_FaceJutsu.bundle/Fonts")
        for face in Typeface.allCases {
            let url = dir.appendingPathComponent(face.file + ".ttf")
            CTFontManagerRegisterFontsForURL(url as CFURL, .process, nil)
        }
    }

    /// The face at a size and weight, set through the font's own variation axes (and its optical
    /// size, as browsers set it from the size). Falls back to the system font when it's missing.
    static func uiFont(_ face: Typeface, _ size: CGFloat, _ weight: CGFloat) -> UIFont {
        let key = "\(face.rawValue) \(size) \(weight)"
        lock.lock(); defer { lock.unlock() }
        if let font = cache[key] { return font }
        let font: UIFont
        if let base = UIFont(name: face.rawValue, size: size) {
            var axes: [NSNumber: NSNumber] = [tag("wght"): NSNumber(value: Double(weight))]
            if face.hasOpticalSize { axes[tag("opsz")] = NSNumber(value: Double(min(max(size, 12), 96))) }
            let descriptor = base.fontDescriptor.addingAttributes([
                UIFontDescriptor.AttributeName(rawValue: kCTFontVariationAttribute as String): axes,
            ])
            font = UIFont(descriptor: descriptor, size: size)
        } else if face == .mono {
            font = .monospacedSystemFont(ofSize: size, weight: systemWeight(weight))
        } else {
            font = .systemFont(ofSize: size, weight: systemWeight(weight))
        }
        cache[key] = font
        return font
    }

    private static func tag(_ s: String) -> NSNumber {
        NSNumber(value: s.unicodeScalars.reduce(UInt32(0)) { ($0 << 8) | $1.value })
    }

    private static func systemWeight(_ w: CGFloat) -> UIFont.Weight {
        switch w {
        case ..<450: .regular
        case ..<550: .medium
        case ..<650: .semibold
        case ..<750: .bold
        default: .heavy
        }
    }
}

extension Font {
    static func ui(_ size: CGFloat, _ weight: CGFloat = 400) -> Font {
        Font(Fonts.uiFont(.ui, size, weight) as CTFont)
    }

    static func display(_ size: CGFloat, _ weight: CGFloat = 700) -> Font {
        Font(Fonts.uiFont(.display, size, weight) as CTFont)
    }

    static func mono(_ size: CGFloat, _ weight: CGFloat = 500) -> Font {
        Font(Fonts.uiFont(.mono, size, weight) as CTFont)
    }
}

extension View {
    /// The display-font overline: small capitals, spaced out.
    func overline(_ size: CGFloat = 11, color: Color = Ink.fg) -> some View {
        font(.display(size, 700))
            .tracking(size * 0.08)
            .textCase(.uppercase)
            .foregroundStyle(color)
    }
}
