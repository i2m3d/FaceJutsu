import SwiftUI

// The app's moves, all from its own ideas: keys that sink, one key that travels, seals that stamp, ink
// and the poof. How long each takes comes from Motion.

/// Whether taps are felt, from Settings.
enum Feel {
    static var on: Bool { UserDefaults.standard.object(forKey: "haptics") as? Bool ?? true }
}

/// The rectangle of a key on its way between two places, `v` going 1 to 0: the leading edge gets
/// there first and the trailing edge catches up.
private func travel(from: CGRect, to: CGRect, v: CGFloat) -> CGRect {
    let q = Motion.eased(1 - v)
    let fast = min(1, q / 0.5), slow = q < 0.5 ? q / 0.5 * 0.3 : 0.3 + (q - 0.5) / 0.5 * 0.7
    let right = to.minX > from.minX
    let left = from.minX + (to.minX - from.minX) * (right ? slow : fast)
    let rightEdge = from.maxX + (to.maxX - from.maxX) * (right ? fast : slow)
    return CGRect(x: left, y: to.minY, width: rightEdge - left, height: to.height)
}

/// The paper key under a row of choices. One key that moves, so it slides rather than fades. A new
/// `trigger` starts the move from `from` to `to`; with no `from` it pops in.
struct TravellingKey: View {
    let from: CGRect?
    let to: CGRect?
    let trigger: Int

    var body: some View {
        Color.clear.overlay(alignment: .topLeading) {
            if let to {
                KeyframeAnimator(initialValue: CGFloat(0), trigger: trigger) { v in
                    let r = from.map { travel(from: $0, to: to, v: v) } ?? to
                    cornerShape.fill(Ink.keyBg)
                        .background { cornerShape.fill(Ink.keyEdge).offset(y: Metric.edgeSmall) }
                        .frame(width: r.width, height: r.height)
                        .scaleEffect(from == nil ? 1 - 0.14 * v : 1)
                        .offset(x: r.minX, y: r.minY)
                } keyframes: { _ in
                    KeyframeTrack(\.self) {
                        MoveKeyframe(CGFloat(1))
                        LinearKeyframe(CGFloat(0), duration: Motion.move)
                    }
                }
            }
        }
        .allowsHitTesting(false)
    }
}

/// Where each key of a row is.
struct KeyRects: PreferenceKey {
    static let defaultValue: [AnyHashable: CGRect] = [:]
    static func reduce(value: inout [AnyHashable: CGRect], nextValue: () -> [AnyHashable: CGRect]) {
        value.merge(nextValue()) { $1 }
    }
}

extension View {
    /// Reports this key's rectangle in the named row.
    func reportKey(_ id: AnyHashable, in space: String) -> some View {
        background {
            GeometryReader { geo in
                Color.clear.preference(key: KeyRects.self, value: [id: geo.frame(in: .named(space))])
            }
        }
    }
}

private struct StampValue {
    var scale: CGFloat = 1, angle: Double = 0, opacity: Double = 1, ring: CGFloat = 1, ringAlpha: Double = 0
}

/// A seal pressed on: it comes down large and askew and lands, with a small ring of ink.
private struct Stamped<T: Equatable>: ViewModifier {
    let trigger: T
    let ring: Bool

    func body(content: Content) -> some View {
        content.keyframeAnimator(initialValue: StampValue(), trigger: trigger) { view, v in
            view
                .background {
                    if ring { markShape.stroke(Ink.accent, lineWidth: 1.5).scaleEffect(v.ring).opacity(v.ringAlpha) }
                }
                .scaleEffect(v.scale)
                .rotationEffect(.degrees(v.angle))
                .opacity(v.opacity)
        } keyframes: { _ in
            KeyframeTrack(\.scale) {
                MoveKeyframe(1.9)
                CubicKeyframe(0.9, duration: Motion.quick * 0.6)
                CubicKeyframe(1, duration: Motion.quick * 0.4)
            }
            KeyframeTrack(\.angle) {
                MoveKeyframe(-16)
                CubicKeyframe(2, duration: Motion.quick * 0.6)
                CubicKeyframe(0, duration: Motion.quick * 0.4)
            }
            KeyframeTrack(\.opacity) {
                MoveKeyframe(0)
                CubicKeyframe(1, duration: Motion.quick * 0.6)
            }
            KeyframeTrack(\.ring) {
                MoveKeyframe(1)
                LinearKeyframe(1, duration: Motion.quick * 0.5)
                CubicKeyframe(2.1, duration: Motion.flourish - Motion.quick * 0.5)
            }
            KeyframeTrack(\.ringAlpha) {
                MoveKeyframe(0)
                LinearKeyframe(0, duration: Motion.quick * 0.5)
                MoveKeyframe(0.7)
                CubicKeyframe(0, duration: Motion.flourish - Motion.quick * 0.5)
            }
        }
    }
}

/// Coming in with a small move, a little after what's before it.
private struct Rise: ViewModifier {
    let delay: Double
    let dx: CGFloat, dy: CGFloat
    @State private var shown = false
    @Environment(\.accessibilityReduceMotion) private var calm

    func body(content: Content) -> some View {
        content
            .opacity(shown || calm ? 1 : 0)
            .offset(x: shown || calm ? 0 : dx, y: shown || calm ? 0 : dy)
            .onAppear { withAnimation(Motion.ease(Motion.quick).delay(delay)) { shown = true } }
    }
}

/// The small move an icon makes when its tool is used.
enum IconMove {
    case hop, slide, sparkle, tilt, snip, squeeze, nudge, swing(Double), turn(Double), still
}

private struct IconAct<T: Equatable>: ViewModifier {
    let move: IconMove
    let trigger: T

    func body(content: Content) -> some View {
        content.keyframeAnimator(initialValue: CGFloat(1), trigger: trigger) { view, p in
            let beat = sin(p * .pi), wave = sin(p * 2 * .pi)
            switch move {
            case .hop: view.offset(y: -3 * beat)
            case .slide: view.offset(x: 3 * wave)
            case .sparkle: view.rotationEffect(.degrees(24 * beat)).scaleEffect(1 + 0.16 * beat)
            case .tilt: view.rotationEffect(.degrees(9 * wave))
            case .snip: view.scaleEffect(1 - 0.2 * beat)
            case .squeeze: view.scaleEffect(x: 0.72 + 0.28 * p, y: 1)
            case .nudge: view.offset(x: 4 * (1 - p))
            case .swing(let a): view.rotationEffect(.degrees(a * beat))
            case .turn(let a): view.rotationEffect(.degrees(a * wave))
            case .still: view
            }
        } keyframes: { _ in
            KeyframeTrack(\.self) {
                MoveKeyframe(CGFloat(0))
                LinearKeyframe(CGFloat(1), duration: Motion.quick)
            }
        }
    }
}

extension View {
    func stamped<T: Equatable>(on trigger: T, ring: Bool = false) -> some View {
        modifier(Stamped(trigger: trigger, ring: ring))
    }

    /// Comes in after `index` others have started, capped so a long list doesn't lag.
    func rise(_ index: Int = 0, dx: CGFloat = 0, dy: CGFloat = 6, step: Double = 0.02) -> some View {
        modifier(Rise(delay: min(Double(index) * step, 0.06), dx: dx, dy: dy))
    }

    func iconAct<T: Equatable>(_ move: IconMove, on trigger: T) -> some View {
        modifier(IconAct(move: move, trigger: trigger))
    }

    /// A quick pop when `trigger` turns true.
    func popWhen(_ trigger: Bool) -> some View {
        keyframeAnimator(initialValue: CGFloat(1), trigger: trigger) { view, s in
            view.scaleEffect(trigger ? s : 1)
        } keyframes: { _ in
            KeyframeTrack(\.self) {
                MoveKeyframe(CGFloat(0.7))
                CubicKeyframe(CGFloat(1.14), duration: Motion.quick * 0.6)
                CubicKeyframe(CGFloat(1), duration: Motion.quick * 0.4)
            }
        }
    }
}

/// A fade with a small lift. What's leaving can't catch a tap from the moment it starts to go.
struct NoCatchFade: Transition {
    func body(content: Content, phase: TransitionPhase) -> some View {
        content
            .opacity(phase.isIdentity ? 1 : 0)
            .offset(y: phase.isIdentity ? 0 : 8)
            .allowsHitTesting(phase.isIdentity)
    }
}

private struct PoofValue {
    var scale: CGFloat = 0.2, spread: CGFloat = 1, alpha: Double = 0
}

/// The swap landing in a puff of smoke, as on the desktop: circles in two layers, an inked edge under
/// a paper fill, sized to the face.
struct Poof: View {
    let size: CGFloat
    @State private var run = false

    private static let puffs: [(CGFloat, CGFloat, CGFloat)] = [
        (-0.33, -0.08, 0.54), (-0.12, -0.29, 0.58), (0.17, -0.27, 0.52), (0.33, 0, 0.56),
        (0.19, 0.25, 0.54), (-0.1, 0.29, 0.56), (-0.29, 0.19, 0.48), (0, 0, 0.67),
    ]

    var body: some View {
        KeyframeAnimator(initialValue: PoofValue(), trigger: run) { v in
            ZStack {
                layer(v, grow: max(3, size * 0.05), color: Color(hex: 0x16131a))
                layer(v, grow: 0, color: Color(hex: 0xfbf8f3))
            }
            .opacity(v.alpha)
        } keyframes: { _ in
            KeyframeTrack(\.scale) {
                CubicKeyframe(1, duration: 0.21)
                CubicKeyframe(1.18, duration: 0.41)
            }
            KeyframeTrack(\.spread) {
                LinearKeyframe(1, duration: 0.21)
                CubicKeyframe(1.6, duration: 0.41)
            }
            KeyframeTrack(\.alpha) {
                CubicKeyframe(1, duration: 0.075)
                LinearKeyframe(1, duration: 0.21)
                CubicKeyframe(0, duration: 0.335)
            }
        }
        .allowsHitTesting(false)
        .onAppear { run = true }
    }

    private func layer(_ v: PoofValue, grow: CGFloat, color: Color) -> some View {
        ZStack {
            ForEach(Array(Self.puffs.enumerated()), id: \.offset) { _, puff in
                Circle().fill(color)
                    .frame(width: puff.2 * size + grow, height: puff.2 * size + grow)
                    .scaleEffect(v.scale)
                    .offset(x: puff.0 * size * v.spread, y: puff.1 * size * v.spread)
            }
        }
    }
}
