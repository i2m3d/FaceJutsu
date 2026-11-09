import SwiftUI

@main
struct FaceJutsuApp: App {
    @AppStorage("appearance") private var appearance = Appearance.system

    init() {
        Fonts.register()
    }

    var body: some Scene {
        WindowGroup {
            EditorView()
                .tint(Ink.accent)
                .onAppear { appearance.apply() }
                .onChange(of: appearance) { _, now in now.apply() }
        }
    }
}
