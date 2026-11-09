# Mobile

The iPhone app. It runs the face swap on the phone's own GPU, not in a browser.

- `shared/`: the engine, written in Rust. The app uses it through generated bindings.
- `ios/`: the SwiftUI app, as a Swift package. Built with xtool, on Linux or macOS, into an unsigned
  `.ipa`.
