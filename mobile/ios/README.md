# ios

The iPhone app: a SwiftUI front end for the engine in `../shared`. It's a Swift package built with
[xtool](https://github.com/xtool-org/xtool), so it builds on Linux (or WSL) as well as macOS. The
build produces an unsigned `.ipa`, which users install with SideStore. It needs iOS 26 or later.

The app doesn't include the models. It downloads them, from the addresses in
`packaging/config.toml`, or takes them from Files.

## Building

Needs rustup with the iOS target (`rustup target add aarch64-apple-ios`), xtool set up with its
Darwin Swift SDK (`xtool setup`), and python3. Then:

```
bash mobile/ios/build.sh
```

It builds the engine as a static library for `aarch64-apple-ios`, generates its Swift bindings into
`build/bindings`, and builds the app with `xtool dev build --ipa` into `build/FaceJutsu.ipa`.

## Releasing

```
bash mobile/ios/release.sh
```

From a committed checkout, it writes the versioned `.ipa`, the third-party notices, what it was
built from and the SHA-256 sums to `build/release/`. The version and build number are in
`Info.plist`.

- `Package.swift`: the app target, plus two targets for the engine bindings, which point into `build/`.
- `xtool.yml`, `Info.plist`: the bundle ID, name, version and the app's settings.
- `Sources/FaceJutsu/`: the app.
