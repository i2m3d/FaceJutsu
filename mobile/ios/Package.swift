// swift-tools-version: 6.0

import PackageDescription

// The app as a Swift package, built into an .ipa with xtool (see build.sh). The engine's Rust
// library and its generated bindings are written to build/ by build.sh before the package builds.
let package = Package(
    name: "FaceJutsu",
    platforms: [
        // Metal 4's tensor ops
        .iOS("26.0"),
        .macOS(.v14),
    ],
    products: [
        // xtool builds the app from the package's one library product
        .library(name: "FaceJutsu", targets: ["FaceJutsu"]),
    ],
    targets: [
        .target(
            name: "FaceJutsu",
            dependencies: ["FaceJutsuEngine"],
            // Fonts: the desktop's, as TrueType. Config (copied in by build.sh): where the models
            // are downloaded from, and the third-party notices
            resources: [.copy("Fonts"), .copy("Config")],
            // (source paths relative to the package)
            swiftSettings: [.unsafeFlags(["-file-prefix-map", "\(Context.packageDirectory)=."])],
            linkerSettings: [.unsafeFlags(["-Xlinker", "-S", "-Xlinker", "-x"])]
        ),
        // the generated Swift side of the engine bindings
        .target(
            name: "FaceJutsuEngine",
            dependencies: ["facejutsu_engineFFI"],
            path: "build/bindings/swift",
            swiftSettings: [.unsafeFlags(["-file-prefix-map", "\(Context.packageDirectory)=."])],
            linkerSettings: [
                .unsafeFlags(["-L\(Context.packageDirectory)/build/engine"]),
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
            ]
        ),
        // the C header of the engine bindings, linking the Rust static library
        .systemLibrary(
            name: "facejutsu_engineFFI",
            path: "build/bindings/ffi"
        ),
    ],
    swiftLanguageModes: [.v5]
)
