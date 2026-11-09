#!/usr/bin/env bash
# Builds the iOS app into an unsigned build/FaceJutsu.ipa, on Linux (or WSL) or macOS.
# Needs rustup with the aarch64-apple-ios target, xtool with its Darwin Swift SDK, and python3.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
shared="$here/../shared"
build="$here/build"
target=aarch64-apple-ios
# keeps the Rust objects on the same minimum iOS as the app (Package.swift)
export IPHONEOS_DEPLOYMENT_TARGET=26.0
# its own target folder, so a Linux build doesn't fight a Windows or macOS one over the same files
export CARGO_TARGET_DIR="$shared/target/$(uname -s | tr A-Z a-z)"
# source paths relative to the checkout
export CARGO_TARGET_AARCH64_APPLE_IOS_RUSTFLAGS="--remap-path-prefix=$root=src --remap-path-prefix=${CARGO_HOME:-$HOME/.cargo}=cargo --remap-path-prefix=${RUSTUP_HOME:-$HOME/.rustup}=rustup"
# without a system C compiler, use the clang that comes with Swift for builds that run here
if ! command -v cc >/dev/null && command -v clang >/dev/null; then
  export CC=clang
  host=$(rustc -vV | sed -n 's/^host: //p' | tr a-z- A-Z_)
  export "CARGO_TARGET_${host}_LINKER=clang" "CARGO_TARGET_${host}_RUSTFLAGS=-Clink-arg=-fuse-ld=lld"
fi

rm -rf "$build/bindings" "$build/engine" "$build/FaceJutsu.ipa"
mkdir -p "$build/bindings/swift" "$build/bindings/ffi" "$build/engine"

# where the app downloads its models from, their sizes, and the third-party notices
config="$here/Sources/FaceJutsu/Config"
rm -rf "$config"
mkdir -p "$config"
cp "$root/packaging/config.toml" "$root/server/manifest.json" "$config/"
python3 "$here/notices.py" "$config/THIRD-PARTY-NOTICES.txt"

echo "== 1/3 engine"
cargo build --manifest-path "$shared/Cargo.toml" --release --target "$target"
cp "$CARGO_TARGET_DIR/$target/release/libfacejutsu_engine.a" "$build/engine/"

echo "== 2/3 bindings"
# run from the crate, since the generator reads its cargo metadata
(cd "$shared" && cargo run --features bindgen --bin uniffi-bindgen -- \
  generate --library "$build/engine/libfacejutsu_engine.a" --language swift \
  --out-dir "$build/bindings/gen")
mv "$build/bindings/gen/facejutsu_engine.swift" "$build/bindings/swift/"
mv "$build/bindings/gen/facejutsu_engineFFI.h" "$build/bindings/ffi/"
cat > "$build/bindings/ffi/module.modulemap" <<'EOF'
module facejutsu_engineFFI {
    header "facejutsu_engineFFI.h"
    link "facejutsu_engine"
    export *
}
EOF
rm -rf "$build/bindings/gen"

echo "== 3/3 app"
cd "$here"
xtool dev build --configuration release --ipa
ipa=$(find xtool -maxdepth 2 -name '*.ipa' -newer "$build/engine/libfacejutsu_engine.a" | head -1)
[ -n "$ipa" ] || { echo "no .ipa from xtool" >&2; exit 1; }
cp "$ipa" "$build/FaceJutsu.ipa"
echo "built $build/FaceJutsu.ipa"
