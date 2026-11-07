#!/usr/bin/env bash
# Builds the release executable: packaging/build-release.sh [target triple]
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)

python "$root/packaging/notices.py"

target=${1:-}
cd "$root/server"
cargo build --release --locked ${target:+--target "$target"}

out="target/${target:+$target/}release"
exe="$out/FaceJutsu"
[ -f "$exe.exe" ] && exe="$exe.exe"
echo "Built $exe."
