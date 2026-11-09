#!/usr/bin/env bash
# Makes a release of the iOS app from a committed checkout, in build/release:
#   FaceJutsu-iOS-<version>.ipa, THIRD-PARTY-NOTICES.txt, BUILD-INFO.txt and SHA256SUMS.
# The version and build number come from Info.plist.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
git=(git -c safe.directory='*' -C "$root")

if [ -n "$("${git[@]}" status --porcelain)" ]; then
  echo "uncommitted changes: a release is built from a commit" >&2
  exit 1
fi

value() { sed -n "/<key>$1<\/key>/{n;s/.*<string>\(.*\)<\/string>.*/\1/p;}" "$here/Info.plist" | tr -d '\r'; }
version=$(value CFBundleShortVersionString)
number=$(value CFBundleVersion)
[ -n "$version" ] && [ -n "$number" ] || { echo "no version in Info.plist" >&2; exit 1; }

# (the archive's times are in UTC, whatever this machine's time zone)
export TZ=UTC
export SOURCE_DATE_EPOCH=$("${git[@]}" log -1 --format=%ct)
bash "$here/build.sh"

out="$here/build/release"
ipa="FaceJutsu-iOS-$version.ipa"
rm -rf "$out"
mkdir -p "$out"
cp "$here/build/FaceJutsu.ipa" "$out/$ipa"
cp "$here/Sources/FaceJutsu/Config/THIRD-PARTY-NOTICES.txt" "$out/"

cat > "$out/BUILD-INFO.txt" <<EOF
FaceJutsu for iOS $version (build $number)
Commit: $("${git[@]}" rev-parse HEAD)
Minimum: iOS 26
Unsigned: install with SideStore
$(rustc --version)
$(swift --version 2>&1 | head -1)
$(xtool --version)
EOF

(cd "$out" && sha256sum "$ipa" THIRD-PARTY-NOTICES.txt BUILD-INFO.txt > SHA256SUMS)
cat "$out/SHA256SUMS"
echo "released $out/$ipa"
