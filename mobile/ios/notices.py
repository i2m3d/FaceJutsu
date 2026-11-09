"""Writes the licenses of the Rust libraries in the iOS app to the file given."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
spec = importlib.util.spec_from_file_location("notices", ROOT / "packaging/notices.py")
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def main(out):
    shared = ROOT / "mobile/shared"
    run = lambda *a: subprocess.run(["cargo", *a], cwd=shared, check=True, capture_output=True, encoding="utf-8").stdout
    run("fetch", "--locked")
    meta = json.loads(run("metadata", "--format-version", "1", "--locked", "--filter-platform", "aarch64-apple-ios"))
    packages = common.normal_dependencies(meta)
    by_text, order, listing = {}, [], []
    for p in packages:
        label = f"{p['name']} {p['version']}"
        listing.append(f"{label}: {p.get('license') or 'see its license files'}" + (f" ({p['repository']})" if p.get("repository") else ""))
        for text in common.license_texts(p):
            key = hashlib.sha256(text.encode()).hexdigest()
            if key not in by_text:
                by_text[key] = {"text": text, "names": []}
                order.append(key)
            by_text[key]["names"].append(label)
    rule = common.RULE
    parts = ["Third-party notices\n\nFaceJutsu for iOS is built with the open source software below. Its fonts' licenses "
             "come with them, in the app's Fonts folder. FaceJutsu doesn't include the models it uses: their licenses are "
             "listed in the app, under Settings, Licenses."]
    parts.append(f"{rule}\nRust libraries\n{rule}\n\n" + "\n".join(listing))
    for key in order:
        parts.append(f"{rule}\n" + "\n".join(by_text[key]["names"]) + f"\n{rule}\n\n" + by_text[key]["text"])
    Path(out).write_text("\n\n\n".join(parts) + "\n", encoding="utf-8", newline="\n")
    print(f"notices: {len(packages)} libraries")


if __name__ == "__main__":
    main(sys.argv[1])
