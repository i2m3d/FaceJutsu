"""Writes web/app/THIRD-PARTY-NOTICES.txt: the licenses of what the release executable contains.

The Rust libraries come from `cargo metadata` (for every system, so the file is the same whichever
one it's made on), with the license files they ship; then Mediabunny and the fonts from web/app/.
"""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web/app/THIRD-PARTY-NOTICES.txt"
LICENSE_NAMES = ("license", "licence", "copying", "notice", "unlicense")
RULE = "=" * 78


def normal_dependencies(meta):
    """The packages built into the executable: not those only for building or testing it."""
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    packages = {p["id"]: p for p in meta["packages"]}
    seen, todo = set(), [meta["resolve"]["root"]]
    while todo:
        for dep in nodes[todo.pop()]["deps"]:
            if dep["pkg"] not in seen and any(k["kind"] is None for k in dep["dep_kinds"]):
                seen.add(dep["pkg"])
                todo.append(dep["pkg"])
    return sorted((packages[i] for i in seen), key=lambda p: (p["name"], p["version"]))


def license_texts(package):
    folder = Path(package["manifest_path"]).parent
    files = [f for f in sorted(folder.iterdir()) if f.is_file() and f.name.lower().startswith(LICENSE_NAMES)]
    return [f.read_text(encoding="utf-8", errors="replace").strip() for f in files]


def main():
    server = ROOT / "server"
    subprocess.run(["cargo", "fetch", "--locked"], cwd=server, check=True)
    meta = json.loads(subprocess.run(["cargo", "metadata", "--format-version", "1", "--locked"], cwd=server, check=True, capture_output=True, encoding="utf-8").stdout)

    packages = normal_dependencies(meta)
    # (many share one text word for word: it's given once, with all their names)
    by_text, order, listing = {}, [], []
    for p in packages:
        label = f"{p['name']} {p['version']}"
        listing.append(f"{label}: {p.get('license') or 'see its license files'}" + (f" ({p['repository']})" if p.get("repository") else ""))
        for text in license_texts(p):
            key = hashlib.sha256(text.encode()).hexdigest()
            if key not in by_text:
                by_text[key] = {"text": text, "names": []}
                order.append(key)
            by_text[key]["names"].append(label)

    parts = [
        "Third-party notices\n\nFaceJutsu is built with the open source software below. FaceJutsu doesn't include the "
        "models it uses: their licenses are listed in the app, under Storage & about.",
        f"{RULE}\nMediabunny (https://mediabunny.dev)\nCopyright (c) 2026-present, Vanilagy and contributors\n"
        "Used unmodified. Its source: https://github.com/Vanilagy/mediabunny\n"
        f"{RULE}\n\n" + (ROOT / "web/app/vendor/mediabunny.LICENSE").read_text(encoding="utf-8").strip(),
    ]
    for f in sorted((ROOT / "web/app/fonts").glob("OFL-*.txt")):
        parts.append(f"{RULE}\nFont: {f.stem.removeprefix('OFL-')}\n{RULE}\n\n" + f.read_text(encoding="utf-8").strip())
    parts.append(f"{RULE}\nRust libraries\n{RULE}\n\n" + "\n".join(listing))
    for key in order:
        entry = by_text[key]
        parts.append(f"{RULE}\n" + "\n".join(entry["names"]) + f"\n{RULE}\n\n" + entry["text"])

    text = "\n\n\n".join(parts) + "\n"
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"{OUT.relative_to(ROOT)}: {len(packages)} libraries, {len(text.encode()) // 1024} KB")


if __name__ == "__main__":
    main()
