"""Builds the static site from web/ and site/, into out/site by default: minified files to upload to
any static host.

    python site/build.py [--out <folder>]
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
SITE = ROOT / "site"
ESBUILD = "esbuild@0.25.12"

# Where the page says something that's only true of the app with its server: (file, old, new). Each
# must be found, so a change in the page can't quietly leave its wording behind.
WORDING = [
    ("index.html", 'title="Show it in the outputs folder">Show in folder', 'title="Download it again">Download again'),
    ("index.html", "Open models folder", "Remove the models"),
    ("index.html", "Your finished photos and videos.", "Your finished photos and videos go to your browser's downloads, or where you choose with Save as…."),
    ("app/main.js", "Each photo with a face given a character (the others are left out)", "Each photo with a face given a character, together in one zip (the others are left out)"),
    ("index.html", "<title>FaceJutsu</title>", '<title>FaceJutsu</title>\n<meta name="referrer" content="no-referrer">'),
    ("app/main.js", "The local server isn't answering", "FaceJutsu can't start here"),
    ("app/main.js", "Start FaceJutsu again, then reload this page. (${e.message})", "It keeps its models in this browser's storage, which a private window may not allow. Try a normal window. (${e.message})"),
    ("app/main.js", "The files without a Download have no address in config.toml, next to the app.", "The files without a Download can't be fetched right now."),
]

# Only there for the server: the buttons that show a folder.
STYLE = "\n#modelsOpen,#openOutputsBtn{display:none!important}\n"


def apply_wording(out):
    for name, old, new in WORDING:
        path = out / name
        text = path.read_text(encoding="utf-8")
        if old not in text:
            sys.exit(f"{name} no longer says: {old}")
        path.write_text(text.replace(old, new), encoding="utf-8")


def minify_code(out):
    """Every script and the stylesheet, in place. Not bundled: the engine's worker is its own file."""
    scripts = [p.relative_to(out).as_posix() for p in (out / "app").rglob("*.js")] + ["app/style.css"]
    subprocess.run(
        ["npx", "--yes", ESBUILD, *scripts, "--minify", "--format=esm", "--target=es2022", "--keep-names", "--legal-comments=none", "--charset=utf8", "--outdir=.", "--outbase=.", "--allow-overwrite", "--log-level=warning"],
        cwd=out, check=True, shell=sys.platform == "win32",
    )


def minify_page(out):
    path = out / "index.html"
    text = re.sub(r"<!--.*?-->", "", path.read_text(encoding="utf-8"), flags=re.S)
    text = re.sub(r"\n\s*\n+", "\n", text)
    path.write_text(text, encoding="utf-8")


def model_files(out):
    config = tomllib.loads((ROOT / "packaging/config.toml").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "server/manifest.json").read_text(encoding="utf-8"))
    (out / "models").mkdir()
    (out / "models/urls.json").write_text(json.dumps(config["model_urls"], separators=(",", ":")), encoding="utf-8")
    (out / "models/manifest.json").write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "out/site")
    out = parser.parse_args().out.resolve()
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(WEB, out)
    shutil.copy(SITE / "workspace.js", out / "app/workspace.js")
    shutil.copy(SITE / "zip.js", out / "app/zip.js")
    with (out / "app/style.css").open("a", encoding="utf-8") as css:
        css.write(STYLE)
    apply_wording(out)
    minify_code(out)
    minify_page(out)
    model_files(out)
    files = [p for p in out.rglob("*") if p.is_file()]
    print(f"{out}: {len(files)} files, {sum(p.stat().st_size for p in files) / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
