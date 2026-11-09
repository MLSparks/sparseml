"""Assemble the website into one folder for GitHub Pages.

    python site/build.py --out _site [--stack-data DIR]

Copies site/ as is, plus the sparkrun sources the in-browser pages run
(sparkrun/py/sparkrun/*.py and sparkrun/py/trip.py), plus optional fresh
Stack Explorer data. That's all: no bundler, no templates.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
SKIP = {"build.py", "README.md", "google-sheet", "__pycache__"}


def _ignore(folder: str, names: list[str]) -> list[str]:
    top = Path(folder) == SITE
    return [n for n in names if n == "__pycache__" or (top and n in SKIP)]


def build(out: Path, stack_data: Path | None = None) -> Path:
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(SITE, out, ignore=_ignore)
    py = out / "sparkrun" / "py"
    (py / "sparkrun").mkdir(parents=True)
    sources = sorted((ROOT / "sparkrun" / "sparkrun").glob("*.py"))
    for f in sources:
        shutil.copy2(f, py / "sparkrun" / f.name)
    (py / "files.json").write_text(json.dumps([f.name for f in sources]))  # what the browser pages load
    shutil.copy2(ROOT / "sparkrun" / "examples" / "trip.py", py / "trip.py")
    if stack_data and stack_data.exists():
        shutil.copytree(stack_data, out / "stack" / "data", dirs_exist_ok=True)
    (out / ".nojekyll").touch()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "_site")
    ap.add_argument("--stack-data", type=Path)
    a = ap.parse_args()
    print(f"built {build(a.out, a.stack_data)}")
