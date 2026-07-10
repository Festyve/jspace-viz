# Copyright 2026 Michael Zhang
# SPDX-License-Identifier: Apache-2.0
"""Export a serverless demo site to docs/ (GitHub Pages ready).

Precomputes the full readout grid — including per-token rank tables, so
click-to-trace works client-side — for each example prompt in both lens modes,
then copies the frontend next to the JSON. The frontend auto-detects the
missing API and switches to static mode.

    python scripts/export_static.py --preset gpt2
    python -m http.server -d docs 8400   # local check
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jspace_viz.analysis import read_grid  # noqa: E402
from jspace_viz.lens import JacobianLens  # noqa: E402
from jspace_viz.model import load_model  # noqa: E402
from jspace_viz.presets import EXAMPLES, PRESETS  # noqa: E402
from jspace_viz.server import STATIC_DIR  # noqa: E402


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="gpt2")
    parser.add_argument("--out", default="docs")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--device", default=None, help="cpu to avoid contending with a running fit")
    parser.add_argument("--dtype", default=None)
    parser.add_argument(
        "--link", action="append", default=[],
        help="'label=url' chips linking to sibling demo pages; repeatable",
    )
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    model = load_model(preset.model_id, dtype=args.dtype or preset.dtype, device=args.device)
    kind, *rest = preset.lens
    lens = (
        JacobianLens.from_pretrained(rest[0], filename=rest[1])
        if kind == "hub"
        else JacobianLens.load(rest[0])
    )

    data_dir = os.path.join(args.out, "data")
    os.makedirs(data_dir, exist_ok=True)
    index = {
        "model_id": preset.model_id,
        "n_layers": model.n_layers,
        "d_model": model.d_model,
        "fitted_layers": lens.source_layers,
        "lens_n_prompts": lens.n_prompts,
        "links": [
            {"name": name, "url": url}
            for name, url in (spec.split("=", 1) for spec in args.link)
        ],
        "examples": [],
    }
    for example in EXAMPLES:
        slug = slugify(example["name"])
        index["examples"].append({"name": example["name"], "prompt": example["prompt"], "slug": slug})
        for mode in ("jlens", "logit"):
            grid = read_grid(
                model, lens, example["prompt"],
                mode=mode, top_k=args.top_k, max_seq_len=192, track_all=True,
            )
            path = os.path.join(data_dir, f"{slug}_{mode}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(grid, fh, ensure_ascii=False, separators=(",", ":"))
            print(f"wrote {path} ({os.path.getsize(path) // 1024} KB)")

    with open(os.path.join(data_dir, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False)
    os.makedirs(os.path.join(args.out, "static"), exist_ok=True)
    shutil.copy(os.path.join(STATIC_DIR, "index.html"), os.path.join(args.out, "index.html"))
    for name in ("app.js", "app.css"):
        shutil.copy(os.path.join(STATIC_DIR, name), os.path.join(args.out, "static", name))
    print(f"done — serve locally with: python -m http.server -d {args.out} 8400")


if __name__ == "__main__":
    main()
