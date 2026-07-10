# Copyright 2026 Michael Zhang
# SPDX-License-Identifier: Apache-2.0
"""Calibrate the workspace-chip panel against neutral text.

Glitch tokens (rare junk tokens with odd unembedding geometry) show up in
J-lens readouts on *any* prompt. This script measures each word's average
mid-band chip score over WikiText prompts; the frontend subtracts it, so only
prompt-specific content survives — TF-IDF for the workspace panel.

    python scripts/make_chip_baseline.py --preset deepseek-coder-1.3b
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jspace_viz.analysis import read_grid  # noqa: E402
from jspace_viz.fitting import load_wikitext_prompts  # noqa: E402
from jspace_viz.lens import JacobianLens  # noqa: E402
from jspace_viz.model import load_model  # noqa: E402
from jspace_viz.presets import PRESETS  # noqa: E402

# Mirrors the scoring in static/app.js renderWorkspace().
WORDLIKE = re.compile(r"^[a-z][a-z'’-]{2,}$", re.IGNORECASE)


def chip_scores(grid_data: dict) -> dict[str, float]:
    vocab = grid_data["vocab"]
    fitted = [r for r in grid_data["grid"] if not r["is_output"]]
    lo, hi = int(len(fitted) * 0.25), -(-int(len(fitted) * 0.85) // 1)
    scores: dict[str, float] = {}
    for row in fitted[lo:hi]:
        for t in range(4, grid_data["seq_len"]):
            for tid, p in zip(row["top_ids"][t], row["top_probs"][t]):
                s = vocab[tid]
                if not (s.startswith(" ") or s.startswith("▁")):
                    continue
                if not WORDLIKE.match(s.strip()) or p < 0.03:
                    continue
                w = s.strip().lower()
                scores[w] = scores.get(w, 0.0) + p
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--n-prompts", type=int, default=24)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    model = load_model(preset.model_id, dtype=args.dtype or preset.dtype, device=args.device)
    kind, *rest = preset.lens
    if kind == "hub":
        lens = JacobianLens.from_pretrained(rest[0], filename=rest[1])
    elif os.path.exists(rest[0]):
        lens = JacobianLens.load(rest[0])
    else:
        repo, filename = preset.lens_fallback
        lens = JacobianLens.from_pretrained(repo, filename=filename)

    totals: dict[str, float] = {}
    prompts = load_wikitext_prompts(args.n_prompts)
    for i, prompt in enumerate(prompts):
        grid_data = read_grid(model, lens, prompt, mode="jlens", top_k=8, max_seq_len=96)
        for word, score in chip_scores(grid_data).items():
            totals[word] = totals.get(word, 0.0) + score
        print(f"prompt {i + 1}/{len(prompts)}", flush=True)

    baseline = {
        w: round(s / len(prompts), 4)
        for w, s in sorted(totals.items(), key=lambda kv: -kv[1])
        if s / len(prompts) >= 0.05
    }
    out = args.out or os.path.join(
        os.path.dirname(__file__), "..", "jspace_viz", "data", "chip_baselines",
        f"{preset.model_id.split('/')[-1]}.json",
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(baseline, fh, ensure_ascii=False, indent=0)
    print(f"wrote {out}: {len(baseline)} words; top:",
          dict(list(baseline.items())[:8]))


if __name__ == "__main__":
    main()
