# Copyright 2026 Michael Zhang
# SPDX-License-Identifier: Apache-2.0
"""Fit a Jacobian lens for a model.

Examples:
    # DeepSeek on an Apple-Silicon laptop (hours; checkpointed + resumable,
    # and --partial writes a usable lens after every prompt):
    python scripts/fit_lens.py --preset deepseek-coder-1.3b --n-prompts 50

    # Any HF model:
    python scripts/fit_lens.py --model-id EleutherAI/pythia-70m-deduped \
        --n-prompts 100 --out lenses/pythia-70m.pt
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jspace_viz.fitting import fit, load_wikitext_prompts  # noqa: E402
from jspace_viz.model import load_model  # noqa: E402
from jspace_viz.presets import PRESETS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default=None, help="preset name (sets model + output path)")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--out", default=None, help="output lens .pt path")
    parser.add_argument("--n-prompts", type=int, default=100)
    parser.add_argument("--dim-batch", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--dtype", default=None, choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--prompts-file", default=None, help="newline-delimited corpus override (default: WikiText-103 stream)")
    args = parser.parse_args()

    model_id, out, dtype = args.model_id, args.out, args.dtype
    if args.preset:
        preset = PRESETS[args.preset]
        model_id = model_id or preset.model_id
        dtype = dtype or preset.dtype
        if out is None and preset.lens[0] == "file":
            out = preset.lens[1]
    if not model_id:
        raise SystemExit("pass --preset or --model-id")
    if out is None:
        out = f"lenses/{model_id.split('/')[-1]}_jlens_wikitext.pt"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as fh:
            prompts = [line for line in fh.read().split("\n\n") if len(line.strip()) > 200]
        prompts = prompts[: args.n_prompts]
    else:
        prompts = load_wikitext_prompts(args.n_prompts)
    logging.info("corpus: %d prompts", len(prompts))

    model = load_model(model_id, dtype=dtype or "auto")
    logging.info("model %s on %s: %d layers, d_model=%d", model_id, model.device, model.n_layers, model.d_model)

    lens = fit(
        model,
        prompts,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        checkpoint_path=out + ".ckpt",
        save_partial_to=out,
    )
    lens.save(out)
    logging.info("saved %r to %s", lens, out)


if __name__ == "__main__":
    main()
