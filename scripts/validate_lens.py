# Copyright 2026 Michael Zhang
# SPDX-License-Identifier: Apache-2.0
"""Validate this repo's Jacobian estimator against a reference lens.

Fits a small lens locally and reports the per-layer cosine similarity of the
resulting ``J_l`` matrices against a lens fit with the official
anthropics/jacobian-lens code (via the Neuronpedia collection on the Hub).
Cosines should rise toward ~1.0 in later layers even with few prompts.

    python scripts/validate_lens.py            # pythia-70m, 8 prompts
    python scripts/validate_lens.py --n-prompts 32
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from jspace_viz.fitting import fit, load_wikitext_prompts  # noqa: E402
from jspace_viz.lens import JacobianLens  # noqa: E402
from jspace_viz.model import load_model  # noqa: E402

REF_REPO = "neuronpedia/jacobian-lens"
REF_FILE = "pythia-70m-deduped/jlens/Salesforce-wikitext/pythia-70m-deduped_jacobian_lens.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--ref-repo", default=REF_REPO)
    parser.add_argument("--ref-file", default=REF_FILE)
    parser.add_argument("--n-prompts", type=int, default=8)
    parser.add_argument("--dim-batch", type=int, default=64)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    model = load_model(args.model_id, dtype="float32")
    mine = fit(model, load_wikitext_prompts(args.n_prompts), dim_batch=args.dim_batch)
    ref = JacobianLens.from_pretrained(args.ref_repo, filename=args.ref_file)

    print(f"\ncos(J_l ours [{mine.n_prompts} prompts], J_l reference [{ref.n_prompts} prompts]):")
    for layer in mine.source_layers:
        cos = torch.nn.functional.cosine_similarity(
            mine.jacobians[layer].flatten(), ref.jacobians[layer].flatten(), dim=0
        ).item()
        print(f"  layer {layer}: {cos:.4f}")


if __name__ == "__main__":
    main()
