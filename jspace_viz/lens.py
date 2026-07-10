# Copyright 2026 Michael Zhang
# SPDX-License-Identifier: Apache-2.0
"""The fitted Jacobian lens: per-layer transport matrices ``J_l``.

``lens_l(h) = unembed(J_l @ h)`` where ``J_l = E[dh_final / dh_l]`` averaged
over a text corpus (see :mod:`jspace_viz.fitting`).

The on-disk format (keys ``J`` / ``n_prompts`` / ``source_layers`` /
``d_model``) matches the reference implementation at
https://github.com/anthropics/jacobian-lens so lens files are interchangeable.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch


class JacobianLens:
    """Per-layer ``J_l`` matrices plus save/load/merge/transport."""

    def __init__(
        self,
        jacobians: dict[int, torch.Tensor],
        *,
        n_prompts: int,
        d_model: int,
    ) -> None:
        self.jacobians = {int(layer): J.float() for layer, J in jacobians.items()}
        self.source_layers = sorted(self.jacobians)
        self.n_prompts = n_prompts
        self.d_model = d_model

    def __repr__(self) -> str:
        lo, hi = self.source_layers[0], self.source_layers[-1]
        return (
            f"JacobianLens(d_model={self.d_model}, n_prompts={self.n_prompts}, "
            f"layers=[{lo}..{hi}] ({len(self.source_layers)}))"
        )

    def save(self, path: str, *, dtype: torch.dtype = torch.float16) -> None:
        torch.save(
            {
                "J": {layer: J.to(dtype) for layer, J in self.jacobians.items()},
                "n_prompts": self.n_prompts,
                "source_layers": self.source_layers,
                "d_model": self.d_model,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> JacobianLens:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if "J" not in state:
            raise ValueError(
                f"{path} does not look like a lens file (keys: {sorted(state)})"
            )
        return cls(
            jacobians=state["J"],
            n_prompts=state["n_prompts"],
            d_model=state["d_model"],
        )

    @classmethod
    def from_pretrained(
        cls, name_or_path: str, *, filename: str = "lens.pt"
    ) -> JacobianLens:
        """Load from a local file, a local directory, or a HF Hub repo id."""
        if os.path.isfile(name_or_path):
            return cls.load(name_or_path)
        if os.path.isdir(name_or_path):
            return cls.load(os.path.join(name_or_path, filename))
        from huggingface_hub import hf_hub_download

        return cls.load(hf_hub_download(name_or_path, filename))

    @classmethod
    def merge(cls, lenses: Sequence[JacobianLens]) -> JacobianLens:
        """``n_prompts``-weighted mean of lenses fitted on disjoint prompt sets."""
        if not lenses:
            raise ValueError("merge() needs at least one lens")
        first = lenses[0]
        for other in lenses[1:]:
            if other.source_layers != first.source_layers or other.d_model != first.d_model:
                raise ValueError("lenses disagree on source_layers / d_model")
        n_total = sum(lens.n_prompts for lens in lenses)
        merged = {
            layer: sum(lens.jacobians[layer] * lens.n_prompts for lens in lenses) / n_total
            for layer in first.source_layers
        }
        return cls(jacobians=merged, n_prompts=n_total, d_model=first.d_model)

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        """``J_l @ h`` for a residual of shape ``[..., d_model]``."""
        J = self.jacobians[layer].to(residual.device, residual.dtype)
        return residual @ J.T
