# Copyright 2026 Michael Zhang
# SPDX-License-Identifier: Apache-2.0
"""jspace-viz: live J-space / Jacobian-lens visualizer for open-weights LMs.

Independent implementation of the Jacobian lens from "Verbalizable
Representations Form a Global Workspace in Language Models" (Anthropic, 2026,
https://transformer-circuits.pub/2026/workspace/index.html).

The estimator and the on-disk lens format follow the official reference
implementation (https://github.com/anthropics/jacobian-lens, Apache-2.0), so
lens files are interchangeable: lenses fitted here load there and vice versa,
including the prebaked community lenses on the HuggingFace Hub (e.g.
``neuronpedia/jacobian-lens``).
"""

from jspace_viz.fitting import fit, jacobian_for_prompt
from jspace_viz.lens import JacobianLens
from jspace_viz.model import WrappedModel, load_model

__all__ = [
    "JacobianLens",
    "WrappedModel",
    "fit",
    "jacobian_for_prompt",
    "load_model",
]
