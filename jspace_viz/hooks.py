# Copyright 2026 Michael Zhang
# SPDX-License-Identifier: Apache-2.0
"""Forward-hook context manager that captures residual-stream tensors."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ActivationRecorder:
    """Record the output hidden state of selected residual blocks.

    Captured tensors are not detached, so they can be used directly as
    ``inputs``/``outputs`` of :func:`torch.autograd.grad`.

    Args:
        blocks: The model's residual blocks (e.g. ``model.layers``).
        at: Block indices to record.
        start_graph_at: If set, the captured tensor at this index gets
            ``requires_grad_(True)``. With all parameters frozen this roots
            the autograd graph at that block's output, so backward passes only
            traverse the layers above it.
    """

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        at: Sequence[int],
        *,
        start_graph_at: int | None = None,
    ) -> None:
        self._blocks = blocks
        self._indices = sorted(set(at) | ({start_graph_at} if start_graph_at is not None else set()))
        self._start_graph_at = start_graph_at
        self.activations: dict[int, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _hook_for(self, index: int):
        root_here = index == self._start_graph_at

        def hook(module: nn.Module, inputs, output) -> None:
            tensor = output if torch.is_tensor(output) else output[0]
            if root_here:
                tensor.requires_grad_(True)
            self.activations[index] = tensor

        return hook

    def __enter__(self) -> ActivationRecorder:
        try:
            for index in self._indices:
                handle = self._blocks[index].register_forward_hook(self._hook_for(index))
                self._handles.append(handle)
        except Exception:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
