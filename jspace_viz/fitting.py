# Copyright 2026 Michael Zhang
# SPDX-License-Identifier: Apache-2.0
"""Fitting the Jacobian lens.

Estimator (following the paper and the reference implementation): for one
prompt, ``J_l`` rows are obtained by injecting a one-hot cotangent at output
dimension ``i`` — set at *every valid target position at once* — and reading
the gradient at each source position, then averaging over source positions:

    J_l[i, :] = mean_p  sum_{p' >= p}  d h_final[p', i] / d h_l[p, :]

The prompt is replicated ``dim_batch`` times along the batch axis so each
backward pass yields ``dim_batch`` rows. The first ``SKIP_FIRST_N_POSITIONS``
positions (attention sinks) and the final position are excluded. Per-prompt
Jacobians are averaged over the corpus.

Cost: one forward and ``ceil(d_model / dim_batch)`` backward passes per
prompt; total backward FLOPs scale with ``d_model``, so fitting a 1.3B model
takes minutes per prompt on a laptop. Quality saturates quickly with corpus
size (paper §9.3): ~10 prompts already beats the logit lens, ~100 is usable,
the paper uses 1000.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Sequence

import torch

from jspace_viz.hooks import ActivationRecorder
from jspace_viz.lens import JacobianLens
from jspace_viz.model import WrappedModel

logger = logging.getLogger(__name__)

#: Early positions act as attention sinks with atypical residual statistics.
SKIP_FIRST_N_POSITIONS = 16


def valid_position_mask(seq_len: int, *, skip_first: int = SKIP_FIRST_N_POSITIONS) -> torch.Tensor:
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[skip_first : seq_len - 1] = True
    if not mask.any():
        raise ValueError(f"prompt too short: seq_len={seq_len}, need > {skip_first + 1}")
    return mask


def jacobian_for_prompt(
    model: WrappedModel,
    prompt: str,
    source_layers: Sequence[int],
    *,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
) -> tuple[dict[int, torch.Tensor], int, int]:
    """Per-layer Jacobian estimate for one prompt.

    Returns ``(jacobians, seq_len, n_valid_positions)`` with each ``J_l`` a
    ``[d_model, d_model]`` fp32 CPU tensor. The target is the final block's
    output (the final-layer residual stream, before the final norm).
    """
    d_model = model.d_model
    target_layer = model.n_layers - 1
    source_layers = sorted(set(source_layers))
    if source_layers[0] < 0 or source_layers[-1] >= target_layer:
        raise ValueError(f"source layers must lie in [0, {target_layer - 1}]")

    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]
    position_mask = valid_position_mask(seq_len, skip_first=skip_first)
    n_valid = int(position_mask.sum())

    jacobians = {
        layer: torch.zeros(d_model, d_model, dtype=torch.float32)
        for layer in source_layers
    }
    n_passes = math.ceil(d_model / dim_batch)

    with (
        ActivationRecorder(
            model.layers,
            at=[*source_layers, target_layer],
            start_graph_at=source_layers[0],
        ) as recorder,
        torch.enable_grad(),
    ):
        model.forward(input_ids.expand(dim_batch, -1))
        target_act = recorder.activations[target_layer]  # [dim_batch, seq, d]
        source_acts = [recorder.activations[layer] for layer in source_layers]

        valid_pos = position_mask.nonzero(as_tuple=True)[0].to(target_act.device)
        batch_idx = torch.arange(dim_batch, device=target_act.device)
        cotangent = torch.zeros_like(target_act)

        for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
            n_dims = min(dim_batch, d_model - dim_start)
            cotangent.zero_()
            cotangent[
                batch_idx[:n_dims, None],
                valid_pos[None, :],
                dim_start + batch_idx[:n_dims, None],
            ] = 1.0
            grads = torch.autograd.grad(
                outputs=target_act,
                inputs=source_acts,
                grad_outputs=cotangent,
                retain_graph=pass_idx < n_passes - 1,
            )
            for layer, grad in zip(source_layers, grads, strict=True):
                rows = grad[:n_dims, valid_pos.to(grad.device), :].float().mean(dim=1)
                jacobians[layer][dim_start : dim_start + n_dims, :] = rows.cpu()
            del grads

    return jacobians, seq_len, n_valid


def _atomic_save(obj: object, path: str) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def fit(
    model: WrappedModel,
    prompts: Sequence[str],
    *,
    source_layers: Sequence[int] | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    checkpoint_path: str | None = None,
    save_partial_to: str | None = None,
) -> JacobianLens:
    """Fit ``J_l`` over ``prompts`` (running mean of per-prompt Jacobians).

    Args:
        checkpoint_path: If set, a resumable running-sum checkpoint is written
            after every prompt and resumed from automatically.
        save_partial_to: If set, a *usable* lens file is also written there
            after every prompt, so a long fit can be visualized while it runs.
    """
    if source_layers is None:
        source_layers = list(range(model.n_layers - 1))
    source_layers = sorted(set(source_layers))

    jacobian_sum = {
        layer: torch.zeros(model.d_model, model.d_model, dtype=torch.float32)
        for layer in source_layers
    }
    n_done, next_idx = 0, 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if state["source_layers"] != source_layers or state["skip_first"] != skip_first:
            raise ValueError(f"checkpoint {checkpoint_path} was fit with other settings")
        jacobian_sum, n_done, next_idx = state["jacobian_sum"], state["n_done"], state["next_idx"]
        logger.info("resuming: %d/%d prompts done", next_idx, len(prompts))

    def write_state() -> None:
        if checkpoint_path:
            _atomic_save(
                {
                    "jacobian_sum": jacobian_sum,
                    "n_done": n_done,
                    "next_idx": next_idx,
                    "source_layers": source_layers,
                    "skip_first": skip_first,
                },
                checkpoint_path,
            )
        if save_partial_to and n_done > 0:
            lens = JacobianLens(
                {l: jacobian_sum[l] / n_done for l in source_layers},
                n_prompts=n_done,
                d_model=model.d_model,
            )
            tmp = f"{save_partial_to}.tmp.{os.getpid()}"
            lens.save(tmp)
            os.replace(tmp, save_partial_to)

    for prompt_idx, prompt in enumerate(prompts):
        if prompt_idx < next_idx:
            continue
        t0 = time.perf_counter()
        try:
            per_prompt, seq_len, n_valid = jacobian_for_prompt(
                model,
                prompt,
                source_layers,
                dim_batch=dim_batch,
                max_seq_len=max_seq_len,
                skip_first=skip_first,
            )
        except ValueError as exc:
            logger.warning("skipping prompt %d: %s", prompt_idx, exc)
            next_idx = prompt_idx + 1
            continue
        # Relative shift of the running mean; falls ~1/n once converged.
        if n_done > 0:
            rel_change = max(
                (
                    (per_prompt[l] - jacobian_sum[l] / n_done).norm()
                    / ((n_done + 1) * (jacobian_sum[l] / n_done).norm())
                ).item()
                for l in source_layers
            )
        else:
            rel_change = float("nan")
        for layer in source_layers:
            jacobian_sum[layer] += per_prompt[layer]
        n_done += 1
        next_idx = prompt_idx + 1
        logger.info(
            "prompt %d/%d  seq_len=%d n_valid=%d  %.0fs  rel_change=%.2e",
            prompt_idx + 1, len(prompts), seq_len, n_valid,
            time.perf_counter() - t0, rel_change,
        )
        write_state()

    if n_done == 0:
        raise ValueError("no prompts were long enough to fit on")
    return JacobianLens(
        {l: jacobian_sum[l] / n_done for l in source_layers},
        n_prompts=n_done,
        d_model=model.d_model,
    )


def load_wikitext_prompts(n_prompts: int, *, min_chars: int = 600) -> list[str]:
    """First ``n_prompts`` WikiText-103 train records >= ``min_chars`` chars,
    streamed from the Hub (requires ``datasets``). Same corpus convention as
    the Neuronpedia community lenses."""
    from datasets import load_dataset

    stream = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True
    )
    prompts: list[str] = []
    for record in stream:
        if len(record["text"].strip()) >= min_chars:
            prompts.append(record["text"])
            if len(prompts) == n_prompts:
                break
    return prompts
