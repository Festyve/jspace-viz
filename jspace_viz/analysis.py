# Copyright 2026 Michael Zhang
# SPDX-License-Identifier: Apache-2.0
"""Computing the layer x position readout grid and workspace-band metrics."""

from __future__ import annotations

from typing import Any

import torch

from jspace_viz.hooks import ActivationRecorder
from jspace_viz.lens import JacobianLens
from jspace_viz.model import WrappedModel

#: Leading positions excluded from the per-layer summary metrics (attention
#: sinks distort them; same rationale as fitting's skip_first).
METRICS_SKIP_FIRST = 4


def _excess_kurtosis(x: torch.Tensor) -> torch.Tensor:
    """Excess kurtosis along the last dim. High values = a few extreme logits
    stand out from the bulk — the paper's signature of workspace layers."""
    mu = x.mean(-1, keepdim=True)
    centered = x - mu
    var = centered.pow(2).mean(-1)
    return centered.pow(4).mean(-1) / var.pow(2).clamp_min(1e-12) - 3.0


@torch.no_grad()
def read_grid(
    model: WrappedModel,
    lens: JacobianLens,
    prompt: str,
    *,
    mode: str = "jlens",  # "jlens" | "logit"
    top_k: int = 8,
    max_seq_len: int = 256,
    pinned_ids: list[int] | None = None,
    chat: bool = False,
    track_all: bool = False,
) -> dict[str, Any]:
    """One forward pass, then per-layer lens readouts at every position.

    Returns a JSON-ready dict: per-cell top-k tokens/probs, per-cell entropy
    and kurtosis, pinned-token ranks, and per-layer workspace metrics
    (excess kurtosis, lens next-token accuracy, adjacent-position top-1
    agreement). The final layer is always read with ``J = I`` — that row is
    the model's actual output distribution.

    With ``track_all`` the result also carries a ``ranks`` map
    ``{token_id: [n_layers][seq] rank}`` covering every token that appears in
    any top-k cell — everything a *static* frontend needs to serve pin/trace
    interactions without a server (see ``scripts/export_static.py``).
    """
    if chat:
        prompt = model.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    final_layer = model.n_layers - 1
    layers = sorted(set(lens.source_layers) | {final_layer})
    pinned = list(pinned_ids or [])

    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]
    ids_list = input_ids[0].tolist()
    with ActivationRecorder(model.layers, at=layers) as recorder:
        model.forward(input_ids)
        acts = {l: recorder.activations[l].detach() for l in layers}

    # Greedy continuation for the "model says" readout — answers are phrases,
    # not single tokens.
    generated = model.hf_model.generate(
        input_ids,
        max_new_tokens=12,
        do_sample=False,
        pad_token_id=model.tokenizer.eos_token_id,
    )
    continuation = model.tokenizer.decode(
        generated[0, seq_len:], skip_special_tokens=True
    )

    next_ids = input_ids[0, 1:].to(model.device)  # target for next-token acc
    valid = slice(METRICS_SKIP_FIRST, seq_len - 1)

    grid: list[dict[str, Any]] = []
    layer_metrics: list[dict[str, Any]] = []
    token_ids_seen: set[int] = set(ids_list) | set(pinned)
    pinned_t = torch.tensor(pinned, dtype=torch.long, device=model.device) if pinned else None
    rank_tables: dict[int, torch.Tensor] = {}  # layer -> [seq, vocab] full ranks
    tracked_ids: set[int] = set()

    for layer in layers:
        residual = acts[layer][0].float()
        if mode == "jlens" and layer in lens.jacobians:
            residual = lens.transport(residual, layer)
        logits = model.unembed(residual).float()  # [seq, vocab]
        probs = logits.softmax(-1)

        top = probs.topk(top_k, dim=-1)
        top_ids = top.indices  # [seq, k]
        token_ids_seen.update(top_ids.flatten().tolist())

        entropy = -(probs.clamp_min(1e-12).log() * probs).sum(-1)
        kurt = _excess_kurtosis(logits)

        row: dict[str, Any] = {
            "layer": layer,
            "is_output": layer == final_layer,
            "top_ids": top_ids.cpu().tolist(),
            "top_probs": [[round(p, 5) for p in ps] for ps in top.values.cpu().tolist()],
            "entropy": [round(e, 3) for e in entropy.cpu().tolist()],
            "kurtosis": [round(k, 2) for k in kurt.cpu().tolist()],
        }
        if pinned_t is not None:
            # rank of each pinned token at each position (0 = top)
            pinned_logits = logits[:, pinned_t]  # [seq, n_pinned]
            ranks = (logits.unsqueeze(-1) > pinned_logits.unsqueeze(1)).sum(1)
            row["pinned_ranks"] = ranks.cpu().tolist()
        if track_all:
            # full-vocab rank table for this layer (static-export path; memory
            # is seq * vocab ints per layer, so keep prompts short)
            sorted_idx = logits.argsort(dim=-1, descending=True)
            full_rank = torch.empty_like(sorted_idx)
            full_rank.scatter_(
                1,
                sorted_idx,
                torch.arange(logits.shape[-1], device=logits.device).expand_as(sorted_idx),
            )
            rank_tables[layer] = full_rank.to(torch.int32).cpu()
            tracked_ids.update(top_ids.flatten().tolist())
        grid.append(row)

        top1 = top_ids[:, 0]
        layer_metrics.append(
            {
                "layer": layer,
                "next_token_acc": round(
                    (top1[valid] == next_ids[valid]).float().mean().item(), 4
                ),
                "mean_kurtosis": round(kurt[valid].mean().item(), 2),
                "top1_autocorr": round(
                    (top1[valid] == top1[METRICS_SKIP_FIRST + 1 : seq_len])
                    .float()
                    .mean()
                    .item(),
                    4,
                ),
            }
        )
        del logits, probs

    decode = model.tokenizer.decode
    vocab = {int(t): decode([int(t)]) for t in token_ids_seen}
    ranks: dict[int, list[list[int]]] | None = None
    if track_all:
        ranks = {
            int(t): [rank_tables[l][:, t].tolist() for l in layers]
            for t in sorted(tracked_ids)
        }
    return {
        "continuation": continuation,
        **({"ranks": ranks} if ranks is not None else {}),
        "mode": mode,
        "prompt": prompt,
        "seq_len": seq_len,
        "layers": layers,
        "context_ids": ids_list,
        "vocab": vocab,
        "grid": grid,
        "layer_metrics": layer_metrics,
        "pinned_ids": pinned,
    }
