# Copyright 2026 Michael Zhang
# SPDX-License-Identifier: Apache-2.0
"""Thin wrapper locating the residual stack inside a HuggingFace causal LM."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class Layout:
    """Attribute paths to the lens-relevant submodules of a ``*ForCausalLM``."""

    path: str  # dotted path to the bare text decoder
    layers: str = "layers"
    norm: str = "norm"
    embed: str = "embed_tokens"
    lm_head: str = "lm_head"


# Tried in order; first layout whose attributes all resolve wins. Covers the
# modern Llama-style default (Llama/Mistral/Gemma/OLMo/DeepSeek), GPT-2, and
# GPT-NeoX (Pythia).
_LAYOUTS: tuple[Layout, ...] = (
    Layout("model"),
    Layout("model.language_model"),
    Layout("transformer", layers="h", norm="ln_f", embed="wte"),  # GPT-2
    Layout("gpt_neox", norm="final_layer_norm", embed="embed_in", lm_head="embed_out"),
)


def _resolve(obj: Any, dotted: str) -> Any:
    return functools.reduce(getattr, dotted.split("."), obj)


def _find_layout(hf_model: nn.Module) -> Layout:
    for layout in _LAYOUTS:
        try:
            decoder = _resolve(hf_model, layout.path)
        except AttributeError:
            continue
        if all(
            hasattr(decoder, a) for a in (layout.layers, layout.norm, layout.embed)
        ) and hasattr(hf_model, layout.lm_head):
            return layout
    raise ValueError(f"cannot locate the residual stack in {type(hf_model).__name__}")


class WrappedModel:
    """A loaded HF model exposing exactly what the lens needs.

    Freezes all parameters in place (Jacobian fitting differentiates with
    respect to activations only).
    """

    def __init__(self, hf_model: nn.Module, tokenizer: Any) -> None:
        hf_model.eval()
        for param in hf_model.parameters():
            param.requires_grad_(False)
        if (
            getattr(tokenizer, "bos_token_id", None) is not None
            and hasattr(tokenizer, "add_bos_token")
        ):
            # Raw-text prompts degrade without an attention-sink BOS.
            tokenizer.add_bos_token = True

        self.hf_model = hf_model
        self.tokenizer = tokenizer
        layout = _find_layout(hf_model)
        self._decoder = _resolve(hf_model, layout.path)
        self.layers: nn.ModuleList = getattr(self._decoder, layout.layers)
        self._final_norm: nn.Module = getattr(self._decoder, layout.norm)
        self._embed: nn.Module = getattr(self._decoder, layout.embed)
        self._lm_head: nn.Module = getattr(hf_model, layout.lm_head)

        config = hf_model.config.get_text_config()
        self.n_layers: int = config.num_hidden_layers
        self.d_model: int = config.hidden_size
        self._logit_softcap: float | None = getattr(
            config, "final_logit_softcapping", None
        )
        if len(self.layers) != self.n_layers:
            raise ValueError("config layer count disagrees with the module list")

    @property
    def device(self) -> torch.device:
        return self._embed.weight.device

    def encode(self, text: str, *, max_length: int = 512) -> torch.Tensor:
        ids = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).input_ids
        return ids.to(self.device)

    def forward(self, input_ids: torch.Tensor) -> Any:
        """Run the bare residual stack (no LM head), hooks visible."""
        return self._decoder(input_ids=input_ids, use_cache=False)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        """Final norm + LM head: ``[..., d_model] -> [..., vocab]``."""
        weight_dtype = self._lm_head.weight.dtype
        logits = self._lm_head(self._final_norm(residual.to(weight_dtype)))
        if self._logit_softcap is not None:
            logits = self._logit_softcap * torch.tanh(logits / self._logit_softcap)
        return logits


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(
    model_id: str, *, dtype: str = "auto", device: str | None = None
) -> WrappedModel:
    """Load ``model_id`` from the HF Hub (weights only, no remote code)."""
    import transformers

    device = device or pick_device()
    if dtype == "auto":
        dtype = "float32" if device == "cpu" else "bfloat16"
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, dtype=getattr(torch, dtype), trust_remote_code=False
    ).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    return WrappedModel(hf_model, tokenizer)
