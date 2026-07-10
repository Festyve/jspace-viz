# Copyright 2026 Michael Zhang
# SPDX-License-Identifier: Apache-2.0
"""Model presets: which HF model to load and where its lens lives.

Lens sources are either ``("hub", repo_id, filename)`` — e.g. the community
lenses Neuronpedia publishes at ``neuronpedia/jacobian-lens`` (fit with the
official reference code on WikiText-103) — or ``("file", path)`` for lenses
fit locally with ``scripts/fit_lens.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

_NEURONPEDIA = "neuronpedia/jacobian-lens"


@dataclass(frozen=True)
class Preset:
    model_id: str
    lens: tuple[str, ...]
    dtype: str = "auto"
    notes: str = ""


PRESETS: dict[str, Preset] = {
    # Runs (and fits) on a 16 GB Apple-Silicon laptop.
    "deepseek-coder-1.3b": Preset(
        model_id="deepseek-ai/deepseek-coder-1.3b-instruct",
        lens=("file", "lenses/deepseek-coder-1.3b-instruct_jlens_wikitext.pt"),
        notes="Fit locally with scripts/fit_lens.py (no public DeepSeek lens exists).",
    ),
    # Instant smoke test: tiny model, prebaked community lens.
    "gpt2": Preset(
        model_id="gpt2",
        lens=("hub", _NEURONPEDIA, "gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"),
        dtype="float32",
    ),
    "pythia-70m": Preset(
        model_id="EleutherAI/pythia-70m-deduped",
        lens=(
            "hub",
            _NEURONPEDIA,
            "pythia-70m-deduped/jlens/Salesforce-wikitext/pythia-70m-deduped_jacobian_lens.pt",
        ),
        dtype="float32",
    ),
    # Prebaked lens, model gated behind the Gemma license on the Hub.
    "gemma-3-1b-it": Preset(
        model_id="google/gemma-3-1b-it",
        lens=("hub", _NEURONPEDIA, "gemma-3-1b-it/jlens/Salesforce-wikitext/gemma-3-1b-it_jacobian_lens.pt"),
        notes="Accept the Gemma license on the Hub first.",
    ),
    # Needs a real GPU (24 GB+ for reading, more for fitting).
    "llama-3.1-8b-it": Preset(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        lens=(
            "hub",
            _NEURONPEDIA,
            "llama3.1-8b-it/jlens/Salesforce-wikitext/Llama-3.1-8B-Instruct_jacobian_lens.pt",
        ),
        notes="GPU recommended; model gated on the Hub.",
    ),
    "deepseek-r1-distill-llama-8b": Preset(
        model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        lens=("file", "lenses/deepseek-r1-distill-llama-8b_jlens_wikitext.pt"),
        notes="No public lens; fit on a GPU box (a single A100-hour is plenty for ~100 prompts).",
    ),
}

#: Demo prompts. The first three follow examples shipped with the official
#: jacobian-lens repo (Apache-2.0, Anthropic PBC).
EXAMPLES: list[dict[str, str]] = [
    {
        "name": "Multi-hop: country shaped like a boot",
        "prompt": (
            "Fact: The capital of Japan is Tokyo.\n"
            "Fact: The currency used in the country shaped like a boot is"
        ),
    },
    {
        "name": "ASCII face",
        "prompt": (
            "     _______     \n"
            "   /         \\   \n"
            "  /  ~     ~  \\  \n"
            " (   o     o   ) \n"
            " |      ^      | \n"
            " |             | \n"
            " |   \\_____/   | \n"
            "  \\           /  \n"
            "   \\_________/   \n"
            "      |   |      \n\n"
            "What is this?"
        ),
    },
    {
        "name": "Two-hop: animal that spins webs",
        "prompt": "Question: How many legs does the animal that spins webs have?\nAnswer: It has",
    },
    {
        "name": "Code: off-by-one bug",
        "prompt": (
            "def get_last(items):\n"
            "    return items[len(items)]\n\n"
            "# Review: the bug in this function is"
        ),
    },
    {
        "name": "Code: what does this print?",
        "prompt": (
            "x = [1, 2, 3]\n"
            "y = sum(v * v for v in x)\n"
            "print(y)\n"
            "# The output of this program is"
        ),
    },
]
