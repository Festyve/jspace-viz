# jspace-viz

**Live J-space / Jacobian-lens visualizer for open-weights language models.**

Type a prompt, watch what every layer of the model is *disposed to say* at
every token position — the "verbalizable workspace" from Anthropic's
[*Verbalizable Representations Form a Global Workspace in Language
Models*](https://transformer-circuits.pub/2026/workspace/index.html)
(Gurnee et al., 2026) — rendered as an interactive layer × position heatmap,
with one-click comparison against the vanilla logit lens and live plots of the
paper's workspace-band signatures.

![screenshot](assets/screenshot.png)

## What the Jacobian lens is

The residual stream at layer `l` lives in a different basis than the final
layer, so decoding it directly with the unembedding (the classic *logit lens*)
breaks down in early and middle layers. The Jacobian lens first transports the
activation into the final-layer basis using the **average input–output
Jacobian** over a text corpus:

```
lens_l(h) = unembed( J_l @ h ),    J_l = E[ ∂h_final / ∂h_l ]
```

The expectation runs over prompts, source positions, and all current-and-later
target positions. The rows of `W_U · J_l` are the **J-lens vectors** — one
direction per vocabulary token — and the set of sparse non-negative
combinations of them is the **J-space**, the paper's candidate for a global
workspace: the small, verbalizable, causally load-bearing slice of the model's
state.

This repo is an **independent implementation** of the estimator described in
the paper and its [official reference code](https://github.com/anthropics/jacobian-lens)
(Anthropic PBC, Apache-2.0). The on-disk lens format is byte-compatible in
both directions, so the prebaked community lenses that
[Neuronpedia publishes on the Hub](https://huggingface.co/neuronpedia/jacobian-lens)
load directly, and lenses fit here load in the official tooling.

**Validation** — `scripts/validate_lens.py` fits Pythia-70m on 8 WikiText
prompts and compares against Neuronpedia's 1000-prompt lens (fit with the
official code, on a B200):

| layer | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| cos(J_ours, J_ref) | 0.66 | 0.76 | 0.89 | 0.94 | 0.98 |

Rising toward 1.0 with layer is the expected convergence pattern (early-layer
Jacobians have the highest per-prompt variance), confirming the estimators
match.

## Quick start

```bash
git clone https://github.com/Festyve/jspace-viz && cd jspace-viz
uv venv && uv pip install -e '.[fit]'

# instant demo: GPT-2 + prebaked community lens (auto-downloads)
.venv/bin/jspace-viz --preset gpt2
# → open http://127.0.0.1:8321
```

In the UI:

- **cells** show the lens top-1 token per (layer, position); opacity = probability.
  The bottom row is the model's actual output (`J = I` at the final layer).
- **hover** a cell for the top-k readout, entropy, and excess kurtosis.
- **click** a cell to pin its token and switch to a **rank heatmap** — track
  where in the network (and at which positions) a concept is active, like the
  paper's rank-tracking charts.
- **J-lens ↔ logit lens** toggle: see exactly where the logit lens falls apart
  in middle layers and the Jacobian transport keeps decoding.
- **metrics panel**: per-layer next-token accuracy, mean excess kurtosis, and
  adjacent-position top-1 autocorrelation — the structural signatures the
  paper uses to bound the workspace band (kurtosis band shaded, heuristic).

## DeepSeek (or any other model)

No public DeepSeek lens existed, so this repo fits its own. On an Apple-Silicon
laptop (16 GB is enough for a ~1.3B model — fitting backprops through the model
`d_model` times per prompt):

```bash
# checkpointed + resumable; writes a usable partial lens after every prompt
.venv/bin/python scripts/fit_lens.py --preset deepseek-coder-1.3b --n-prompts 40
.venv/bin/jspace-viz --preset deepseek-coder-1.3b
```

Presets (see `jspace_viz/presets.py` — any HF causal LM with a Llama-style,
GPT-2, or NeoX layout works via `--model-id` + `--lens`):

| preset | model | lens | runs on |
|---|---|---|---|
| `gpt2` | gpt2 (124M) | prebaked (Neuronpedia) | anything |
| `pythia-70m` | Pythia-70m | prebaked (Neuronpedia) | anything |
| `deepseek-coder-1.3b` | deepseek-coder-1.3b-instruct | fit locally (~hours on M-series) | 16 GB laptop |
| `gemma-3-1b-it` | Gemma-3-1B-it (gated) | prebaked (Neuronpedia) | 16 GB laptop |
| `llama-3.1-8b-it` | Llama-3.1-8B-it (gated) | prebaked (Neuronpedia) | 24 GB+ GPU |
| `deepseek-r1-distill-llama-8b` | R1-Distill-Llama-8B | fit yourself | A100-class GPU |

Corpus follows the Neuronpedia convention (WikiText-103 stream, ≤128 tokens,
first 16 positions skipped as attention sinks). Quality saturates fast: the
paper reports ~10 prompts already beats the logit lens, ~100 is solid, 1000 is
what they ship.

## Roadmap

- J-space sparse decomposition (gradient pursuit, k ≤ 25) per cell
- interventions: J-lens-vector swaps/steering (the paper's thought-swap protocol)
- role–filler binding probes on top of the workspace readout

## Credits & license

- Paper: Gurnee et al., *Verbalizable Representations Form a Global Workspace
  in Language Models*, Transformer Circuits, 2026.
- Reference implementation: [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) (Apache-2.0).
  The estimator here follows it; three demo prompts in `presets.py` are adapted
  from its examples.
- Prebaked lenses: [neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens) on the Hub.

Code: Apache-2.0 (see `LICENSE`).
