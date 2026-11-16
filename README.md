# jspace-viz

Explore how decoded tokens change across a language model's layers. Compare a Jacobian lens with a standard logit lens, then click a token to trace its rank.

[Try the demo](https://festyve.github.io/jspace-viz/)

I built the interactive viewer, a Jacobian-lens estimator, and a resumable fitting pipeline. The method comes from [Gurnee et al. (2026)](https://transformer-circuits.pub/2026/workspace/index.html). This is an independent implementation, with support for the [official lens format](https://github.com/anthropics/jacobian-lens).

## Try it

The demo opens on a short Python example using DeepSeek-Coder 1.3B. Results are precomputed; no setup is needed. Custom prompts require a local run.

1. Read the prompt. It sums the squares of 1, 2, and 3, giving 14.
2. Compare **Jacobian lens** and **Logit lens** at the highlighted final prompt column.
3. Click a cell to trace that token. Rank 1 is highest. Use **Show tokens** to return.

Rows are layers. Columns are token positions. Each cell shows the lens's highest-scoring token; stronger blue means a higher score. Green headers mark generated text. Whitespace appears as `[space]`, `\n`, or `\t`.

These are decoded activations, not a transcript of reasoning. A meaningful token alone does not establish a causal role. Suggested tokens and chart shading use heuristics.

## Run locally

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Festyve/jspace-viz
cd jspace-viz
uv venv
uv pip install -e '.[fit]'
.venv/bin/jspace-viz --preset deepseek-coder-1.3b
```

Open `http://127.0.0.1:8321`. The first run downloads about 3 GB. For a smaller download, use `--preset gpt2` (about 600 MB). More models are listed in [presets.py](jspace_viz/presets.py).

## Fit and check a lens

```bash
.venv/bin/python scripts/fit_lens.py --preset deepseek-coder-1.3b --n-prompts 40 --dim-batch 16
.venv/bin/python scripts/validate_lens.py
```

Fitting saves resumable checkpoints. The demo uses a 40-prompt lens. The validation script compares fitted matrices with a Neuronpedia reference lens; similarity is a consistency check, not proof of equivalence.

To export precomputed examples for static hosting:

```bash
.venv/bin/python scripts/export_static.py --preset deepseek-coder-1.3b
```

## Credits

Based on [the paper](https://transformer-circuits.pub/2026/workspace/index.html) and [Anthropic's reference implementation](https://github.com/anthropics/jacobian-lens). Community lenses come from [Neuronpedia](https://huggingface.co/neuronpedia/jacobian-lens); the DeepSeek lens is [published here](https://huggingface.co/Festyve/jspace-lenses).

[Apache-2.0 license](LICENSE).
