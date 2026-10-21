# GPT from scratch

A GPT built in PyTorch.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

Run a script:

```sh
uv run python file.py
```

Or open JupyterLab and pick the "Python (gpt-from-scratch)" kernel:

```sh
uv run jupyter lab
```

Training runs on Apple Silicon via PyTorch's MPS backend.

## Layout

- `data/` holds datasets and reference material. It is git-ignored.
- `checkpoints/` holds saved model weights. It is git-ignored.
