# First GPU Runbook

## Hardware

Recommended first run:

- NVIDIA GPU with **24 GB VRAM** or more
- CUDA-capable PyTorch environment
- no H100 required

The v0 run freezes the 0.6B backbone and trains only the dynamic decision head.

## Setup

```bash
git clone <repo>
cd open-system-one
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[train,dev]'
```

## Sanity checks

```bash
make test
make data
PYTHONPATH=. python scripts/validate_dataset.py data/simulator
```

## Fast decision gate

```bash
PYTHONPATH=. python scripts/train.py --config configs/train_head_fast.yaml
```

Inspect:

```text
checkpoints/head-fast/summary.json
```

Do not scale if:

- dev NLL does not improve during training
- Brier remains near the uniform baseline
- outputs are unstable under candidate reordering

## Main head run

```bash
PYTHONPATH=. python scripts/train.py --config configs/train_head.yaml
```

## Semantic smoke baseline

```bash
PYTHONPATH=. python scripts/benchmark_option_likelihood.py \
  --model Qwen/Qwen3-0.6B-Base \
  --data data/benchmark/semantic_smoke.jsonl
```

The current option-likelihood implementation is intentionally naive and does not reuse KV prefixes. Use it for quality, not final latency comparison.

## Jev

Only after the benchmark is frozen:

```bash
export TYPESAFE_API_KEY=...
PYTHONPATH=. python scripts/benchmark_jev.py \
  --data data/benchmark/semantic_smoke.jsonl
```

## What to send back for the next iteration

The most useful artifacts are:

```text
checkpoints/head-fast/summary.json
checkpoints/head-fast/train_log.json
checkpoints/head-fast/test_predictions.jsonl
checkpoints/head-fast/ood_predictions.jsonl
results/option_likelihood.summary.json
results/jev_predictions.summary.json   # if available
```

With those outputs, the next decision is evidence-driven: fix data/head, add semantic datasets, or move to LoRA/larger backbone.
