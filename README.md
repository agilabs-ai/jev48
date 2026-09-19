# Open System One

**State + decision candidates in. Full probability distributions out. Zero output-token decoding.**

This repository is a weekend research build to test how much of the useful **Jev / System One** primitive can be reproduced with a small open model, transparent training data, proper scoring rules, and very little compute.

It is **not** affiliated with TypeSafe and does not claim to reproduce Jev's private architecture or RLCD recipe.

## The experiment

We want to answer one question:

> Can a small open backbone become a fast, calibrated, dynamic decision model that generalizes beyond its training templates?

The v0 architecture is deliberately simple:

```text
state + question + candidate A ─┐
state + question + candidate B ─┼─ one batched backbone forward ─ dynamic set head ─ softmax
state + question + candidate C ─┘
```

Important honesty point: **v0 batches complete candidate paths. It does not yet share the state/question prefix inside the trained path.** It has zero autoregressive decoding, but prefix-sharing is a separate systems optimization and is benchmarked separately.

## What already works

The repo currently contains:

- a validated decision-example schema
- variable 2–255 candidate support in the schema
- a permutation-equivariant dynamic decision head
- soft-target cross entropy + categorical Brier training
- post-hoc temperature calibration
- Brier / NLL / ECE / TV / risk-coverage metrics
- known-probability simulators
- a fully held-out simulator OOD domain
- split/family/content leakage validation
- candidate-order permutation augmentation
- hard-case mining
- OpenAI + Anthropic teacher-ensemble plumbing
- raw Jev benchmark adapter
- generative option-likelihood baseline
- FastAPI serving path
- a frozen semantic smoke benchmark
- CPU-only end-to-end training validation
- unit tests that run without model downloads

See [`STATUS.md`](STATUS.md) for exactly what has and has not been executed, and [`CHATGPT_BUILD_LOG.md`](CHATGPT_BUILD_LOG.md) for the auditable in-chat build record.

## Local smoke test

No GPU, model download, or API key required:

```bash
make test
make smoke
PYTHONPATH=. python scripts/train_tiny_cpu.py --steps 150
```

The CPU model is deliberately tiny and is **not** a Jev competitor. It exists to prove that the data → dynamic head → proper loss → calibration → evaluation path runs end-to-end before spending money.

## First real model run

Install training dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[train,dev]'
```

Generate / validate data:

```bash
make data
PYTHONPATH=. python scripts/validate_dataset.py data/simulator
```

Train only the decision head over Qwen3-0.6B:

```bash
PYTHONPATH=. python scripts/train.py --config configs/train_head_fast.yaml
```

If the fast run beats the untuned baseline on Brier + NLL, run:

```bash
PYTHONPATH=. python scripts/train.py --config configs/train_head.yaml
```

**Do not jump to LoRA/full fine-tuning until that gate passes.**

## Baselines

### Untuned option likelihood

```bash
PYTHONPATH=. python scripts/benchmark_option_likelihood.py \
  --model Qwen/Qwen3-0.6B-Base \
  --data data/benchmark/semantic_smoke.jsonl
```

This implementation intentionally re-encodes prefixes and is a correctness baseline. An optimized prefix-cache implementation is the next systems baseline.

### Jev

Requires `TYPESAFE_API_KEY`:

```bash
PYTHONPATH=. python scripts/benchmark_jev.py \
  --data data/benchmark/semantic_smoke.jsonl
```

Freeze benchmark files **before** querying Jev:

```bash
PYTHONPATH=. python scripts/freeze_benchmark.py data/benchmark/semantic_smoke.jsonl
```

## Premium teacher labeling

Mine hard examples after a student run:

```bash
PYTHONPATH=. python scripts/mine_hard_cases.py \
  --predictions checkpoints/head-v0/test_predictions.jsonl \
  --source-data data/simulator/test.jsonl \
  --top 1000
```

Then label only those hard cases, for example:

```bash
OPENAI_API_KEY=... PYTHONPATH=. python scripts/label_hard_cases.py \
  --input data/hard_cases.jsonl \
  --openai-model gpt-6-astra \
  --samples-each 2
```

The point is to use frontier intelligence as a **scalpel**, not as the dataset factory.

## Main metrics

We report at minimum:

- accuracy / optimal-action agreement
- Brier score
- negative log likelihood
- ECE
- total variation to known target distributions
- risk vs. coverage
- latency vs. candidate count
- latency vs. state length

For soft simulator targets, calibration/error metrics use the **target probability of the model's chosen class**, not a fake binary argmax label.

## What would count as a win?

A useful weekend result does **not** require beating Jev everywhere.

Strong outcomes include:

1. a 600M open model matches Jev on several narrow domains;
2. a tiny specialized model gets most of the useful decision behavior for <$300;
3. simple option likelihood already captures most of the benefit;
4. Jev clearly wins on OOD generalization, revealing where the actual secret sauce begins.

All four are informative.

## Why the repo starts with a project spec

The project intentionally preserves [`PROJECT_SPEC.md`](PROJECT_SPEC.md) as an artifact of the build process. The experiment is partly meta: how much of an OSS Jev-style replication can be built by ChatGPT in one weekend, including the methodology, code, data pipeline, evaluation harness, and debugging loop?

The benchmark matters more than the story. A sample multi-question request is in [`examples/systemone_request.json`](examples/systemone_request.json).
