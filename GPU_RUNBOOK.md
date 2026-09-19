# OpenJev GPU runbook

## Default

Use the Modal job:

```bash
modal run modal_app.py
```

It uses an A100 for convenience and reproducibility. **An H100 is not required.**

## What is trained

OpenJev v0 does not initialize a new model from Qwen.

It loads the public NanoJev checkpoint and performs a short full-model fine-tune using NanoJev's own training implementation:

```text
NanoJev checkpoint
  + semantic public labels
  + simulator probability distributions
  + Brier objective
  → OpenJev
```

Default training parameters:

```text
steps: 150
batch_questions: 1
microbatch_questions: 1
max_length: 512
backbone_lr: 2e-5
head_lr: 2e-4
gradient_checkpointing: true
precision: bf16
```

## Why A100 if H100 is unnecessary?

Only to reduce operational risk for the first run. The model is ~0.6B; after a successful run, reducing hardware cost is an optimization rather than a research question.

## Required outputs

```text
results/nanojev_open_decision.summary.json
results/openjev_open_decision.summary.json
results/openjev_vs_nanojev.md
results/openjev_gate.json
checkpoints/openjev/
```
