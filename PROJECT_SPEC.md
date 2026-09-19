# OpenJev — Weekend Project Spec

## Goal

Build the strongest credible weekend Jev-style OSS result by **starting from NanoJev and changing as little as possible**.

Primary hypothesis:

> NanoJev already has the right small-model decision primitive. The highest-leverage improvement is a broader training distribution plus better calibration supervision, not a new architecture.

## Base

Pinned source:

```text
TianyuCodings/NanoJev
71a513bb0163b5634467842b523ee0c0ed6fb1c7
```

Public checkpoint:

```text
C-Tianyu/NanoJev
```

OpenJev v0 should preserve:

- Qwen3-0.6B backbone
- NanoJev decision heads
- dynamic 2–255 candidate choices
- Boolean / choice / score semantics where supported
- zero autoregressive output decoding
- complete probability distributions

## Changes in v0

Only three intended changes:

### 1. Broader semantic data

Add real public decision tasks such as:

- Banking77 routing / intent classification
- BoolQ binary semantic decisions
- additional public classification/routing tasks only if needed

### 2. Known-probability supervision

Add controlled simulator examples where the underlying probability distribution is known rather than guessed by an LLM.

Examples:

- fraud risk
- churn
- delivery failure
- equipment failure

### 3. Frozen semantic/OOD benchmark

`OpenDecisionBench` should test:

- seen semantic tasks
- completely unseen domains
- large candidate counts (including 77-way routing)
- deterministic candidate permutation
- calibration
- accuracy
- Brier
- NLL
- ECE
- risk/coverage where applicable

## Non-goals for v0

Do not:

- redesign NanoJev's architecture
- train a model from scratch
- use H100s by default
- spend money on frontier-labeling before the public-label experiment
- claim “first OSS Jev”
- hide that OpenJev is derived from NanoJev
- optimize against NanoJev after seeing benchmark outputs

## Experiment

### Step 1 — freeze benchmark

Build semantic/OOD benchmark and write a SHA256 manifest before running either model.

### Step 2 — base NanoJev

Run the public NanoJev checkpoint on the frozen rows.

### Step 3 — OpenJev fine-tune

Warm-start from the exact same NanoJev checkpoint.

Default first training configuration:

```text
objective: gold_distribution
loss: brier
steps: 150
head_steps: 0
backbone_lr: 2e-5
head_lr: 2e-4
max_length: 512
gradient_checkpointing: true
```

This is deliberately conservative. If it improves the frozen benchmark, iterate from measured failure slices.

### Step 4 — compare

Primary table:

| Model | Accuracy ↑ | Brier ↓ | NLL ↓ | ECE ↓ |
|---|---:|---:|---:|---:|
| NanoJev | | | | |
| OpenJev | | | | |

Report:

- overall
- seen domains
- OOD domains
- per-domain slices

## Launch gates

Headline “OpenJev beats NanoJev” is allowed only if the frozen benchmark passes a predeclared gate:

### Semantic gate

Either:

- OpenJev Brier < NanoJev Brier and accuracy is within 2pp, or
- OpenJev accuracy is ≥5pp higher with no major Brier regression.

### OOD gate

Report separately. A strong OOD claim requires lower Brier/TV with useful accuracy on fully held-out domains.

## If v0 loses

Do not change the benchmark.

Diagnose in this order:

1. semantic data balance
2. catastrophic forgetting of NanoJev capabilities
3. simulator/public-data mixture
4. learning rate / number of steps
5. calibration temperature
6. only then architecture changes

Potential v0.2:

- replay a small amount of NanoJev's original training distribution during semantic fine-tuning to reduce forgetting
- lower backbone LR
- separate semantic and calibration stages

## Marketing framing

Strongest possible truthful result:

> I gave ChatGPT the best open-source Jev replica and told it to make it better. It changed almost nothing about the architecture — just the training distribution — and OpenJev beat the original NanoJev on a frozen semantic/OOD benchmark.

If it does not beat NanoJev, publish the real result rather than changing the gate.
