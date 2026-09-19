# Marketing / launch gates

NanoJev already exists. That changes the claim.

The project should **not** launch as “the first open-source Jev” or imply that no open reproduction existed.

The strongest defensible stunt is:

> **I asked ChatGPT to rebuild a Jev-style decision model in a weekend. Then we benchmarked it against Jev and the strongest open baseline.**

If the numbers cooperate, upgrade the claim to:

> **ChatGPT built a Jev-style model in a weekend that beats NanoJev on a broader semantic decision benchmark.**

## Why NanoJev matters

NanoJev already demonstrates:

- Qwen3-0.6B backbone
- dynamic 2–255 candidate choices
- zero output-token decoding
- complete distributions
- calibrated-decision experiments
- strong navigation/game results

Therefore merely recreating its architecture is not enough to create a strong technical story.

## Headline benchmark gate

Freeze `OpenDecisionBench` **before** looking at NanoJev results.

Compare, on exactly the same rows:

1. NanoJev public checkpoint
2. Open System One
3. untuned Qwen option-likelihood baseline
4. Jev, if API access is available
5. one cheap frontier model, optional

Primary metrics:

- accuracy / argmax agreement
- Brier
- NLL
- ECE
- total variation on known-distribution cases
- risk/coverage

Systems metrics, reported separately:

- p50/p95 or full-batch latency
- questions/sec
- candidate-count scaling
- state-length scaling

## What counts as “better than NanoJev”

A credible marketing win requires **at least one meaningful, predeclared dimension**, not cherry-picking a random cell after results arrive.

Preferred win condition:

### Gate A — semantic generality

Open System One beats NanoJev on the frozen multi-domain semantic benchmark on:

- lower Brier **and**
- non-inferior accuracy (within 2 percentage points),

or:

- at least +5 percentage points accuracy with no catastrophic calibration regression.

### Gate B — OOD calibration

On a fully held-out domain, Open System One has meaningfully lower Brier/TV than NanoJev while retaining useful top-1 accuracy.

### Gate C — efficiency

On matched hardware and matched input batches, Open System One produces comparable quality with materially better throughput or lower memory.

This is harder because our v0 and NanoJev both duplicate complete candidate paths and do not perform prefix sharing.

## What does **not** count

Do not claim “better” because:

- we win on a benchmark generated from our own training templates
- we compare our tuned model against NanoJev on data NanoJev was never intended for without also disclosing that fact
- latency was measured on different GPUs
- one model includes cold-start/model-load time and the other does not
- we cherry-pick one domain after inspecting all outcomes

## Home-field benchmark

For completeness, also run NanoJev's own released navigation benchmark.

It is fine if NanoJev wins there.

The more interesting result may be:

| Benchmark | NanoJev | Open System One |
|---|---|---|
| NanoJev navigation | stronger | weaker |
| broad semantic decisions | weaker | stronger |
| calibrated simulator OOD | ? | ? |

That supports a truthful claim that we extended the idea from a specialized reproduction toward a more general semantic decision model.

## Go / no-go launch logic

### Strong launch

We beat NanoJev on a predeclared broad semantic/calibration gate.

### Good launch

We match it surprisingly closely with an independently built pipeline, tiny spend, and less than a weekend of iteration.

### Weak launch

NanoJev beats us everywhere meaningful.

If that happens, do **not** force the “copy of Jev” story. Either:

1. use NanoJev as initialization and improve it openly, or
2. pivot the stunt to a transparent replication attempt: “ChatGPT tried to recreate Jev in a weekend; here is exactly where it failed.”

The first option is better for marketing if we can ship a real improvement.
