# Execution Plan

## Phase 0 — local scaffolding — DONE

- [x] schema + canonical candidate formatting
- [x] dynamic decision head
- [x] soft CE + Brier
- [x] calibration code
- [x] probability-quality metrics
- [x] known-probability simulators
- [x] OOD simulator domain
- [x] data leakage validator
- [x] semantic smoke benchmark
- [x] benchmark hash/freeze tool
- [x] CPU baseline
- [x] exact dynamic-head CPU training smoke
- [x] Jev adapter
- [x] frontier teacher ensemble
- [x] hard-case miner
- [x] API serving skeleton
- [x] tests

## Phase 1 — first real backbone — NEXT

### Run A: untuned Qwen option likelihood

```bash
PYTHONPATH=. python scripts/benchmark_option_likelihood.py \
  --model Qwen/Qwen3-0.6B-Base \
  --data data/benchmark/semantic_smoke.jsonl
```

Purpose: establish how much we get for free.

### Run B: fast head-only model

```bash
PYTHONPATH=. python scripts/train.py --config configs/train_head_fast.yaml
```

Gate:

- trained head must improve dev NLL / Brier over untuned decision behavior
- probabilities must remain stable under candidate permutation
- no major regression from calibration

If not: fix data/architecture before scaling.

### Run C: proper head-only model

```bash
PYTHONPATH=. python scripts/train.py --config configs/train_head.yaml
```

Then evaluate seen test + OOD separately.

## Phase 2 — real semantic data

The simulator is for calibration science, not broad intelligence.

Add objective datasets for:

- intent classification
- routing
- entailment
- relevance
- tool selection
- moderation/safety
- semantic matching
- document classification

Rules:

- preserve licenses and source IDs
- group related rows before splitting
- final benchmark source must differ from training generator/source
- never make final claims on synthetic-only evaluation

## Phase 3 — hard-case distillation

1. Run the best student over a large candidate pool.
2. Rank by entropy + margin + mistakes.
3. Send only the hardest ~1k–5k cases to premium teachers.
4. Randomize candidate order across teacher samples.
5. Aggregate distributions and disagreement.
6. Retrain and re-evaluate.

## Phase 4 — Jev comparison

Before the first Jev request:

```bash
PYTHONPATH=. python scripts/freeze_benchmark.py <benchmark.jsonl>
```

Then run Jev once against the frozen benchmark.

Report:

- aggregate metrics
- domain metrics
- raw predictions
- benchmark hash
- latency distribution
- usage/cost

Never alter that benchmark version after inspecting Jev.

## Phase 5 — systems optimization

Only after quality is credible:

1. optimized prefix-cache candidate scoring baseline
2. state-prefix sharing for the trained architecture if feasible
3. candidate-count scaling: 2 / 4 / 8 / 16 / 32 / 64 / 255
4. context-length scaling
5. throughput / concurrency

This is where we learn whether Jev's architecture has a meaningful systems advantage beyond specialization.

## Phase 6 — LoRA/full fine-tune

Only if head-only clearly underfits semantic tasks.

Order:

1. LoRA on 0.6B
2. 1.5B–2B backbone
3. 4B-class backbone
4. full fine-tuning only if benchmarks justify it

Do not solve a data/generalization problem by blindly buying more GPU.
