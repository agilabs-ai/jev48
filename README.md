# OpenJev

**A small, open Jev-style decision model built by extending NanoJev rather than rebuilding it.**

OpenJev starts from the public **NanoJev** checkpoint and training code, then adds a broader semantic training distribution, simulator-grounded probability supervision, and a frozen multi-domain benchmark.

The weekend question is deliberately narrow:

> Can a very small amount of additional training turn NanoJev from a strong game/navigation reproduction into a more general semantic decision model — without changing the core 0.6B architecture?

OpenJev is **not affiliated with TypeSafe** and does not claim to reproduce Jev's proprietary architecture or RLCD method.

## What changed vs NanoJev

The first OpenJev experiment changes as little as possible:

```text
NanoJev public checkpoint
        │
        ├── same Qwen3-0.6B backbone
        ├── same dynamic decision architecture
        ├── same 2–255 candidate interface
        ├── same zero output-token decoding
        │
        ▼
+ semantic decision data
+ known-probability simulator data
+ Brier/proper-distribution fine-tuning
        │
        ▼
      OpenJev
```

No new architecture is required for v0. The point is to isolate whether **training distribution + calibration supervision** are enough to create a meaningfully broader model.

## Default experiment

Run one command on Modal:

```bash
python -m pip install 'modal>=1.1,<2'
modal setup
modal run modal_app.py
```

The default run:

1. builds public semantic train/test data;
2. mixes it with known-probability simulator data;
3. freezes and hashes `OpenDecisionBench`;
4. downloads the pinned NanoJev source + public checkpoint;
5. benchmarks **base NanoJev** before training;
6. validates our data using NanoJev's own schema validator;
7. fine-tunes the NanoJev checkpoint for a small number of steps;
8. benchmarks **OpenJev** on the exact same frozen rows;
9. applies predeclared launch gates;
10. persists the checkpoint + raw results.

Round 1 uses **no paid OpenAI/Anthropic APIs**.

See [`RUN_NOW.md`](RUN_NOW.md).

## Training/evaluation data

Seen training domains include:

- Banking77-style intent/routing decisions
- BoolQ-style binary semantic decisions
- controlled known-probability simulator tasks

OOD benchmark domains include completely held-out public domains such as:

- DBpedia14
- AG News

Candidate order is deterministically shuffled in the frozen benchmark so neither model can benefit from stable label positions.

## What counts as a win

We do not need OpenJev to beat NanoJev on NanoJev's own maze benchmark.

The predeclared main gate is broader semantic decision quality:

- lower Brier with accuracy within 2 percentage points of NanoJev, **or**
- at least +5 percentage points of accuracy without a large calibration regression.

OOD is reported separately.

See [`MARKETING_GATES.md`](MARKETING_GATES.md).

## Attribution

OpenJev v0 is explicitly **derived from NanoJev**:

- source baseline: `TianyuCodings/NanoJev`
- public checkpoint: `C-Tianyu/NanoJev`
- pinned source revision: `71a513bb0163b5634467842b523ee0c0ed6fb1c7`

NanoJev is MIT licensed. See [`NOTICE.md`](NOTICE.md) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Repo layout

```text
openjev/
├── openjev/                 # benchmark/data/calibration support library
├── scripts/                 # data, benchmark, adapters, ablations
├── tests/                   # local verification
├── data/                    # small local smoke/simulator fixtures
├── experiments/             # independent-from-scratch ablation retained for history
├── modal_app.py             # default OpenJev training + NanoJev comparison
├── PROJECT_SPEC.md
├── RUN_NOW.md
├── MARKETING_GATES.md
└── NOTICE.md
```

The independent ChatGPT-built decision model is retained as an **ablation**, not the headline model. That work was useful for building the benchmark and understanding the problem, but the default project now optimizes for the actual weekend objective: **improve the strongest existing open baseline with the smallest credible change.**
