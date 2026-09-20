---
license: apache-2.0
base_model: Mapika/decider-2b
pipeline_tag: text-classification
library_name: transformers
tags:
- decision-model
- calibration
- system-one
- jev48
---
# jev48-2b

This checkpoint is the derivative selected by the **Jev48** weekend reproduction experiment.
It derives from **Mapika/decider-2b**; the upstream architecture and starting weights
are not Jev48 inventions. Jev48 tests empirical soft human-vote distributions against
hard-majority preference labels while retaining replay data.

## Lineage

- Starting model: `Mapika/decider-2b`
- Upstream repository: `Mapika/decider`
- Upstream commit: `b08acf787d5d1f718a8c36c4677960f43772c7be`
- Selected trial: `soft-lr3e-6`
- Benchmark SHA-256: `61a429af4502688f44d3d80a211f39282b6ff7ae9b78aac29fe1e7c029fe65a8`
- Fitted temperature: `1.041342653459261`

## Locked evaluation

| Slice | Accuracy ↑ | Brier ↓ | ECE ↓ |
|---|---:|---:|---:|
| MT-Bench expert-vote test | 0.6207 | 0.4110 | 0.0587 |
| Held-out transfer OOD | 0.7926 | 0.3039 | 0.0457 |

Temperature was fitted only on the frozen calibration split after model selection.
Locked test/OOD were not used for selection. Jev outputs were not used for training.

## Independent public Jev benchmark

This is an aggregate, unpaired comparison on `LocalLLaMA/typed-decisions`, not a
live Jev run on Jev48's original rows. The benchmark maintainers published the
Jev row; Jev48 ran separately, zero-shot, on the pinned test split.

| Model | Accuracy ↑ | Brier ↓ | KL ↓ | ECE ↓ |
|---|---:|---:|---:|---:|
| TypeSafe Jev 1.13.0 (published) | 0.7270 | 0.1480 | 1.4420 | 0.1440 |
| jev48-2b (zero-shot) | 0.5770 | 0.2551 | 0.4964 | 0.1201 |

Benchmark revision: `ea9306458d6e9563628369a3d1e72e362fb381d2`. Jev leads accuracy and Brier;
Jev48 has lower reported KL and ECE.

## Intended use

Typed, generation-free decision scoring through the upstream `decider` interface.
Validate calibration on your own outcome-labelled workload before consequential use.

## Attribution and licenses

Jev48 code is MIT. This model derives from `Mapika/decider-2b`; preserve all applicable
Apache-2.0 and underlying Qwen/model/dataset obligations. See
[`agilabs-ai/jev48`](https://github.com/agilabs-ai/jev48) and `THIRD_PARTY_NOTICES.md`.
