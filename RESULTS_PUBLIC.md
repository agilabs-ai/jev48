# Jev48 public benchmark comparison

Machine-generated from the pinned raw predictions. Do not edit numbers by hand.

This is an **aggregate comparison on an independent public benchmark**, not a paired
run against Jev on Jev48's original frozen rows. Jev's row was measured and published
by the `LocalLLaMA/typed-decisions` maintainers; Jev48 was run separately on the exact
pinned test split. Jev48 is evaluated zero-shot and used none of this benchmark's train rows.

| Model | Kind | Accuracy ↑ | Soft acc ↑ | Brier ↓ | KL ↓ | ECE ↓ | Score MAE ↓ | Within 1 ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TypeSafe Jev 1.13.0 | published general-model row | 0.727 | 0.580 | 0.148 | 1.442 | 0.144 | 0.391 | 0.952 |
| Jev48 / `soft-lr3e-6` | reproduced zero-shot general model | 0.577 | 0.508 | 0.255 | 0.496 | 0.120 | 0.483 | 0.899 |

- Benchmark: `LocalLLaMA/typed-decisions` at `ea9306458d6e9563628369a3d1e72e362fb381d2`.
- Test size: 400 cases / 2000 decisions.
- Jev48 median observed latency: 263.0 ms/case on Modal L40S; published Jev p50: 710 ms/case. Hardware/network differ, so this is not a controlled latency claim.
- Jev48 raw prediction SHA-256: `7e6319ddb51a7de982980244648dcaa962a0c52b9d05b2ade02d78f9ccb90e9a`.
- Public benchmark: https://huggingface.co/datasets/LocalLLaMA/typed-decisions

## Honest launch claim

On this public zero-shot benchmark, Jev leads Jev48 by 15.0 percentage points in accuracy and also leads on Brier and score MAE. Jev48 has lower ECE and KL to the benchmark's soft targets. Those distribution metrics do not erase Jev's substantial accuracy lead.
