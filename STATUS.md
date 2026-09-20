# Status

## Completed

- code-level audit of major public Jev-like implementations;
- pinned public starting point (`Mapika/decider`, commit `b08acf...`);
- MT-Bench expert-vote aggregation with question-level leakage-safe splits;
- held-out transfer suite bridge;
- replay/regression suite bridge;
- empirical soft-target training objective;
- hard-vs-soft controlled ablation;
- dev-only candidate-selection gate;
- identical post-hoc temperature fitting;
- resumable TypeSafe Jev client;
- paired bootstrap script;
- machine-generated static launch page;
- Modal end-to-end runner;
- local unit/integrity tests;
- frozen H200 experiment and dev-only selection;
- locked base-versus-Jev48 test/OOD evaluation;
- persisted derivative checkpoint, raw predictions, manifests, and launch page;
- final receipt audit with no secret hits.
- zero-shot run on the pinned `LocalLLaMA/typed-decisions` public benchmark;
- machine-generated aggregate comparison against its published Jev 1.13.0 row.

## Locked open-model result

- selected derivative: `soft-lr3e-6`;
- preference test: 62.1% accuracy, 0.4110 Brier;
- transfer OOD: 79.3% accuracy, 0.3039 Brier;
- successful H200 list-price estimate: $2.87;
- live Jev outputs used for training or selection: 0.

## External work remaining

1. stage the complete public receipts and `jev48-2b` checkpoint for release;
2. run the final release/license audit over the staged package;
3. publish the repository, derivative, receipts, and measured result page when authorized.

Do not describe the public Jev row as paired or as a run on Jev48's original benchmark.
The allowed comparison is the explicitly labeled independent aggregate in `RESULTS_PUBLIC.md`.
