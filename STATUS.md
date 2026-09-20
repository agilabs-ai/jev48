# Status

## Completed

- code-level audit of major public Jev-like implementations;
- pinned public starting point (`Mapika/decider`, commit `b08acf...`);
- MT-Bench expert-vote aggregation with question-level leakage-safe splits;
- held-out transfer suite bridge;
- replay/regression suite bridge;
- empirical soft-target training objective;
- hard-vs-soft development comparison;
- dev-only candidate-selection gate;
- identical post-hoc temperature fitting;
- resumable TypeSafe Jev client;
- paired bootstrap script;
- machine-generated static launch page;
- Modal end-to-end runner;
- local unit/integrity tests;
- frozen H200 experiment and dev-only selection;
- locked base-versus-Jev48 test/OOD evaluation;
- persisted derivative checkpoint, private row-level predictions, manifests, and launch page;
- final receipt audit with no secret hits.
- zero-shot run on the pinned `LocalLLaMA/typed-decisions` public benchmark;
- machine-generated aggregate comparison against its published Jev 1.13.0 row.

## Locked open-model result

- selected derivative: `soft-lr3e-6`;
- preference test: 62.1% accuracy, 0.4111 Brier;
- transfer OOD: 79.3% accuracy, 0.3039 Brier;
- successful H200 list-price estimate: $2.87;
- live Jev outputs used for training or selection: 0.

## Remaining model publication gate

The source repository and safe aggregate receipts are public at
https://github.com/agilabs-ai/jev48. The Hugging Face checkpoint remains private /
not uploaded until the AGI Labs namespace is authenticated, the upload is
re-downloaded by immutable revision, every hash is verified, and inference passes.
Mixed-source row-level data remains withheld.

Do not describe the public Jev row as paired or as a run on Jev48's original benchmark.
The allowed comparison is the explicitly labeled independent aggregate in `RESULTS_PUBLIC.md`.
