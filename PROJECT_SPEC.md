# Jev48 project spec

## Question

**How close can ChatGPT get to TypeSafe's Jev in one weekend if it may use everything publicly available?**

This is not a from-scratch reproduction claim.

## Public starting point

After auditing public Jev-like projects, Jev48 pins:

```text
Mapika/decider
commit b08acf787d5d1f718a8c36c4677960f43772c7be
model Mapika/decider-2b
```

## Jev48 modification

Test whether preference calibration improves when multiple human judgments are preserved as an empirical target distribution rather than reduced to one majority label.

Primary preference source: pinned MT-Bench human judgments over real model responses.

## Evaluation

Two locked axes:

1. **Human preference** — unseen MT-Bench question groups, scored against empirical expert-vote distributions.
2. **Transfer OOD** — public tasks marked held-out/evaluation-only by the pinned upstream training registry.

Models in final table:

- untouched public `decider-2b` starting point;
- selected Jev48 reproduction;
- native Jev, queried only after selection is frozen.

Metrics: accuracy, multiclass Brier, NLL/ECE where meaningful, plus paired bootstrap comparisons.

## Naming

- Experiment: **Jev48**
- Derivative checkpoint: **`jev48-2b`**, only if a derivative wins selection.
- If the public base wins, no new model checkpoint is branded as Jev48.

## Integrity

See `INTEGRITY.md` and `BENCHMARK_POLICY.md`.
