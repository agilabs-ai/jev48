# Results

The frozen Modal run `jev48-1789858979` selected `soft-lr3e-6` using development
rows only. Relative to the untouched base on the locked MT-Bench preference slice,
Jev48 changed accuracy from 58.19% to 62.07% and Brier from 0.4708 to 0.4110.

| Locked slice | Metric delta (Jev48 − base) | 95% paired bootstrap interval |
|---|---:|---:|
| preference test | accuracy +0.0388 | −0.0081 to +0.0921 |
| preference test | Brier −0.0598 | −0.1003 to −0.0207 |
| transfer OOD | accuracy −0.0074 | −0.0147 to −0.0015 |
| transfer OOD | Brier +0.0043 | −0.0017 to +0.0105 |

Negative Brier is better. Preference Brier improved clearly; preference accuracy is
uncertain; transfer accuracy regressed slightly; transfer Brier is inconclusive.
Only the winning soft-label trial was evaluated on locked rows, so this is not a
locked hard-vs-soft causal ablation.

On the independent `LocalLLaMA/typed-decisions` benchmark, Jev's source-published
accuracy is 72.7% and Jev48's reproduced zero-shot accuracy is 57.7%. This unpaired
synthetic teacher-agreement benchmark does not measure real-world correctness. See
[`RESULTS_PUBLIC.md`](RESULTS_PUBLIC.md) and [`receipts/`](receipts/).
