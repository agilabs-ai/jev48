# Results

The frozen Modal run `jev48-1789867911` selected `soft-lr3e-6` using development
rows only. Relative to the untouched base on the locked MT-Bench preference slice,
Jev48 changed accuracy from 58.19% to 62.07% and Brier from 0.4708 to 0.4111.

| Locked slice | Metric delta (Jev48 − base) | 95% paired bootstrap interval |
|---|---:|---:|
| preference test | accuracy +0.0388 | −0.0081 to +0.0921 |
| preference test | Brier −0.0597 | −0.1000 to −0.0208 |
| transfer OOD | accuracy −0.0074 | −0.0147 to −0.0015 |
| transfer OOD | Brier +0.0044 | −0.0015 to +0.0105 |

Negative Brier is better. Preference Brier improved clearly; preference accuracy is
uncertain; transfer accuracy regressed slightly; transfer Brier is inconclusive.
Only the winning soft-label trial was evaluated on locked rows, so this is not a
locked hard-vs-soft causal ablation.

On the independent `LocalLLaMA/typed-decisions` benchmark, Jev's source-published
accuracy is 72.7% and Jev48's reproduced zero-shot accuracy is 57.7%. This unpaired
synthetic teacher-agreement benchmark does not measure real-world correctness. See
[`RESULTS_PUBLIC.md`](RESULTS_PUBLIC.md) and [`receipts/`](receipts/).

## Additional public Jev comparisons

Two more complete public suites were run zero-shot after launch preparation. They
were not used for training or candidate selection, and the earlier unfavorable
typed-decisions result remains published.

| Benchmark | Metric | TypeSafe Jev | Jev48 | Difference |
|---|---:|---:|---:|---:|
| PhishNChips v5.2 (2,000 emails) | AUROC | 0.689 | 0.769 | +0.081 |
| PhishNChips v5.2 (2,000 emails) | accuracy at 0.5 | 62.6% | 50.0% | −12.6 pp |
| JevBench v1.2.2 public (231 tasks) | accuracy | 86.6% | 69.7% | −16.9 pp |
| JevBench easy tier (all 48 public tasks) | accuracy | 100.0% | 100.0% | tied |

The phishing result is a ranking result, not decision parity: Jev48's descriptive
AUROC is higher, but it assigned nearly every phishing probability below 0.5,
producing 0.1% recall at the fixed threshold. Jev's row-level outputs are unavailable,
so this comparison is aggregate and unpaired with no interval for the difference.
On JevBench, the easy tier is a pre-defined full tier,
but the complete 231-task public comparison remains the primary result. The paired
family-cluster bootstrap interval for Jev48 minus Jev accuracy over all 231 public tasks is −26.1 to
−8.6 percentage points. Public receipts contain aggregate summaries and pinned-input
hashes; row-level predictions remain in the private Modal receipt tree because the
benchmark inputs carry source-specific redistribution terms.
