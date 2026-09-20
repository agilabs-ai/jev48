# Jev48 public benchmark comparison

Machine-generated from the pinned raw predictions. Do not edit numbers by hand.

This is an **aggregate comparison on an independent public benchmark**, not a paired
run against Jev on Jev48's original frozen rows. Jev's row was measured and published
by the `LocalLLaMA/typed-decisions` maintainers; Jev48 was run separately on the exact
pinned test split. Jev48 is evaluated zero-shot and used none of this benchmark's train rows.

| Model | Kind | Accuracy ↑ |
|---|---|---:|
| TypeSafe Jev 1.13.0 | source-published aggregate | 0.727 |
| Jev48 / `soft-lr3e-6` | reproduced zero-shot aggregate | 0.577 |

- Benchmark: `LocalLLaMA/typed-decisions` at `ea9306458d6e9563628369a3d1e72e362fb381d2`.
- Test size: 400 cases / 2000 decisions.
- Jev48 median observed latency: 250.5 ms/case on Modal L40S; published Jev p50: 710 ms/case. Hardware/network differ, so this is not a controlled latency claim.
- Jev48 raw prediction SHA-256: `db712a50593c0cbfff6b464a26159bae5447ef59d73919bd2fc217ed9197d2ad`.
- Public benchmark: https://huggingface.co/datasets/LocalLLaMA/typed-decisions

The benchmark's gold distribution is the mean of three samples from a separate
~4B teacher model. It therefore measures teacher agreement, not real-world
correctness. The upstream card reports other metrics but ships no scorer code; they
are preserved in the machine receipt but are not asserted as cross-system-comparable.

## Honest launch claim

On this public zero-shot benchmark, Jev leads Jev48 by 15.0 percentage points in accuracy. This is descriptive: no paired Jev predictions are available for a significance test.
# Complete public Jev benchmark scorecard

The launch scorecard includes every suite in [`BENCHMARK_REGISTRY.md`](BENCHMARK_REGISTRY.md).
The three additional full-suite runs were frozen in commits `9c242f5` and `b600a64`
before their respective paid outcomes.

| Benchmark | Jev48 | Published Jev | Population | Comparison |
|---|---:|---:|---:|---|
| LocalLLaMA/typed-decisions | 57.7% | 72.7% | 2,000 decisions | aggregate, unpaired |
| PhishNChips v5.2 | 50.1% accuracy / .769 AUROC | 62.6% / .689 | 2,000 emails | aggregate, unpaired |
| JevBench v1.2.2 public | 69.7% | 86.6% | 231 tasks | paired public outcomes |
| BTZSC pilot v1 | **83.3%** | 75.3% | 300 texts | aggregate, identical sampling |
| Determinest code review | 81.9% | 99.0% | 480 rule decisions | Jev reference aggregates 3 rounds |
| CLASH text contradiction | 0.0% | 98.6% | 1,289 cases | aggregate, identical main condition |

BTZSC is the one full-suite accuracy win. On PhishNChips, Jev48 has higher AUROC
but worse thresholded decisions. The CLASH result is a real failure: the model puts
nearly all probability on one source-grounded answer instead of the explicit
contradiction option. No suite or result was removed after evaluation.

The additional GPU subprocess runtimes totaled 416.3 L40S-seconds including one
discarded CLASH adapter attempt. At the recorded Modal L40S rate of $0.000542/s,
that is $0.226 of measured GPU runtime; allowing conservatively for container startup
keeps this benchmark round below $0.40, versus the authorized $10 ceiling.
