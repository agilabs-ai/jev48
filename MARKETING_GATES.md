# OpenJev launch gates

OpenJev is derived from NanoJev, so the launch story must be a **before/after improvement**, not “first OSS Jev.”

## Preferred claim

Only if the frozen benchmark supports it:

> **I gave ChatGPT NanoJev and told it to make it better. OpenJev beats the original on a frozen semantic decision benchmark while keeping the same 600M architecture.**

## Primary gate

On exactly the same frozen benchmark:

### Pass if either

1. OpenJev has lower Brier than NanoJev and accuracy is no more than 2 percentage points lower; or
2. OpenJev accuracy is at least 5 percentage points higher without a major Brier regression.

## OOD gate

Report fully held-out domains separately.

A specific OOD-improvement claim requires a meaningful calibration improvement while retaining useful top-1 accuracy.

## Do not claim “better” from

- a benchmark changed after seeing NanoJev outputs
- different candidate order
- different test rows
- unmatched hardware latency
- one cherry-picked domain
- synthetic-only test data

## If OpenJev loses

Do not change the benchmark.

The next experiment should address measured failure slices, most likely catastrophic forgetting or data balance, while retaining base NanoJev as the comparator.
