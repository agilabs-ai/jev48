# Jev48 project spec

## Objective

Measure what an AI coding/research agent can reproduce from Jev within a weekend when it may use any public information and open-source work.

## Primary KPI

A clear, defensible direct comparison with Jev that a non-specialist can understand quickly, without sacrificing methodological integrity.

## Technical strategy

1. Pin the strongest public base found during the audit (`Mapika/decider-2b`).
2. Add one substantive, measurable change: empirical human-vote distribution supervision.
3. Keep Jev entirely out of the training/selection loop.
4. Evaluate on a frozen human-preference test and broad held-out transfer suite.
5. Publish raw receipts, hashes, costs, elapsed time, and upstream attribution.

## Success is not defined as “beat Jev”

Useful outcomes include close reproduction, a clear remaining gap, or evidence that calibration/generalization is the hard part. The launch language is selected after the locked results exist.
