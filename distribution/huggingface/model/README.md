---
license: mit
library_name: pytorch
pipeline_tag: text-classification
tags:
- decision-model
- probabilistic-classification
- system-one
- jev
- reproducible-research
base_model: Mapika/decider-2b
---

# Jev48-2B

Jev48-2B is the selected checkpoint from an open, auditable weekend reproduction of TypeSafe's Jev typed probabilistic decision model. It is based on `Mapika/decider-2b` and fine-tuned with empirical human-vote distributions from pinned MT-Bench judgments.

The experiment, checkpoint, source, benchmark adapters, frozen receipts, and all six eligible reproducible public benchmark results are open.

- Project: https://getedge.cc/jev48/
- Source: https://github.com/edgelabs-ai/jev48
- Release: https://github.com/edgelabs-ai/jev48/releases/tag/v1.0.0
- Full model card: https://github.com/edgelabs-ai/jev48/blob/main/MODEL_CARD.md
- Public methodology: https://github.com/edgelabs-ai/jev48/blob/main/RESULTS_PUBLIC.md
- Receipts: https://github.com/edgelabs-ai/jev48/tree/main/receipts

## Public benchmark scorecard

| Benchmark | Cases | Published Jev | Jev48 | Protocol |
|---|---:|---:|---:|---|
| Typed decisions | 2,000 | 72.7% | 57.7% | Aggregate / unpaired |
| PhishNChips v5.2 | 2,000 | 62.6% | 50.0% | Aggregate / unpaired |
| JevBench public v1.2.2 | 231 | 86.6% | 69.7% | Paired outcomes |
| BTZSC pilot | 300 | 75.3% | 83.3% | Aggregate / unpaired |
| Code review | 480 | 99.0% | 81.9% | Repeated Jev reference |
| CLASH conflicts | 1,289 | 98.6% | 0.0% | Aggregate / unpaired |

Jev48 wins one of six suites across 6,300 evaluated decisions or cases. On PhishNChips, Jev48 records AUROC 0.769 versus the source-published Jev result of 0.689 despite lower fixed-threshold accuracy.

## Important limitations

- Jev48 is independent and not affiliated with TypeSafe.
- Private Jev access was unavailable; not every comparison is a paired same-row evaluation.
- Jev outputs were not used for training or model selection.
- The locked result does not establish a causal hard-label versus soft-label ablation.
- Review the complete model card and third-party notices before use or redistribution.

## Citation

See https://github.com/edgelabs-ai/jev48/blob/main/CITATION.cff.
