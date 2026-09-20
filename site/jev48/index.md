# Jev48: Open Jev Reproduction and Benchmark

Jev48 is an open, auditable reproduction of TypeSafe's Jev typed probabilistic decision model. ChatGPT built it in one weekend using public information, open-source code, and external compute. Edge Labs AI publishes the 2B checkpoint, source code, benchmark adapters, frozen receipts, and every eligible reproducible public result.

Jev48 is independent and not affiliated with TypeSafe. Jev outputs were not used for training or model selection.

## Complete public scorecard

| Benchmark | Cases | Jev | Jev48 | Delta | Protocol |
|---|---:|---:|---:|---:|---|
| Typed decisions | 2,000 | 72.7% | 57.7% | -15.0 pts | Aggregate / unpaired |
| PhishNChips v5.2 | 2,000 | 62.6% | 50.0% | -12.6 pts | Aggregate / unpaired |
| JevBench public v1.2.2 | 231 | 86.6% | 69.7% | -16.9 pts | Paired outcomes |
| BTZSC pilot | 300 | 75.3% | 83.3% | +8.0 pts | Aggregate / unpaired |
| Code review | 480 | 99.0% | 81.9% | -17.2 pts | Repeated Jev reference |
| CLASH conflicts | 1,289 | 98.6% | 0.0% | -98.6 pts | Aggregate / unpaired |

Across 6,300 evaluated decisions or cases, Jev48 wins one of six suites. On PhishNChips, Jev48 records AUROC 0.769 versus the source-published Jev result of 0.689, despite lower fixed-threshold accuracy.

## Canonical resources

- Project: https://getedge.cc/jev48/
- Source: https://github.com/edgelabs-ai/jev48
- Model: https://github.com/edgelabs-ai/jev48/releases/tag/v1.0.0
- Methodology: https://github.com/edgelabs-ai/jev48/blob/main/RESULTS_PUBLIC.md
- Benchmark registry: https://github.com/edgelabs-ai/jev48/blob/main/BENCHMARK_REGISTRY.md
- Receipts: https://github.com/edgelabs-ai/jev48/tree/main/receipts
- Citation: https://github.com/edgelabs-ai/jev48/blob/main/CITATION.cff
- Licenses: https://github.com/edgelabs-ai/jev48/blob/main/THIRD_PARTY_NOTICES.md
