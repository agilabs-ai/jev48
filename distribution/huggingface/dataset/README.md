---
license: mit
task_categories:
- text-classification
tags:
- benchmark
- decision-model
- jev
- reproducible-research
pretty_name: Jev48 Public Benchmark Results
size_categories:
- 1K<n<10K
---

# Jev48 Public Benchmark Results

Machine-readable summary receipts for the six eligible reproducible public benchmark comparisons in the Jev48 experiment.

The package contains aggregate JSON summaries, source and run manifests, hashes, and the public methodology. It does not republish restricted row-level benchmark inputs or predictions.

- Interactive report: https://getedge.cc/jev48/
- Source repository: https://github.com/edgelabs-ai/jev48
- Benchmark registry: https://github.com/edgelabs-ai/jev48/blob/main/BENCHMARK_REGISTRY.md
- Methodology: https://github.com/edgelabs-ai/jev48/blob/main/RESULTS_PUBLIC.md
- Model release: https://github.com/edgelabs-ai/jev48/releases/tag/v1.0.0

## Scope

The six summaries cover 6,300 evaluated decisions or cases: Typed decisions, PhishNChips v5.2, JevBench public v1.2.2, BTZSC, code review, and CLASH conflicts.

Comparison protocols differ by suite. Consumers must preserve the paired, aggregate/unpaired, and repeated-reference labels from the source receipts.

## Licensing

Original Jev48 code and documentation are MIT licensed. Upstream datasets, benchmark inputs, models, and other components retain their original terms. See `THIRD_PARTY_NOTICES.md` in the source repository.
