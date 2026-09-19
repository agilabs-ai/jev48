# Benchmark Policy

## Why this exists

A Jev comparison is extremely easy to fool ourselves on. This file defines the rules before paid target-model evaluation.

## Rules

1. **Freeze first.** Hash the final benchmark before querying Jev.
2. **No target tuning.** Never change benchmark examples because of Jev results.
3. **No generator monoculture.** Final evaluation cannot be produced only by the same LLM/template source used for training.
4. **Keep related families together.** Paraphrases or related source rows cannot cross splits.
5. **Report soft-probability metrics.** Accuracy alone is insufficient.
6. **Report OOD separately.** Do not hide OOD collapse in a large IID aggregate.
7. **Report systems metrics separately from model quality.** Quality and latency are different axes.
8. **Raw outputs are artifacts.** Publish predictions where licensing/privacy permits.
9. **No secret exclusions after results.** Predefine invalid-example criteria.
10. **Version instead of editing.** Any benchmark change creates a new benchmark version and hash.

## Current semantic smoke set

`data/benchmark/semantic_smoke.jsonl` is only an engineering smoke set. It is **not** the final publishable benchmark and should not be used for broad intelligence claims.
