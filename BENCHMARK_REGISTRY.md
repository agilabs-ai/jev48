# Public Jev benchmark registry

Frozen 2026-09-19 before the additional paid runs. A suite is eligible only when
its complete evaluation population, labels, Jev reference result, and native
request mapping are public and reproducible. No result-dependent subsets are used.

## Included

| Benchmark | Public population | Jev reference | Comparison |
|---|---:|---:|---|
| LocalLLaMA/typed-decisions | 400 cases / 2,000 decisions | 72.7% | same public test split; aggregate reference |
| PhishNChips v5.2 | 2,000 emails | 62.6% accuracy / .689 AUROC | exact nine-question protocol; aggregate reference |
| JevBench v1.2.2 | 231 public tasks | 86.6% | paired public task outcomes |
| BTZSC pilot v1 | 300 texts | 75.3% macro accuracy | exact pinned sampling and native choice protocol |
| Determinest code review | 24 evaluation families / 480 rule decisions | 99.0% raw decision accuracy | same cases and primary rule protocol; Jev aggregates three rounds |
| CLASH contradiction detection | 1,289 cases | 98.6% | same published text-vs-text main condition |

## Audited exclusions

| Candidate | Reason excluded |
|---|---|
| `anpicasso/hermes-jev-approvals` | The 153-case corpus is mined from private Hermes history and intentionally not published. |
| `kunchenguid/no-mistakes` issue studies | Measures downstream launch behavior and agent-specific prebriefs, not a standalone Jev decision population that Jev48 can replay equivalently. |
| `browser-use/jev-ultrafast` | Browser-agent latency study; no Jev quality reference population. |
| `kyotofin/tax-doc-classifier` | Evaluation downloads a changing IRS corpus at runtime; the repository does not publish a pinned row-level Jev reference artifact for a paired replay. |
| `Zaious/jev-capability-atlas` | Capability demonstrations and latency suites, not a single frozen comparative quality benchmark with complete Jev labels. |
| `iammrduncan/typesafe-ai-benchmark` | Stateful theater trajectories diverge after earlier decisions and the published Jev run is a single browser session; replaying only favorable static scenes would violate the full-suite rule. |

Exclusion is methodological, not a judgment about project quality. Repositories can
become eligible in a later registry version if they publish the missing frozen
artifacts or an equivalent replay protocol.
