# Third-party notices

## Mapika/decider

Jev48's default starting model and implementation lineage.

```text
repo:   https://github.com/Mapika/decider
commit: b08acf787d5d1f718a8c36c4677960f43772c7be
model:  Mapika/decider-2b
model revision: 4a0e86782adfdb7393e04b8ec9f6b939dca09273
license: Apache-2.0
```

Source: https://github.com/Mapika/decider/tree/b08acf787d5d1f718a8c36c4677960f43772c7be
License text: [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt)

Jev48 does not claim the upstream architecture, base training corpus, inference engine, or initial weights as original work.

## Qwen

The Mapika model card identifies `Qwen/Qwen3.5-2B-Base` as its base and Apache-2.0
as the applicable license. The exact Qwen revision used by Mapika was not disclosed;
Jev48 does not invent one.

Citation: Qwen Team, “Qwen3.5: Towards Native Multimodal Agents,” February 2026,
https://qwen.ai/blog?id=qwen3.5

## MT-Bench / LMSYS

Dataset: `lmsys/mt_bench_human_judgments`, revision
`ee34b9d273a7a35e4415c87678526c56c471098c`, CC BY 4.0. Jev48 modifies it by
grouping expert votes, replacing one genuine empty response with an explicit marker,
and assigning outcome-blind question-level splits. Mixed row-level derivatives are
not redistributed in this release.

Source: https://huggingface.co/datasets/lmsys/mt_bench_human_judgments/tree/ee34b9d273a7a35e4415c87678526c56c471098c
License: https://creativecommons.org/licenses/by/4.0/
The source material is provided without warranties; see Section 5 of CC BY 4.0.

Citation: Lianmin Zheng et al., “Judging LLM-as-a-Judge with MT-Bench and Chatbot
Arena,” arXiv:2306.05685 (2023).

## LocalLLaMA/typed-decisions

Independent public comparison dataset, revision
`ea9306458d6e9563628369a3d1e72e362fb381d2`, Apache-2.0. Jev48 evaluates the
test split zero-shot and publishes aggregate results plus an unmodified snapshot of
the source card. Its synthetic targets are means of teacher-model samples.

Source: https://huggingface.co/datasets/LocalLLaMA/typed-decisions/tree/ea9306458d6e9563628369a3d1e72e362fb381d2
License text: [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt)

## TypeSafe Jev

“Jev” is used descriptively as the closed target/reference system. Jev48 is independent and not affiliated with TypeSafe.

## JevBench

Independent decision benchmark, revision
`e105a48f8cdb7f3babb3594424f73e5d7bdc97b9`. Jev48 evaluates all 231 public
tasks (48 easy, 72 original, 111 hard) and compares against source-published Jev
1.13.0 outcomes. JevBench's MIT grant expressly covers its harness and 72 original
decisions only; its other public tasks retain their respective source terms.
Consequently Jev48 publishes aggregate summaries, source hashes, and code—not task
state/question text or row-level expected labels/families for the other 159 tasks.
It does not claim a score on the benchmark's private tasks.

Source: https://github.com/fstandhartinger/jevbench/tree/e105a48f8cdb7f3babb3594424f73e5d7bdc97b9

## PhishNChips v5.2 and jev-phishing-bench

PhishNChips v5.2 declares `license: other` and carries mixed source-specific
terms: project-generated synthetic content is MIT; Nazario and parts of the GitHub
Phishing Database are CC BY 4.0; OpenPhish URLs have academic-research permission;
PhishTank URLs follow its terms; and Tranco seeds require research attribution.
The pinned `SOURCE_LICENSES.md` is redistributed in `receipts/upstream/`.

Jev48 publishes only case IDs, binary labels, its own probabilities, and timings—
not email bodies or URLs. It uses the exact nine-question request from
`jev-phishing-bench` at commit
`1d56e8c64d029a9554a0874e2ef2901ed196e230`. Jev's comparison row is aggregate
and unpaired because its row-level predictions were not published.

Sources: https://huggingface.co/datasets/AreLit/PhishNChips and
https://github.com/anisselbd/jev-phishing-bench/tree/1d56e8c64d029a9554a0874e2ef2901ed196e230

## Mixed transfer and replay sources

The pinned Mapika registry loads 17 transfer tasks and four replay/regression tasks.
Their row-level combined derivative is withheld until every source's redistribution
terms are individually audited. Public receipts contain only hashes, counts, source
identifiers, and aggregate metrics.

## Other public prior art

The research process also inspected public work including `jaredpalmer/kev`, `TianyuCodings/NanoJev`, `TheoLeeCJ/SemIf`, `razorback16/openjev`, and `daseinlabs/open-jev`. See `RELATED_WORK.md`.
