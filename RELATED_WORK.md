# Related work discovered during the challenge

The Jev-like open ecosystem formed almost immediately after Jev's release. These projects are related work, controls, and public prior art — not the consumer-facing story of Jev48.

## Mapika/decider

https://github.com/Mapika/decider

Selected as the primary starting point after code-level audit. Qwen3.5-2B, typed one-pass decisions, 2–255 options, public training pipeline, calibration evaluation, System One-compatible API, Apache-2.0.

Pinned by Jev48 at:

```text
b08acf787d5d1f718a8c36c4677960f43772c7be
```

## jaredpalmer/kev

https://github.com/jaredpalmer/kev

Qwen2.5-0.5B + LoRA/readout. Particularly important because it already publishes a live Jev comparison: close on its shared-task dev set, much further behind on transfer. That reinforced Jev48's decision to make transfer a first-class locked benchmark.

## TianyuCodings/NanoJev

https://github.com/TianyuCodings/NanoJev

0.6B dedicated decision-head implementation with full training/evaluation pipeline. This was the first base considered in the project; the git history records that path before the broader ecosystem audit changed the decision.

## TheoLeeCJ/SemIf

https://github.com/TheoLeeCJ/SemIf

Direct option-logit and reranking work with careful evaluation methodology. It also reconstructs a 102-row subset from TypeSafe's published evaluation artifacts where source snapshots are available.

## razorback16/openjev

https://github.com/razorback16/openjev

Jev-compatible decision server over DiffusionGemma. Useful systems reference.

## daseinlabs/open-jev

https://github.com/daseinlabs/open-jev

Prefix-shared option scoring over Gemma. Useful evidence that raw option probabilities can be badly overconfident even when the decision interface looks Jev-like.

## TypeSafe Jev

https://typesafe.ai/

Closed reference system and target of the experiment. Jev48 is not affiliated with TypeSafe.
