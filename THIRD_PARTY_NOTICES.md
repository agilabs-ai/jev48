# Third-party notices

Jev48's own orchestration/evaluation code is MIT-licensed. It does not vendor the upstream model source in the repository; Modal clones pinned upstream code during image construction.

## Mapika/decider

Primary public starting point and runtime/training implementation.

- repository: https://github.com/Mapika/decider
- pinned commit: `b08acf787d5d1f718a8c36c4677960f43772c7be`
- weights: `Mapika/decider-2b`
- upstream license: Apache-2.0

Any redistributed derivative model weights/code must preserve the obligations of the upstream license and of the Qwen base model used by decider.

## MT-Bench human judgments

- dataset: `lmsys/mt_bench_human_judgments`
- pinned revision: `ee34b9d273a7a35e4415c87678526c56c471098c`
- dataset card/license must be reviewed again before redistributing transformed rows publicly.

Jev48 records provenance and can publish generation scripts/manifests instead of source text if redistribution terms require it.

## Other datasets

The transfer/replay suites are generated from public datasets referenced by `Mapika/decider`. Each source retains its own license. Before publishing transformed raw text, run the release audit and either:

1. redistribute only sources whose terms permit it; or
2. publish deterministic builder scripts + IDs/hashes instead of the text.

## TypeSafe Jev

“Jev” and “System One” are used descriptively for comparison. Jev48 is not affiliated with TypeSafe. TypeSafe API outputs should be redistributed only to the extent permitted by applicable terms; otherwise publish aggregate metrics and cryptographic receipts/IDs rather than prohibited raw content.
