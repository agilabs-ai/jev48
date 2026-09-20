# Published receipts

This directory contains the aggregate, provenance, selection, calibration, and
bootstrap receipts safe to publish with Jev48. `SHA256SUMS` covers every file.

Row-level mixed-dataset inputs and predictions are intentionally withheld pending
a per-source redistribution audit. Their hashes, row counts, builders, and source
identifiers remain published. The complete private receipt tree is preserved in
Modal volume `jev48-artifacts`, run `jev48-1789867911`.

`upstream/typed-decisions-README.md` is the immutable Apache-2.0 benchmark card at
revision `ea9306458d6e9563628369a3d1e72e362fb381d2`. Jev48 does not possess Jev's
row-level predictions, so that comparison is descriptive and unpaired. Only
accuracy is treated as directly comparable; the source card ships no scorer code
with which to prove parity for its reported distribution metrics.

The additional public summaries cover all 2,000 PhishNChips v5.2 emails and all
231 public JevBench v1.2.2 tasks. Row-level predictions remain in the private Modal
receipt tree because both benchmarks carry source-specific redistribution terms.
Source revisions, input hashes, comparison limitations, and uncertainty intervals
are embedded in the summaries. Pinned upstream metric and mixed-source licensing
evidence is bundled under `upstream/`.
