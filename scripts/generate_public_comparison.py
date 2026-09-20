#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the third-party Jev comparison from public receipts.")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    predictions = Path(args.predictions)
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines() if line]
    receipt = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    ours, jev = receipt["metrics"], receipt["jev_reference"]
    median_latency = statistics.median(float(row["latency_ms"]) for row in rows)
    raw_hash = hashlib.sha256(predictions.read_bytes()).hexdigest()
    text = f"""# Jev48 public benchmark comparison

Machine-generated from the pinned raw predictions. Do not edit numbers by hand.

This is an **aggregate comparison on an independent public benchmark**, not a paired
run against Jev on Jev48's original frozen rows. Jev's row was measured and published
by the `LocalLLaMA/typed-decisions` maintainers; Jev48 was run separately on the exact
pinned test split. Jev48 is evaluated zero-shot and used none of this benchmark's train rows.

| Model | Kind | Accuracy ↑ |
|---|---|---:|
| TypeSafe Jev 1.13.0 | source-published aggregate | {jev['accuracy']:.3f} |
| Jev48 / `soft-lr3e-6` | reproduced zero-shot aggregate | {ours['accuracy']:.3f} |

- Benchmark: `LocalLLaMA/typed-decisions` at `{receipt['benchmark_revision']}`.
- Test size: {len(rows)} cases / {ours['n_decisions']} decisions.
- Jev48 median observed latency: {median_latency:.1f} ms/case on Modal L40S; published Jev p50: {jev['median_case_latency_ms']:.0f} ms/case. Hardware/network differ, so this is not a controlled latency claim.
- Jev48 raw prediction SHA-256: `{raw_hash}`.
- Public benchmark: https://huggingface.co/datasets/LocalLLaMA/typed-decisions

The benchmark's gold distribution is the mean of three samples from a separate
~4B teacher model. It therefore measures teacher agreement, not real-world
correctness. The upstream card reports other metrics but ships no scorer code; they
are preserved in the machine receipt but are not asserted as cross-system-comparable.

## Honest launch claim

On this public zero-shot benchmark, Jev leads Jev48 by {(jev['accuracy'] - ours['accuracy']) * 100:.1f} percentage points in accuracy. This is descriptive: no paired Jev predictions are available for a significance test.
"""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
