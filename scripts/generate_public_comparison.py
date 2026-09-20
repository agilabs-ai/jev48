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

| Model | Kind | Accuracy ↑ | Soft acc ↑ | Brier ↓ | KL ↓ | ECE ↓ | Score MAE ↓ | Within 1 ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TypeSafe Jev 1.13.0 | published general-model row | {jev['accuracy']:.3f} | {jev['soft_accuracy']:.3f} | {jev['brier']:.3f} | {jev['kl']:.3f} | {jev['ece_15']:.3f} | {jev['score_mae']:.3f} | {jev['within_one_level']:.3f} |
| Jev48 / `soft-lr3e-6` | reproduced zero-shot general model | {ours['accuracy']:.3f} | {ours['soft_accuracy']:.3f} | {ours['brier']:.3f} | {ours['kl']:.3f} | {ours['ece_15']:.3f} | {ours['score_mae']:.3f} | {ours['within_one_level']:.3f} |

- Benchmark: `LocalLLaMA/typed-decisions` at `{receipt['benchmark_revision']}`.
- Test size: {len(rows)} cases / {ours['n_decisions']} decisions.
- Jev48 median observed latency: {median_latency:.1f} ms/case on Modal L40S; published Jev p50: {jev['median_case_latency_ms']:.0f} ms/case. Hardware/network differ, so this is not a controlled latency claim.
- Jev48 raw prediction SHA-256: `{raw_hash}`.
- Public benchmark: https://huggingface.co/datasets/LocalLLaMA/typed-decisions

## Honest launch claim

On this public zero-shot benchmark, Jev leads Jev48 by {(jev['accuracy'] - ours['accuracy']) * 100:.1f} percentage points in accuracy and also leads on Brier and score MAE. Jev48 has lower ECE and KL to the benchmark's soft targets. Those distribution metrics do not erase Jev's substantial accuracy lead.
"""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
