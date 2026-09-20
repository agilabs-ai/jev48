#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def f(value) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a derivative model card from frozen Jev48 receipts.")
    ap.add_argument("--selection", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--public-summary")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    sel, summary, source = read(args.selection), read(args.summary), read(args.source)
    cal = summary["calibrated"]
    test, ood = cal.get("by_split", {}).get("test", {}), cal.get("by_split", {}).get("ood", {})
    public_block = ""
    if args.public_summary:
        public = read(args.public_summary)
        ours, jev = public["metrics"], public["jev_reference"]
        public_block = f"""
## Independent public Jev benchmark

This is an aggregate, unpaired comparison on `LocalLLaMA/typed-decisions`, not a
live Jev run on Jev48's original rows. The benchmark maintainers published the
Jev row; Jev48 ran separately, zero-shot, on the pinned test split.

| Model | Accuracy ↑ | Brier ↓ | KL ↓ | ECE ↓ |
|---|---:|---:|---:|---:|
| TypeSafe Jev 1.13.0 (published) | {f(jev['accuracy'])} | {f(jev['brier'])} | {f(jev['kl'])} | {f(jev['ece_15'])} |
| jev48-2b (zero-shot) | {f(ours['accuracy'])} | {f(ours['brier'])} | {f(ours['kl'])} | {f(ours['ece_15'])} |

Benchmark revision: `{public['benchmark_revision']}`. Jev leads accuracy and Brier;
Jev48 has lower reported KL and ECE.
"""
    text = f"""---
license: apache-2.0
base_model: Mapika/decider-2b
pipeline_tag: text-classification
library_name: transformers
tags:
- decision-model
- calibration
- system-one
- jev48
---
# jev48-2b

This checkpoint is the derivative selected by the **Jev48** weekend reproduction experiment.
It derives from **Mapika/decider-2b**; the upstream architecture and starting weights
are not Jev48 inventions. Jev48 tests empirical soft human-vote distributions against
hard-majority preference labels while retaining replay data.

## Lineage

- Starting model: `{source['base_model']}`
- Upstream repository: `{source['upstream_repo']}`
- Upstream commit: `{source['upstream_commit']}`
- Selected trial: `{sel['winner']['name']}`
- Benchmark SHA-256: `{summary['benchmark_sha256']}`
- Fitted temperature: `{summary['temperature']}`

## Locked evaluation

| Slice | Accuracy ↑ | Brier ↓ | ECE ↓ |
|---|---:|---:|---:|
| MT-Bench expert-vote test | {f(test.get('accuracy'))} | {f(test.get('brier'))} | {f(test.get('ece_15'))} |
| Held-out transfer OOD | {f(ood.get('accuracy'))} | {f(ood.get('brier'))} | {f(ood.get('ece_15'))} |

Temperature was fitted only on the frozen calibration split after model selection.
Locked test/OOD were not used for selection. Jev outputs were not used for training.
{public_block}
## Intended use

Typed, generation-free decision scoring through the upstream `decider` interface.
Validate calibration on your own outcome-labelled workload before consequential use.

## Attribution and licenses

Jev48 code is MIT. This model derives from `Mapika/decider-2b`; preserve all applicable
Apache-2.0 and underlying Qwen/model/dataset obligations. See
[`agilabs-ai/jev48`](https://github.com/agilabs-ai/jev48) and `THIRD_PARTY_NOTICES.md`.
"""
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
