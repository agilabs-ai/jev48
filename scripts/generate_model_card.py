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

| Model | Accuracy ↑ |
|---|---:|
| TypeSafe Jev 1.13.0 (published) | {f(jev['accuracy'])} |
| jev48-2b (zero-shot) | {f(ours['accuracy'])} |

Benchmark revision: `{public['benchmark_revision']}`. Jev leads accuracy by
{(jev['accuracy'] - ours['accuracy']) * 100:.1f} percentage points. The target is the
mean of three samples from a separate teacher model, so this measures agreement with
that synthetic teacher—not real-world correctness. Jev's row is unpaired and its
scorer implementation is unavailable; no significance or distribution-metric parity
is claimed.
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
- Starting model revision: `{source['base_model_revision']}`
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

## Install and inference

Tested with Python 3.12 on an NVIDIA L40S using bfloat16:

```bash
git clone https://github.com/Mapika/decider.git
cd decider && git checkout {source['upstream_commit']}
python -m pip install '.[train]'
```

```python
import torch
from decider.infer import Decider

model = Decider("agilabs-ai/jev48-2b", device="cuda", dtype=torch.bfloat16,
                temperature=None, use_graphs=False)
questions = {{"choice": {{"type": "choice", "instructions": "Choose one.",
             "criteria": {{"a": "Option A", "b": "Option B"}}}}}}
result = model.system_one("Relevant state goes here.", questions, independent=True)
print(result["answers"]["choice"]["probabilities"])
```

Input is a state string plus typed questions and candidate criteria. Output is a
probability mapping over candidate IDs. CPU inference can use `device="cpu"` and
`torch.float32`. The [published smoke receipt](https://github.com/agilabs-ai/jev48/blob/main/receipts/SMOKE_TEST.json)
records the clean-environment test.

## Limitations

The preference result is small-sample; paired bootstrap uncertainty is published in
the receipts. Only the winning soft-label trial was evaluated on locked rows, so the
locked result is not a hard-vs-soft causal ablation. Calibration uses one pooled scalar
temperature across heterogeneous tasks. Do not use this model as a factual oracle or
for consequential decisions without workload-specific validation.

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
