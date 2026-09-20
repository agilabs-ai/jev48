---
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

- Starting model: `Mapika/decider-2b`
- Starting model revision: `4a0e86782adfdb7393e04b8ec9f6b939dca09273`
- Upstream repository: `Mapika/decider`
- Upstream commit: `b08acf787d5d1f718a8c36c4677960f43772c7be`
- Selected trial: `soft-lr3e-6`
- Benchmark SHA-256: `61a429af4502688f44d3d80a211f39282b6ff7ae9b78aac29fe1e7c029fe65a8`
- Fitted temperature: `1.041604557693133`

## Locked evaluation

| Slice | Accuracy ↑ | Brier ↓ | ECE ↓ |
|---|---:|---:|---:|
| MT-Bench expert-vote test | 0.6207 | 0.4111 | 0.0456 |
| Held-out transfer OOD | 0.7926 | 0.3039 | 0.0452 |

Temperature was fitted only on the frozen calibration split after model selection.
Locked test/OOD were not used for selection. Jev outputs were not used for training.

## Independent public Jev benchmark

This is an aggregate, unpaired comparison on `LocalLLaMA/typed-decisions`, not a
live Jev run on Jev48's original rows. The benchmark maintainers published the
Jev row; Jev48 ran separately, zero-shot, on the pinned test split.

| Model | Accuracy ↑ |
|---|---:|
| TypeSafe Jev 1.13.0 (published) | 0.7270 |
| jev48-2b (zero-shot) | 0.5770 |

Benchmark revision: `ea9306458d6e9563628369a3d1e72e362fb381d2`. Jev leads accuracy by
15.0 percentage points. The target is the
mean of three samples from a separate teacher model, so this measures agreement with
that synthetic teacher—not real-world correctness. Jev's row is unpaired and its
scorer implementation is unavailable; no significance or distribution-metric parity
is claimed.

## Intended use

Typed, generation-free decision scoring through the upstream `decider` interface.
Validate calibration on your own outcome-labelled workload before consequential use.

## Install and inference

Tested with Python 3.12 on an NVIDIA L40S using bfloat16:

```bash
git clone https://github.com/Mapika/decider.git
cd decider && git checkout b08acf787d5d1f718a8c36c4677960f43772c7be
python -m pip install '.[train]'
```

Download the public, hash-verified checkpoint from the GitHub release. The weights
are split only to satisfy the release-asset size limit:

```bash
mkdir jev48-2b && cd jev48-2b
gh release download v1.0.0 --repo agilabs-ai/jev48 \
  --pattern 'model.safetensors.part-*' --pattern 'jev48-2b-config.tar.gz' \
  --pattern SHA256SUMS
shasum -a 256 -c SHA256SUMS
cat model.safetensors.part-aa model.safetensors.part-ab > model.safetensors
printf '20948bb0163f7230d3922e292e919a62cf6c0e0c7600718ab0d152b693227aec  model.safetensors\n' | shasum -a 256 -c -
tar -xzf jev48-2b-config.tar.gz
```

```python
import torch
from decider.infer import Decider

model = Decider("/path/to/jev48-2b", device="cuda", dtype=torch.bfloat16,
                temperature=None, use_graphs=False)
questions = {"choice": {"type": "choice", "instructions": "Choose one.",
             "criteria": {"a": "Option A", "b": "Option B"}}}
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
