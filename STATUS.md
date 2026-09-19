# Current Status

## Executed in this build environment

- 10 unit tests: **passing**
- simulator corpus generated: **2,500 examples**
  - train: 1,300
  - dev: 200
  - calibration: 200
  - test: 300
  - fully held-out OOD domain: 500
- leakage validator: **passing**
  - unique IDs: 2,500
  - unique families: 2,500
  - no exact cross-split content duplication
- semantic smoke benchmark: **35 hand-authored examples / 7 domains**
- benchmark freeze manifest generated
- CPU TF-IDF/Ridge pipeline baseline executed
- exact `DynamicDecisionModel` training path executed with a tiny local backbone
- temperature scaling fitted successfully

## Current smoke findings

These are pipeline checks, **not Jev comparison results**.

The linear CPU baseline reaches 76% argmax agreement on the seen-domain simulator test set but collapses to its uniform behavior on the entirely unseen equipment domain.

The tiny local dynamic-head model likewise reaches 76% seen-domain argmax agreement and fails on the unseen domain. This is expected and useful: it shows the harness exposes OOD generalization failure rather than hiding it.

See `results/cpu_smoke.json` and `results/tiny_cpu_train.json`.

## Not yet executed here

### Real Qwen run

Blocked by this chat container having no internet/model cache and no `transformers` installation. The code and commands are ready.

### Jev benchmark

Blocked until a `TYPESAFE_API_KEY` is available in the execution environment.

### Frontier teacher labeling

Blocked until provider API credentials are available in the execution environment. The OpenAI/Anthropic ensemble code is ready.

### LoRA/full fine-tuning

Intentionally not implemented as the first move. Head-only is the decision gate.

## Immediate next external action

Run:

```bash
pip install -e '.[train,dev]'
make data
PYTHONPATH=. python scripts/train.py --config configs/train_head_fast.yaml
```

A single 24 GB NVIDIA GPU should be a comfortable target for this first experiment; an H100 is not required.
