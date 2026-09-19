# OpenJev execution plan

## Primary path

There is one main experiment:

```text
NanoJev public checkpoint
        ↓
short semantic + calibration fine-tune
        ↓
OpenJev
        ↓
base-vs-OpenJev frozen benchmark
```

## Run A — local validation

```bash
make test
PYTHONPATH=. python scripts/validate_dataset.py data/simulator
```

Expected: tests pass and simulator splits show no family/content leakage.

## Run B — external OpenJev experiment

```bash
modal run modal_app.py
```

This is the first run that matters for the launch.

## Decision after Run B

### If OpenJev passes the predeclared gate

Stop changing the model and reproduce the result with a second seed before launch.

### If OpenJev improves semantic accuracy but hurts calibration

Try lower backbone LR and/or a separate calibration stage.

### If OpenJev improves seen tasks but collapses OOD

Adjust training mixture and replay data; do not alter the benchmark.

### If OpenJev does not improve over NanoJev

Inspect per-domain slices, catastrophic forgetting, and data balance before changing architecture.

## Ablations

The earlier independent model, LoRA path, option-likelihood baseline, and Jev adapter remain in the repo as secondary experiments. They are not the default OpenJev path.
