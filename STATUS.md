# OpenJev status

## Strategy

**Primary model path: NanoJev → OpenJev.**

The earlier independent model is retained only as an ablation/history artifact.

## Verified locally after the pivot

- package namespace is now `openjev`
- **14/14 tests passing**
- Python package/scripts compile cleanly
- simulator data validator passes: 2,500 rows, zero split/family leakage
- CPU evaluation smoke runs end-to-end
- exact dynamic-head CPU training smoke runs end-to-end
- NanoJev request/response adapter tested
- NanoJev training-schema bridge tested
- benchmark summaries include overall / seen / OOD / per-domain slices
- benchmark hash enforcement and launch-gate logic present
- default Modal runner now starts from NanoJev rather than the independent model

## First external experiment

```bash
modal run modal_app.py
```

It uses:

- public NanoJev checkpoint as initialization
- same NanoJev 0.6B architecture
- broader public semantic training data
- known-probability simulator data
- frozen semantic + fully held-out OOD benchmark
- Brier-distribution fine-tuning
- base NanoJev as the mandatory comparator

## External boundary

This chat environment does not contain the user's Modal authentication and therefore cannot launch the external GPU job itself.

No paid LLM API is needed for the first run.
