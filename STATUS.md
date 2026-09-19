# Status

## Current state

**Local implementation: ready for first real GPU benchmark.**

### Verified locally

- 14 unit tests passing
- Python package/scripts compile cleanly
- simulator data: 2,500 rows with no cross-split family/content leakage
- CPU baseline and exact dynamic-head smoke training run end-to-end
- soft-target calibration metrics corrected for probabilistic ground truth
- NanoJev request/response adapter tested
- NanoJev training-data bridge tested
- benchmark summaries include overall / seen / OOD / per-domain slices
- launch-gate script refuses benchmark-hash mismatches

### First real external experiment

`modal run modal_app.py`

It uses:

- Qwen3-0.6B independent head-only model
- automatic LoRA fallback if needed
- public Banking77 + BoolQ seen domains
- fully held-out DBpedia14 + AG News OOD domains
- simulator probability data in training
- public NanoJev checkpoint as a mandatory baseline
- untuned Qwen option-likelihood baseline
- deterministic candidate permutation in the frozen benchmark

### External boundary

This chat container has no outbound package/model network and no Modal credentials. A direct `pip install modal` attempt fails on DNS, and no local Modal profile is mounted here.

The first thing requiring the user's environment is therefore authentication/execution on Modal. No API keys are needed for round 1.

### Plan B

If both independent head-only and LoRA miss the predeclared NanoJev gate, a separate `nanojev_plus_gpu` Modal function is prepared. It explicitly initializes from the public NanoJev checkpoint and is labeled as a derivative experiment.
