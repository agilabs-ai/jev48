# Jev48 GPU runbook

The preferred execution path is Modal:

```bash
python -m pip install 'modal>=1.5,<2'
modal setup
modal run modal_app.py
```

## Hardware

The current job requests one H200. An observed 80GB H100 run exhausted memory on the second training microbatch, while the unchanged configuration completed a candidate on H200. The research claim is not that an H200 is intrinsically required; lowering the token batch could reduce memory at the cost of changing the predefined execution configuration.

The run manifest records elapsed GPU-container time and a clearly labeled H200 list-price estimate. Do not present that estimate as the user's invoice.

## What is trained

Jev48 starts from the pinned public `Mapika/decider-2b` checkpoint and performs a small controlled preference fine-tune.

Candidate study:

```text
untouched decider-2b baseline
hard-majority MT-Bench vote labels
soft empirical MT-Bench vote distributions @ 3 learning rates
```

All candidates include the same replay/regression data. Selection uses dev only.

## Locked final stage

After `results/selection.json` exists:

- evaluate base and selected reproduction on frozen calibration/test/OOD;
- fit scalar temperature for open models on calibration only;
- preserve raw + calibrated predictions;
- if a derivative wins, stage it as `release/model` for publication as `jev48-2b`.

## Jev

Jev is deliberately run in a separate job after selection:

```bash
modal run modal_jev.py --run-name <RUN_NAME>
```

Jev's primary published row uses its native returned probabilities.
