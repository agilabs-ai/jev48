# Jev48 GPU runbook

The preferred execution path is Modal:

```bash
python -m pip install 'modal>=1.5,<2'
modal setup
modal run modal_app.py
```

## Hardware

The current job requests one H100 for iteration speed and reduced operational friction. The research claim is not that an H100 is required.

The run manifest records elapsed H100-container time and a clearly labeled list-price GPU estimate. Do not present that estimate as the user's invoice.

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
