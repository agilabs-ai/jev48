# Modal runbook

The default runner uses one H100 for the model experiments. H100 is chosen for iteration speed/convenience, not because the 2B model fundamentally requires it.

## Setup

```bash
python -m pip install 'modal>=1.5,<2'
modal setup
```

## Core experiment

```bash
modal run modal_app.py
```

The output includes a run name. Artifacts persist to `jev48-artifacts`.

## Live Jev

Store the key in Modal, not source/chat:

```bash
modal secret create jev48-secrets TYPESAFE_API_KEY=YOUR_KEY
# or: modal secret create jev48-secrets OPENROUTER_API_KEY=YOUR_KEY
modal run modal_jev.py --run-name RUN_NAME
```

If interrupted, `benchmark_jev.py` resumes row by row from its existing receipt.

## Download

```bash
modal volume get jev48-artifacts RUN_NAME ./jev48-modal-results
```
