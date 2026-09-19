# Modal runbook — OpenJev

OpenJev v0 uses the public NanoJev checkpoint as its starting point. The default cloud job does not train a separate architecture first.

## Authenticate once

```bash
python -m pip install 'modal>=1.1,<2'
modal setup
```

## Run

```bash
modal run modal_app.py
```

## What the run does

```text
public semantic data + simulator data
                │
                ▼
       validate + freeze benchmark
                │
                ▼
      download base NanoJev
          │             │
          │             └── benchmark base NanoJev
          │
          ▼
  short Brier fine-tune
          │
          ▼
        OpenJev
          │
          └── benchmark same frozen rows
                │
                ▼
          launch gate PASS/FAIL
```

The default GPU is an **A100**, chosen for friction reduction rather than necessity. This is a 0.6B model and does not require an H100.

## Outputs

Artifacts are stored in the Modal volume:

```text
openjev-artifacts
```

The run prints the exact download command. Important files:

```text
results/nanojev_open_decision.summary.json
results/openjev_open_decision.summary.json
results/openjev_vs_nanojev.md
results/openjev_gate.json
checkpoints/openjev/
```

## Cost control

Round 1 uses public labels and requires no paid LLM API.

Do not add frontier-teacher labels until the frozen comparison tells us they are needed.
