# Modal de-risk runbook

The first cloud run does **not** require OpenAI, Anthropic, or Jev API keys. It uses public datasets, public model weights, and one GPU.

## What the run does

`modal run modal_app.py` performs, in order:

1. build the public semantic benchmark
2. merge it with the simulator training corpus
3. run leakage validation
4. freeze/hash the benchmark
5. train the independent head-only Qwen3-0.6B model
6. benchmark it on the exact frozen rows
7. benchmark untuned Qwen option likelihood
8. clone NanoJev at the pinned source commit
9. download NanoJev's public checkpoint
10. benchmark NanoJev on exactly the same rows
11. apply the predeclared marketing gates
12. if head-only misses the quality gate, run the independent LoRA fallback and benchmark it
13. persist every result to a Modal volume

## One-time setup

Run locally on your machine:

```bash
python -m pip install 'modal>=1.1,<2'
modal setup
```

Do **not** paste Modal tokens into chat or commit them to the repo.

## Run

From this repository root:

```bash
modal run modal_app.py
```

The command prints the run name and comparison table.

## Download artifacts

The command prints the exact download command. It will look like:

```bash
modal volume get open-system-one-artifacts derisk-XXXXXXXXXX ./modal-results
```

## GPU choice

The default is an A10. This is deliberate: the first experiment is a 0.6B head-only / LoRA run and NanoJev inference, not a large full fine-tune.

Escalate to A100 only if memory or runtime evidence justifies it.

## Stop conditions

Do not spend more just because the first model loses.

- If head-only passes a NanoJev gate: stop and inspect.
- If head-only loses: run LoRA automatically.
- If LoRA still loses materially: use the prepared NanoJev++ bridge before considering a larger backbone.
- Only move to 1.7B+ after the data shows model capacity is the bottleneck.

## Plan B: NanoJev++ derivative

Only run this if the independent head/LoRA path is not compelling and you deliberately want to improve the existing OSS baseline:

```bash
modal run modal_app.py::nanojev_plus_gpu
```

This uses an A100 because NanoJev's public trainer performs a full-parameter fine-tune with FP32 parameter/optimizer storage. The branch is explicitly derivative and the launch copy must credit NanoJev.
