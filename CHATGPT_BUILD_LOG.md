# ChatGPT Build Log

This file records what was actually built and executed inside the ChatGPT session before any external GPU or paid-model run.

## Starting brief

The first committed artifact is `PROJECT_SPEC.md`: a weekend plan for testing whether the useful core of a Jev-style System One model can be reproduced with a small open backbone, dynamic candidate scoring, calibrated probabilities, and a transparent benchmark.

## Public references inspected

The implementation direction was informed by the public NanoJev and open-jev projects. We deliberately did **not** copy either project wholesale:

- NanoJev establishes that a Qwen3-0.6B backbone plus dynamic decision heads is viable, but its public experiments are strongly game/navigation-oriented.
- open-jev establishes option-likelihood / prefix-cache scoring as an important no-training baseline.

Our v0 is benchmark-first and adds semantic tasks, known-probability simulators, leakage checks, calibration, risk-coverage metrics, hard-case mining, and explicit systems honesty about prefix reuse.

## Built in-chat

- decision schema with dynamic candidate sets
- dynamic scalar + set-attention decision head
- soft-target CE + categorical Brier objective
- temperature scaling
- Brier, NLL, ECE, TV, risk-coverage evaluation
- simulator-grounded probability datasets
- fully held-out OOD simulator domain
- dataset split/family/content leakage validator
- candidate permutation augmentation + invariance check
- semantic smoke benchmark + freeze manifest
- CPU feature baseline
- tiny local backbone using the exact dynamic-head training path
- Qwen3-0.6B head-only trainer
- option-likelihood baseline
- Jev API benchmark adapter
- frontier-teacher ensemble + hard-case pipeline
- FastAPI `/v1/decision` and `/v1/systemone` endpoints
- latency-vs-candidate-count benchmark
- unit tests and GPU runbook

## Executed in-chat

At packaging time:

- unit tests: 10/10 passing
- simulator examples generated: 2,500
- leakage validation: passing
- semantic smoke benchmark: 35 examples / 7 domains
- CPU baseline: completed
- tiny exact dynamic-head training: completed
- calibration fit: completed

The local smoke models deliberately fail on a completely unseen equipment-failure domain. This is a useful finding: the harness exposes OOD generalization failure instead of hiding it.

## Not executed in-chat

The chat container has no external model download access and no user API credentials. Therefore these require an external execution environment:

1. download Qwen3-0.6B weights;
2. run the first real head-only GPU training job;
3. query Jev after the benchmark is frozen;
4. optionally query premium teachers for mined hard cases.

No H100 is required for the first run. See `GPU_RUNBOOK.md`.

## Methodological correction made during build

For soft ground-truth target distributions, ECE and risk-coverage cannot pretend that target argmax is a binary realized outcome. The implementation therefore uses the target probability assigned to the model's selected class when computing expected correctness/error for known-distribution simulator data.

## Claim discipline

Until real Qwen/Jev results exist, this repository is a **runnable replication experiment**, not evidence that Jev has been matched.

## De-risk stage: NanoJev + public benchmark + Modal

Added after the initial runnable implementation:

- NanoJev as a mandatory first-class baseline
- pinned NanoJev source/checkpoint adapters
- independent-vs-derivative disclosure boundary
- Banking77 + BoolQ seen-domain public data builder
- DBpedia14 + AG News fully held-out OOD domains
- deterministic candidate permutation in the frozen benchmark
- overall / seen / OOD / per-domain metric slicing
- benchmark-hash enforcement
- predeclared PASS/FAIL launch gates vs NanoJev
- untuned Qwen batched option-likelihood baseline
- head-only Modal A10 experiment
- automatic independent LoRA fallback
- separately invoked NanoJev++ derivative A100 branch
- data bridge into NanoJev's public training schema
- Modal artifact persistence/download runbook

Local verification after this stage: **14/14 tests pass** and all Python modules compile.

A direct Modal execution is not possible inside the ChatGPT container because no user Modal credentials/profile are mounted and outbound package/model networking is disabled. No secrets were requested or embedded.

## Pivot: OpenJev

The project strategy was changed after recognizing that NanoJev already implements most of the difficult Jev-style decision machinery.

OpenJev v0 now:

- starts from the public NanoJev checkpoint;
- leaves the core 0.6B architecture unchanged;
- adds broader semantic decision data;
- adds known-probability simulator supervision;
- evaluates base NanoJev and OpenJev on the same frozen semantic/OOD benchmark;
- retains the earlier independent implementation only as an ablation/history artifact.

The default `modal run modal_app.py` now executes this derivative OpenJev path directly.
