# ChatGPT build log

> Historical build record. `README.md`, `RESULTS.md`, `STATUS.md`, and
> `RELEASE.md` are authoritative.

This file records the project decisions made inside the chat before external GPU execution.

## Phase 1 — naive independent reproduction

The first approach treated Jev as a model primitive to reproduce independently: small Qwen backbone, dynamic decision head, synthetic/known-probability data, calibration and OOD tests.

That implementation was built and CPU-smoke-tested. It exposed the first real limitation: unseen-domain generalization, not GPU cost.

The git history retains this phase.

## Phase 2 — NanoJev pivot

A public-ecosystem search found NanoJev, which already implemented much of the 0.6B dedicated decision-head path. The project temporarily pivoted to “NanoJev + broader semantic/calibration data.”

This was later abandoned after a deeper ecosystem audit showed that it was not the strongest available starting point.

The git history retains this phase too.

## Phase 3 — ecosystem audit

After the user made virality the primary KPI, the experiment was reframed around the direct story people actually care about:

> Jev launched. Give ChatGPT a weekend, the public internet, and compute. How close can it get?

ChatGPT audited the relevant public projects at the code/methodology level, including:

- Mapika/decider
- jaredpalmer/kev
- TianyuCodings/NanoJev
- TheoLeeCJ/SemIf
- razorback16/openjev
- daseinlabs/open-jev

The audit selected `Mapika/decider-2b` as the primary base because it already covered broad semantic decisions, high-cardinality candidates, calibration tooling and the TypeSafe API shape.

Pinned starting point:

```text
Mapika/decider
b08acf787d5d1f718a8c36c4677960f43772c7be
Mapika/decider-2b
```

## Phase 4 — distinct contribution

A second audit found that `decider` already includes several preference datasets, so simply adding “real preference data” would be incremental.

The project therefore narrowed the actual modification to:

> learn from the empirical distribution of multiple human votes instead of collapsing judgments to one winner.

Implemented in chat:

- frozen MT-Bench question-level split before reading vote outcomes;
- aggregation of multiple expert votes into soft target distributions;
- model identities hidden from model input;
- deterministic response-position randomization;
- 17-task transfer suite from tasks explicitly marked held out in the pinned upstream registry;
- regression/replay suites;
- soft-target cross-entropy + categorical Brier fine-tuning;
- controlled hard-majority vs soft-vote ablation;
- dev-only model-selection gate;
- identical post-hoc temperature calibration;
- resumable live Jev scoring;
- paired bootstrap comparison;
- static result page generated from machine-readable receipts;
- Modal orchestration for the complete experiment.

## Current local verification

Active public-tree tests pass locally with no network/GPU access.

No final model-quality numbers are claimed yet because this chat environment has:

- no NVIDIA GPU;
- no outbound Hugging Face/model network access;
- no user Modal credentials;
- no TypeSafe API key.

Those are execution/authentication boundaries, not unimplemented research steps.

## Human vs ChatGPT roles

Human:

- chose the challenge and virality as the primary KPI;
- can authenticate external services and approve spend;
- can decide whether/when to publish.

ChatGPT:

- researched the ecosystem;
- selected the public starting point;
- designed the benchmark and integrity rules;
- wrote the data/training/evaluation/cloud code;
- will analyze external run receipts and iterate on dev-only evidence;
- will prepare the final release/marketing artifact from measured results.

## Claim discipline

Do not publish “from scratch.”

The defensible framing is:

> I gave ChatGPT 48 hours to recreate Jev using anything publicly available.

The repository explicitly discloses that ChatGPT found and reused public prior art.
