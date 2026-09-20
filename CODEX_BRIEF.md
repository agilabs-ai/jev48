# Codex Handoff — Finish and Publish Jev48

> Historical execution brief. `README.md` and `RELEASE.md` are authoritative.

## Mission

Finish **Jev48** end-to-end.

The public experiment is:

> **TypeSafe spent roughly two years building Jev. I gave ChatGPT one weekend and everything publicly available to see how close it could get.**

This is **not** a from-scratch challenge. Existing public code, weights, papers, datasets, and prior reproductions are allowed. ChatGPT already audited the public ecosystem and selected `Mapika/decider-2b` as the strongest practical starting point.

Your job is now execution, verification, one evidence-driven iteration if justified, and publication. Do not restart the project from scratch.

## Naming

- Project / experiment: **Jev48**
- Repository: **`jev48`**
- Public derivative model: **`jev48-2b`**
- Important: publish `jev48-2b` **only if a derivative checkpoint wins the predeclared dev-only selection gate**.
- If the untouched `Mapika/decider-2b` checkpoint wins, there is no new model release. Publish the Jev48 benchmark/results/code and state that the public base survived selection.
- Jev48 is independent and not affiliated with TypeSafe. Preserve that disclaimer wherever the name could imply otherwise.

## Starting point, pinned

```text
upstream repo:   Mapika/decider
upstream commit: b08acf787d5d1f718a8c36c4677960f43772c7be
starting model:  Mapika/decider-2b
```

Do not silently move those pins. If an upstream compatibility break forces a change, record the old/new revision and exact reason.

## What Jev48 actually changes

The upstream model already handles broad semantic decision tasks and hard-label preference data. Jev48 tests one narrower probability-quality hypothesis:

> **When multiple humans disagree, train on the empirical vote distribution instead of collapsing the votes into one hard winner.**

The controlled ablation compares, from the same starting checkpoint:

- hard-majority MT-Bench preference labels;
- soft empirical MT-Bench human-vote distributions;
- identical replay/regression data and evaluation.

Jev itself is never used for training or model selection.

---

# P0 — Do not change the methodology casually

These are non-negotiable unless a concrete bug makes execution impossible.

1. **Dev-only model selection.** `test` and `ood` stay locked until `results/selection.json` exists.
2. **Jev is queried only after open-model selection is frozen.**
3. **Jev outputs are never training data.**
4. **The public base and the selected Jev48 candidate see exactly the same rows/candidate order/evaluation code.**
5. **Temperature fitting for open models uses calibration only.**
6. **Jev's headline row uses native Jev probabilities.** A post-hoc Jev temperature may be reported only as a diagnostic.
7. **Benchmark hashes must match across final systems.**
8. **Negative/mixed results are publishable. Never move the benchmark to manufacture a win.**
9. **No “from scratch” wording.**
10. **Starting code/weights and all material prior art must be disclosed.**

Read `INTEGRITY.md` and `BENCHMARK_POLICY.md` before running external compute.

---

# P0 — Verify the package before spending compute

From repo root:

```bash
python -m pip install -e '.[dev]'
pytest
python scripts/release_audit.py
python -m compileall -q jev48 scripts tests
```

Expected handoff state: all tests pass and release audit is green.

Inspect `git status`. Do not discard the existing audit history.

Check the current code paths rather than relying on stale comments. The active pipeline is:

```text
modal_app.py
scripts/build_mtbench.py
scripts/build_decider_suites.py
scripts/assemble_jev48.py
scripts/train_decider_soft.py
scripts/select_candidate.py
scripts/benchmark_decider.py
scripts/calibrate_predictions.py
scripts/final_report.py
scripts/generate_launch_facts.py
modal_jev.py
```

---

# P0 — Run the core Modal experiment

The user must authenticate Modal locally. Never ask for pasted Modal secrets.

```bash
python -m pip install 'modal>=1.5,<2'
modal setup
modal run modal_app.py
```

The job is designed to run without a Jev/API key.

It should:

1. build/freeze the MT-Bench human-vote dataset;
2. build held-out transfer + replay/regression suites from the pinned `decider` registry;
3. hash the final benchmark;
4. evaluate untouched `Mapika/decider-2b` on dev;
5. train the predefined hard/soft candidate set;
6. select **using dev only**;
7. evaluate base + selected reproduction on locked `test`/`ood`;
8. fit calibration temperatures only on the calibration split;
9. generate `selection.json`, raw predictions, summaries, run manifest, release artifacts, static page, and `LAUNCH_FACTS.md`;
10. persist everything to Modal volume `jev48-artifacts`.

Do not open locked results while deciding hyperparameters.

## Candidate gate

Use the existing selection logic unless it is demonstrably broken. A derivative should not win because of microscopic noise or catastrophic regression.

If no derivative passes the gate, the untouched public base remains the selected reproduction. That is a valid result.

---

# P1 — Inspect the first run and decide whether one more dev-only iteration is justified

After the run, download the complete artifact directory. Inspect:

```text
results/selection.json
results/dev/*
results/final/*
results/run_manifest.json
results/LAUNCH_FACTS.md
release/MODEL_SOURCE.json
```

### You may run one additional training study if ALL are true

- the issue is visible on train/dev/calibration, not discovered from locked test/OOD;
- there is a simple, principled correction (e.g. learning rate, preference/replay weighting, obvious training bug);
- the benchmark remains unchanged;
- all attempted candidates remain in receipts;
- final candidate selection still happens before opening locked results for that new study.

Do **not** repeatedly tune against locked test/OOD or Jev.

If first run is technically sound, prefer shipping over endless optimization.

---

# P0 — Run the live Jev head-to-head

Only after selection is frozen.

The user may have either direct TypeSafe access or an OpenRouter route. Never request a secret in chat and never commit it.

Locally create one Modal secret:

```bash
modal secret create jev48-secrets TYPESAFE_API_KEY=...
# OR
modal secret create jev48-secrets OPENROUTER_API_KEY=...
```

Then:

```bash
modal run modal_jev.py --run-name <RUN_NAME_FROM_STAGE_1>
```

The Jev runner is resumable. If provider/API shape changed, fix the adapter while preserving the frozen benchmark and raw provider response receipts.

Final report must compare on identical benchmark hash:

- public starting point (`decider-2b`);
- selected Jev48 reproduction;
- native Jev probabilities.

Generate paired bootstrap comparisons already supported by the repo.

---

# P0 — Validate claims before publication

Run:

```bash
python scripts/release_audit.py --final-results <downloaded-run-root>
```

The machine-generated `results/LAUNCH_FACTS.md` is the source of truth for public numerical claims.

Do not manually improve numbers in README, website, social copy, screenshots, or model cards.

### Allowed story

> I gave ChatGPT a weekend to recreate Jev using anything publicly available.

### Not allowed

- “from scratch”
- “ChatGPT invented Jev's architecture”
- “first open Jev”
- “beat Jev” unless the locked metrics actually support the exact scope of that statement
- representing TypeSafe's two-year effort as equivalent to this derivative experiment without explaining the public-prior-art condition

If the selected reproduction is just the upstream base, say that clearly.

---

# P0 — Publication

## GitHub

Publish the complete Jev48 repository with:

- full audit-friendly git history;
- pinned upstream lineage;
- methodology/integrity docs;
- runnable Modal scripts;
- frozen benchmark builders/hashes;
- raw/aggregate receipts where licenses permit;
- static results page generated from receipts;
- clear non-affiliation disclaimer.

Do not rewrite history to hide earlier failed approaches. They are useful evidence of the agent changing strategy after research.

## Hugging Face

Only if a derivative checkpoint won selection:

Publish it as:

```text
jev48-2b
```

Use the generated model card as the starting point. It must state:

- derivative of `Mapika/decider-2b`;
- upstream commit and starting model;
- what Jev48 training changed;
- exact fitted temperature;
- benchmark hash;
- locked test/OOD metrics;
- licenses/attribution;
- that Jev outputs were never training data.

If the base won, **do not upload/rebrand the upstream weights**. Publish only the experiment/results repo.

## Release tag

Create a release/tag only after final receipts are complete, e.g.:

```text
v0.1.0
```

Record final git SHA, Modal run name/ID, benchmark SHA-256, model source, and result-page path.

---

# P1 — Final deliverable to the user

Return a compact release receipt containing:

1. GitHub URL
2. model URL if and only if `jev48-2b` exists
3. static results page URL/artifact
4. Modal run name/ID
5. total elapsed compute time
6. list-price GPU estimate and explicit caveat that it is not necessarily the user's invoice
7. benchmark hash
8. selected candidate + whether it was derivative or untouched base
9. Jev provider/path used
10. exact locked headline metrics with CIs where available
11. any limitations/failures
12. strongest defensible one-sentence launch claim

Do not finish with only “everything passed.” Finish with a published, reproducible artifact or state the exact external permission preventing publication.

---

# Success condition

The project is complete when an independent person can inspect the repo and answer:

- What public work did ChatGPT start from?
- What did it actually change?
- What data did it train on?
- What was frozen before Jev was queried?
- How was the candidate selected?
- How close did it get to Jev on the same rows?
- What did the weekend cost and how long did it take?
- Can I reproduce the comparison?

The viral story matters, but **integrity is part of the story**.
