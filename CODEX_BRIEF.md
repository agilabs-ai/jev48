# Codex Handoff — Finish, Evaluate, and Publish OpenJev

## Mission

Take the current `OpenJev` repository from its present pre-compute state to a reproducible public v0.1 release.

OpenJev v0 is intentionally a **small derivative improvement to NanoJev**, not a new architecture. The core experiment is:

> Start from the public NanoJev 0.6B checkpoint, broaden its semantic + calibrated training distribution, and test whether the resulting OpenJev checkpoint improves general decision quality and calibration on frozen benchmarks without changing the core architecture.

Work autonomously. Do not stop to ask for preferences that can be resolved from the repository, upstream documentation, or measured results. Only ask the user when blocked by an authentication/permission step that genuinely requires them.

Do not paste or expose API keys/tokens in logs, commits, issues, model cards, or chat.

---

## Current project state

The repo already contains:

- OpenJev package + schemas
- public semantic dataset builder
- known-probability simulators
- NanoJev request/response adapter
- NanoJev training-schema bridge
- frozen benchmark hashing
- split/leakage validation
- Brier/NLL/ECE/accuracy metrics
- seen/OOD/per-domain slicing
- launch-gate logic
- Modal runner
- 14 locally passing tests at handoff
- a pinned NanoJev source revision and public checkpoint
- attribution / third-party notices
- an audit-friendly git history

The default external experiment is already intended to be:

```text
NanoJev public checkpoint
        ↓
semantic + known-probability training data
        ↓
short distribution-aware fine-tune
        ↓
OpenJev
        ↓
frozen evaluation vs base NanoJev
```

Do **not** restart the project from scratch unless the existing code is irreparably broken.

---

# Non-negotiable methodology

## 1. NanoJev is the mandatory baseline

Every headline comparison must use the exact public NanoJev checkpoint and the same benchmark rows, candidate order, inference precision, and evaluation code as OpenJev.

Keep the upstream source revision pinned unless a real compatibility bug forces a change. If changed, record the old/new revisions and reason.

## 2. Do not tune on final test/OOD data

Use only:

- `train` for optimization
- `dev` for checkpoint / hyperparameter selection
- `calibration` for temperature fitting

The final frozen `test`, `ood`, and external GPT-output preference benchmark must not be used to choose learning rate, step count, data mix, temperature, or checkpoint.

If multiple experiments are necessary, select the final candidate by dev metrics before opening final test/OOD results.

## 3. Benchmark must be frozen before the final run

Store:

- benchmark SHA-256
- dataset source versions / revisions
- deterministic seed
- candidate ordering policy
- row counts
- model checkpoint hashes

Never modify the final benchmark after seeing NanoJev/OpenJev results.

## 4. Calibration must be fair

Fit a scalar temperature **separately for NanoJev and OpenJev** using the exact same held-out calibration split.

Report both:

- raw metrics (`T=1`)
- calibrated metrics

Primary calibration claims and launch gates should use the calibrated outputs, because both models were given equal access to the calibration set.

Do not fit temperature on test/OOD/MT-Bench preference data.

## 5. Publish negative results honestly

If OpenJev does not beat NanoJev on the predeclared gate, do not move the benchmark or cherry-pick one domain and call the whole model better.

A valid release can still say:

> We broadened NanoJev to semantic decision tasks; here is where it improved and where it did not.

---

# P0 — Get the repository clean and runnable

1. Inspect the existing code and git status.
2. Run the full local test suite.
3. Run Python compilation / static import smoke tests.
4. Search for stale project names (`open-system-one`, `oso`, etc.) and clean user-facing leftovers without needlessly rewriting history.
5. Add a top-level `LICENSE` if missing. The OpenJev-authored code should use a permissive license compatible with the project; preserve NanoJev's MIT attribution and notices. Qwen3-0.6B is upstream Apache-2.0 and must be acknowledged in the model card.
6. Ensure `NOTICE.md` and `THIRD_PARTY_NOTICES.md` clearly disclose that OpenJev v0 starts from and fine-tunes NanoJev.
7. Pin enough dependencies to make the final Modal run reproducible. Prefer the dependency family recorded by NanoJev. Only change versions when a concrete Modal/runtime failure requires it, and document the change.
8. Make `make test` and/or one equivalent command pass from a fresh environment.

Acceptance criterion: a fresh clone can run local non-GPU tests without undocumented manual edits.

---

# P0 — Add a real-GPT-output external evaluation

Add a new **benchmark-only** evaluation built from `lmsys/mt_bench_human_judgments`.

This dataset contains expert pairwise human preferences over real outputs from GPT-4, GPT-3.5, Claude-v1, Vicuna, Alpaca, and LLaMA on MT-Bench prompts.

This benchmark must never enter OpenJev training.

## Build procedure

Create something like:

```text
scripts/build_mtbench_gpt_preference_eval.py
```

Use the `human` split.

Normalize model ordering, then group votes by at least:

```text
(question_id, normalized model pair, turn)
```

Prefer groups where **at least one response is from GPT-4 or GPT-3.5** so the benchmark genuinely contains real GPT outputs.

For each grouped item:

### State

Include:

- original user prompt / conversation context
- response A text
- response B text
- model identities only if the evaluation design explicitly wants model-aware judgment; otherwise hide model names to test content judgment

Default recommendation: **hide model names** from the decision state and retain them only in metadata.

### Question

```text
Which response would human expert evaluators prefer?
```

### Candidates

```text
A preferred
B preferred
Tie
```

### Target distribution

Aggregate all available human votes for the same grouped item into empirical probabilities:

```text
P(A), P(B), P(tie)
```

If only one vote exists, it may remain a one-hot observed outcome but must be tagged as single-vote rather than presented as a precise latent preference probability.

Record:

- number of votes
- model pair
- question id
- turn
- whether GPT-4 is involved
- whether GPT-3.5 is involved
- source license

Freeze this benchmark separately with its own hash.

## Report

For NanoJev and OpenJev report:

- 3-way accuracy
- multiclass Brier
- NLL
- ECE/reliability where sample size permits
- slices: GPT-4 involved / GPT-3.5 involved / turn 1 / turn 2
- multi-vote-only subset separately

This is an **external validation** benchmark. Do not use it for checkpoint selection.

---

# P0 — Modal execution

Use the user's Modal account and the existing `modal_app.py` as the starting point.

If Modal is not authenticated, stop only for the minimum authentication step (for example, asking the user to complete `modal setup` in their environment). Do not ask them to paste secrets.

## First task

Run the existing full experiment on Modal.

Expected top-level command:

```bash
modal run modal_app.py
```

Fix any actual runtime/build/data bugs until a clean run completes.

The run should:

1. build public semantic data;
2. build/merge simulator data;
3. validate leakage;
4. freeze/hash benchmark data;
5. clone the pinned NanoJev source;
6. download the public NanoJev checkpoint;
7. validate the training bridge with NanoJev's own validator;
8. obtain base NanoJev dev/calibration predictions;
9. fine-tune NanoJev into candidate OpenJev checkpoints;
10. select by dev only;
11. fit independent temperatures on calibration only;
12. run the final selected OpenJev + base NanoJev once on frozen test/OOD;
13. run the external MT-Bench GPT-output preference evaluation;
14. persist raw predictions, configs, hashes, temperatures, summaries, checkpoints, timing, and cost metadata.

## Compute

The existing A100 configuration is acceptable for the first clean run. Do not spend engineering time minimizing a few dollars before the pipeline works.

Once correctness is established, optionally verify whether the same training fits a cheaper GPU. Keep the final reproducibility receipt for whichever configuration is published.

---

# P0 — Improve the fine-tune only if measured dev results justify it

The intended v0 contribution is primarily **data + calibration**, not architecture surgery.

Do not alter the NanoJev architecture unless simpler fine-tuning approaches fail.

If the first fine-tune underperforms on **dev**, test a small controlled grid using only train/dev/calibration:

### Candidate knobs

1. lower backbone LR: e.g. `5e-6`, `1e-5`, `2e-5`;
2. training length: e.g. `50`, `100`, `150`, `300` steps;
3. data balance between Banking77 / BoolQ / simulator rows;
4. replay a modest fraction of NanoJev's original public training data to reduce catastrophic forgetting;
5. freeze more of the backbone or use a smaller adaptation if full fine-tuning is unstable;
6. compare CE, Brier, and a mixed objective only if upstream supports them cleanly.

### Selection rule

Choose the final checkpoint using a predeclared dev objective such as:

```text
dev Brier with an accuracy non-inferiority constraint
```

Do not repeatedly evaluate final test/OOD after every dev tweak.

Keep an experiment table with config, dev metrics, elapsed time, GPU, and checkpoint hash.

---

# P0 — Final evaluation matrix

At minimum evaluate:

| Model | Required? |
|---|---|
| public NanoJev checkpoint | yes |
| final OpenJev checkpoint | yes |
| untuned Qwen3-0.6B baseline | recommended |
| TypeSafe Jev | only if legitimate API access is available |

Do not delay the release solely because TypeSafe Jev access is unavailable.

## Core metrics

Report:

- accuracy
- multiclass Brier
- NLL
- ECE + reliability bins
- risk/coverage curve
- domain-balanced macro average

Report separately:

- overall
- seen semantic domains
- held-out OOD domains
- every individual domain
- MT-Bench GPT-output preference benchmark

## Statistical uncertainty

Add bootstrap confidence intervals for headline accuracy/Brier deltas where practical.

Use deterministic bootstrap seeds and publish the script.

## Invariance

Run candidate-permutation invariance checks. Report failures rather than suppressing them.

## Latency

Latency claims are allowed only under matched conditions.

Use persistent model loading for both NanoJev and OpenJev on the same hardware and precision.

Measure at least:

- p50
- p95
- throughput
- candidate counts approximately `2`, `8`, `32`, `77`

Clearly state whether tokenization/network are included.

Do not compare a cold CLI NanoJev call against a warm OpenJev service.

---

# P1 — Optional current Jev comparison

If TypeSafe provides legitimate Jev API access in the user's environment:

1. freeze all evaluation data before the first Jev call;
2. add a thin adapter;
3. query Jev exactly once for the final benchmark version where practical;
4. store raw API responses and version/date metadata;
5. never tune OpenJev after inspecting Jev test outputs.

Treat Jev results as an external comparator, not training labels.

---

# Public GPT-output datasets to consider

## Highest priority: MT-Bench human judgments

Dataset:

```text
lmsys/mt_bench_human_judgments
```

Why:

- actual GPT-4 / GPT-3.5 outputs;
- expert human pairwise preferences;
- small and clean;
- multiple votes can be aggregated into empirical distributions;
- excellent fit for Brier/calibration evaluation.

Use as benchmark only.

## Secondary: Chatbot Arena conversations

Dataset:

```text
lmsys/chatbot_arena_conversations
```

Why:

- ~33K real crowd-sourced pairwise conversations;
- human votes;
- includes GPT-4 and other strong model outputs;
- useful larger external preference set.

Caveat: access is gated/terms-accepted on Hugging Face. Do not make the main release depend on it.

## Secondary: WildChat

Dataset:

```text
allenai/WildChat-1M
```

Why:

- real human ↔ ChatGPT conversations;
- substantial GPT-4 and GPT-3.5 traffic;
- useful for real-world prompt/OOD distribution checks.

Caveat: there are no preference/correctness labels. Do not treat it as calibration ground truth.

## Secondary: LMSYS-Chat-1M

Dataset:

```text
lmsys/lmsys-chat-1m
```

Useful for real-world prompt distribution and model-output robustness, but not direct probability calibration.

## Small direct GPT answer corpus: MT-Bench model answers

The public MT-Bench files include `gpt-4.jsonl` and `gpt-3.5-turbo.jsonl`. These are useful for exact-response inspection and for reproducing the MT-Bench preference construction.

## Arena-Hard

Contains pre-generated answers for many popular models, including several GPT-4 variants, plus judgments. Useful as a larger secondary regression set, but distinguish automated judge labels from human preference labels.

## ShareGPT / ShareGPT-X

Can provide additional real ChatGPT conversation text. Use only after a license/privacy/data-quality review and do not rely on it for the primary benchmark.

### Important limitation

Public GPT conversation datasets generally provide **text outputs, not GPT's internal probability distributions/logits**.

Therefore:

- use them for semantic/OOD evaluation;
- use human votes as observed-outcome/preference supervision where available;
- do **not** claim they provide GPT confidence calibration targets.

If direct current-GPT probability/logprob comparison is desired, create a small separate API-generated benchmark with a frozen prompt set and recorded model/version/date. Do not use that data to train the final model if it is also used as the external comparison.

---

# P0 — Publication blockers to resolve

Before public release ensure:

1. top-level `LICENSE` exists;
2. NanoJev attribution + MIT notice are retained;
3. Qwen3 upstream model/license is credited;
4. dataset licenses are documented per source;
5. the repo does not accidentally commit credentials, Modal state, HF tokens, caches, or raw private artifacts;
6. no claim says OpenJev is affiliated with TypeSafe;
7. no claim says OpenJev independently invented or recreated NanoJev's architecture;
8. benchmark data redistribution is license-audited.

### Dataset publishing caution

The current evaluation mixes sources with different licenses. Do **not** blindly upload one merged dataset under MIT/Apache.

If redistribution terms are unclear, publish:

- dataset builder scripts
- exact upstream dataset identifiers/revisions
- deterministic seeds
- row IDs/source indices
- benchmark hashes
- transformation code

rather than republishing source text.

---

# P0 — GitHub publication

Publish the project repository from the existing git history.

Product/display name:

```text
OpenJev
```

If `openjev` is unavailable or confusing in the user's namespace, use a repo slug such as:

```text
openjev-chatgpt
```

while keeping the product name OpenJev.

README above the fold should contain:

1. one-sentence description;
2. explicit “derived from NanoJev” disclosure;
3. headline benchmark table;
4. install/run example;
5. link to model weights;
6. link to reproducibility/results;
7. limitations.

Create a release/tag such as:

```text
v0.1.0
```

Include exact final commit SHA in all published result metadata.

---

# P0 — Hugging Face publication

If Hugging Face credentials are available, publish a model repo containing the final checkpoint and enough files to run it.

Create a proper `MODEL_CARD.md` / Hugging Face README with:

- OpenJev description
- NanoJev derivation disclosure
- base Qwen model
- architecture unchanged vs NanoJev v0
- training datasets and licenses
- hardware
- training duration/cost
- exact NanoJev source revision/checkpoint
- metrics with confidence intervals
- raw vs calibrated results
- benchmark hashes
- intended use
- limitations
- known OOD failures
- citation/attribution

Do not claim Jev equivalence unless measured evidence supports it.

For datasets/benchmarks, publish only what license review permits. Otherwise publish builder scripts + manifests in GitHub and link them from the model card.

---

# Final release artifacts

The finished public project should contain at least:

```text
README.md
LICENSE
NOTICE.md
THIRD_PARTY_NOTICES.md
MODEL_CARD.md
DATASET_CARD.md or BENCHMARK_CARD.md
REPRODUCIBILITY.md
RESULTS.md
```

And machine-readable receipts:

```text
results/final_metrics.json
results/nanojev_predictions.jsonl
results/openjev_predictions.jsonl
results/mtbench_gpt_preferences_nanojev.jsonl
results/mtbench_gpt_preferences_openjev.jsonl
results/launch_gate.json
results/run_manifest.json
results/latency.json
results/bootstrap_ci.json
```

`run_manifest.json` should include:

- git SHA
- NanoJev SHA
- model hashes
- dataset hashes
- benchmark hashes
- random seeds
- dependency versions
- GPU
- Modal run ID
- temperatures
- exact training command
- elapsed time
- estimated compute cost if available

---

# Release claim policy

Only use claims supported by final frozen results.

## If the gate clearly passes

A valid headline is approximately:

> I gave ChatGPT NanoJev and told it to make it better. OpenJev improves the original on a frozen semantic decision benchmark while keeping the same 600M architecture.

Prefer concrete deltas immediately after the headline.

## If only the GPT-output external benchmark improves

Say exactly that:

> OpenJev improves NanoJev at predicting human preferences over real GPT outputs.

Do not generalize to “better overall.”

## If OpenJev improves calibration but slightly loses accuracy within the predeclared non-inferiority margin

Say:

> OpenJev is better calibrated at similar accuracy.

## If it loses

Publish the experiment honestly. Do not change the frozen benchmark to manufacture a win.

---

# Definition of done

Do not stop at “the code runs.” The project is done when all of the following are true:

- local tests pass from a clean environment;
- Modal training completes;
- final model checkpoint is persisted and downloadable;
- NanoJev and OpenJev have been evaluated on identical frozen rows;
- temperature calibration was fitted fairly;
- external MT-Bench real-GPT-output evaluation is complete;
- final test/OOD was not used for hyperparameter selection;
- raw predictions and machine-readable metrics are saved;
- launch gate is automatically computed;
- publication licenses/attributions are correct;
- GitHub repo is public;
- Hugging Face model is public if credentials permit;
- README/model card contain measured results, not placeholders;
- release URLs, commit SHA, model hash, benchmark hash, Modal run ID, and actual compute cost are returned to the user.

At the end, provide the user with a compact final report containing:

1. **Did OpenJev beat NanoJev?** — yes/no, with exact definition;
2. the final benchmark table;
3. GPT-output benchmark result;
4. model/repo URLs;
5. compute cost and runtime;
6. exact release claim that is defensible;
7. any remaining limitations or failed slices.
