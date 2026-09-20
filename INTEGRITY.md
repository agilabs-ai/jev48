# Integrity protocol

The stunt is only interesting if it survives inspection.

## Challenge definition

The claim is **not** “ChatGPT independently invented Jev” or “from scratch.”

The experiment is:

> Give ChatGPT a fixed time window, public internet access, open-source code, and compute. Ask it to get as close as possible to Jev.

Finding and reusing strong public work is allowed and disclosed.

## Starting point

Primary public starting point:

```text
Mapika/decider
commit b08acf787d5d1f718a8c36c4677960f43772c7be
weights Mapika/decider-2b revision 4a0e86782adfdb7393e04b8ec9f6b939dca09273
```

The upstream architecture and released weights are not presented as Jev48 inventions.

## No Jev leakage

Jev is an external benchmark target only.

- No Jev response is inserted into training data.
- No Jev response is used to select a checkpoint or hyperparameter.
- The default live-Jev command does not query `train` or `dev` rows.
- The locked benchmark is hashed before the live run.

## Split discipline

### MT-Bench human judgments

Question IDs, not individual votes, determine splits. Split assignment uses only SHA-256(question ID + frozen seed), before vote outcomes are aggregated.

### Transfer

Tasks are explicitly held out by the pinned upstream training registry. Within each task, calibration/test selection uses hashes of state/question/options, never labels.

## Candidate selection

The fixed default candidates are:

```text
base decider-2b
hard-majority  lr=1e-6
soft-votes     lr=3e-7
soft-votes     lr=1e-6
soft-votes     lr=3e-6
```

A fine-tuned candidate must satisfy both on dev:

```text
preference Brier improves by >= 0.002
regression accuracy drops by <= 0.015
```

Among eligible candidates, lowest preference-dev Brier wins. If none pass, the base wins.

Locked test/OOD are not read by selection code.

## Calibration

After selection, the open starting point and Jev48 are independently temperature-scaled on the same `calibration` rows. Raw and calibrated receipts are retained.

For a **future paired live Jev run**, Jev's headline row would use its native provider
probabilities because calibrated probability output is part of Jev's product claim.
We could fit the same one-parameter temperature to Jev as a diagnostic, but it would
not replace the primary Jev row. No such live row exists in the current release.

This makes three questions separable:

1. what probability distribution did each system natively produce?
2. how good can the open reproduction become with one identical calibration primitive?
3. how do those calibrated open probabilities compare with Jev's native probabilities?

## Reporting

Primary metrics:

```text
accuracy / modal agreement
Brier score
NLL
ECE
risk-coverage
```

The final page is generated from machine-readable result JSON. Safe aggregate results,
benchmark hashes, provenance, selection receipts, and the paired bootstrap are
published. Mixed-source row-level inputs and predictions remain private pending a
per-source redistribution audit. Complete artifacts remain preserved on Modal.

The public Jev comparison is an unpaired, source-published aggregate. Its synthetic
gold labels are teacher samples, and the source does not provide its scorer code.
Only accuracy is used as a headline comparison; no paired significance claim is made.

## What would invalidate a headline

- editing locked rows after querying Jev;
- selecting a model on test/OOD;
- training on Jev outputs without explicitly changing the experiment;
- presenting upstream architecture/weights as original work;
- hiding a losing candidate/base comparison;
- comparing latency across unmatched hardware as if it were controlled;
- reporting only favorable domains.
