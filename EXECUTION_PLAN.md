# Execution plan

## Stage 0 — complete

- audited public Jev-like implementations;
- selected/pinned `Mapika/decider`;
- implemented human-vote aggregation;
- implemented pinned transfer/replay/regression builders;
- implemented hard-vs-soft training ablation;
- implemented dev-only selection gates;
- implemented post-selection calibration;
- implemented resumable live Jev scoring;
- implemented paired bootstrap + static results page.

## Stage 1 — Modal

```bash
modal run modal_app.py
```

This builds data, trains candidates, selects on dev, and evaluates base/selected model on locked rows.

## Stage 2 — Jev comparison

Preferred when credentials are available: live Jev on the frozen Jev48 rows.

```bash
modal secret create jev48-secrets TYPESAFE_API_KEY=...
# or OPENROUTER_API_KEY=...
modal run modal_jev.py --run-name <stage-1-run>
```

Credential-free fallback: run the selected checkpoint on the pinned independent
`LocalLLaMA/typed-decisions` test split, whose maintainers publish a measured Jev
1.13.0 aggregate row. This is a secondary, unpaired comparison and is labeled as such.

```bash
modal run modal_public_benchmark.py --run-name <stage-1-run>
```

## Stage 3 — release audit

- check data/model licenses;
- verify hashes and row counts;
- run paired bootstrap;
- generate static results page from JSON;
- publish repo/model only after headline text is checked against receipts.
