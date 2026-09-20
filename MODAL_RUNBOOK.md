# Modal runbook — Jev48

## Core experiment

```bash
python -m pip install 'modal>=1.5,<2'
modal setup
modal run modal_app.py
```

The H200 job:

1. runs tests + release audit;
2. builds/fixes hashes for the MT-Bench human-vote and held-out transfer suites;
3. evaluates untouched `Mapika/decider-2b` on dev;
4. trains the predefined hard/soft candidates;
5. selects on dev only;
6. opens locked calibration/test/OOD only after selection;
7. fits calibration temperatures on calibration only;
8. writes all receipts and, if warranted, a `release/model` derivative.

Artifacts persist to volume `jev48-artifacts`.

## Live Jev stage

After core selection:

```bash
modal secret create jev48-secrets TYPESAFE_API_KEY=...
# or OPENROUTER_API_KEY=...
modal run modal_jev.py --run-name <RUN_NAME>
```

The final Jev headline row uses native provider probabilities. A separate Jev temperature fit is diagnostic only.
