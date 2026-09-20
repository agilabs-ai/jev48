# Run Jev48 now

> Historical operator note. Do not rerun paid training for release; see
> `RELEASE.md` for the current verification path.

## 1. Local sanity

```bash
python -m pip install -e '.[dev]'
pytest
python scripts/release_audit.py
```

## 2. Authenticate Modal

```bash
python -m pip install 'modal>=1.5,<2'
modal setup
```

## 3. Run the open-model experiment

```bash
modal run modal_app.py
```

No Jev/API key is required for this stage.

The command prints the `run_name` and the volume download command.

## 4. Only after selection is frozen, run Jev

Create one secret locally:

```bash
modal secret create jev48-secrets TYPESAFE_API_KEY=...
# OR
modal secret create jev48-secrets OPENROUTER_API_KEY=...
```

Then:

```bash
modal run modal_jev.py --run-name <RUN_NAME>
```

## 5. Download receipts

```bash
modal volume get jev48-artifacts <RUN_NAME> ./jev48-modal-results
```

Use `results/LAUNCH_FACTS.md` as the source of truth for public numerical claims.
