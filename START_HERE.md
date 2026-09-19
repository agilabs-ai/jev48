# Start here

Jev48 is already implemented through the external-compute boundary.

## 1. Verify locally

```bash
python -m pip install -e '.[dev]'
pytest
```

## 2. Run the experiment on Modal

```bash
python -m pip install 'modal>=1.5,<2'
modal setup
modal run modal_app.py
```

Copy the printed `run_name`.

## 3. Run live Jev

Only after candidate selection is complete:

```bash
modal secret create jev48-secrets TYPESAFE_API_KEY=YOUR_KEY
# or: modal secret create jev48-secrets OPENROUTER_API_KEY=YOUR_KEY
modal run modal_jev.py --run-name YOUR_RUN_NAME
```

Do not expose the API key in chat, source files, git, logs, or artifacts.

## 4. Download receipts

```bash
modal volume get jev48-artifacts YOUR_RUN_NAME ./jev48-modal-results
```

The final comparison is `results/final_report.md`; raw row-level predictions live beside it.
