# Run now

```bash
python -m pip install 'modal>=1.5,<2'
modal setup
modal run modal_app.py
```

For live Jev after the run:

```bash
modal secret create jev48-secrets TYPESAFE_API_KEY=YOUR_KEY
# or: modal secret create jev48-secrets OPENROUTER_API_KEY=YOUR_KEY
modal run modal_jev.py --run-name RUN_NAME
```

Then:

```bash
modal volume get jev48-artifacts RUN_NAME ./jev48-modal-results
```
