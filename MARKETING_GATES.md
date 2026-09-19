# Jev48 marketing gates

## Core story

> I gave ChatGPT a weekend to recreate Jev using anything publicly available.

## Required disclosure

- Not from scratch.
- Starting point: pinned `Mapika/decider-2b`.
- Jev48 is independent and not affiliated with TypeSafe.
- Existing public replicas/prior art were allowed.

## Model naming gate

The public checkpoint name **`jev48-2b`** is allowed only if a derivative candidate passes the predeclared selection gate.

If the untouched base wins, publish Jev48 as an experiment/benchmark and do not rebrand the upstream checkpoint.

## Performance claims

Only use metrics from the locked receipts / `LAUNCH_FACTS.md`.

Do not say “beats Jev” from one cherry-picked slice. Scope every superiority claim to the metric and benchmark where it is true.

A strong result can also be:

- “got within X points of Jev”;
- “the open public base was already surprisingly close”;
- “soft human-vote training improved Brier by Y without hurting transfer”;
- “Jev kept a large transfer advantage.”
