# Jev48 data sources

## Training / preference ablation

### MT-Bench human judgments

Pinned public source used to aggregate expert A/B/tie votes over real model responses. Question IDs are split before outcomes are inspected so all votes for one question stay in one partition.

The decision model sees anonymized response content; model identity remains metadata.

## Replay / regression

Public tasks from the pinned `Mapika/decider` data registry are used to reduce catastrophic forgetting during the short preference fine-tune.

## Transfer OOD

The frozen transfer suite is built only from tasks marked held-out/evaluation-only in the pinned upstream registry. Candidate order is randomized deterministically and outcome-blind.

## Jev

Jev outputs are evaluation-only. They never enter training, replay, candidate selection, or temperature fitting for Jev48.

At publication time preserve all upstream dataset/model licenses and citations. When redistribution is uncertain, publish builders, source identifiers, revisions, and hashes rather than republishing source text.
