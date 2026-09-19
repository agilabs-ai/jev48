# Data sources

## Human preference

`lmsys/mt_bench_human_judgments`

Pinned revision:

```text
ee34b9d273a7a35e4415c87678526c56c471098c
```

Used because it contains multiple expert judgments over real model conversations. Model identities are stored as provenance but removed from model input.

## Transfer, regression, replay

Loaded via the task registry of pinned `Mapika/decider` commit:

```text
b08acf787d5d1f718a8c36c4677960f43772c7be
```

Exact task lists are frozen in `jev48/decider_bridge.py`.

Transfer tasks are explicitly marked held-out/evaluation-only upstream. Regression/replay tasks are separate in-task eval/train partitions used only to guard against catastrophic fine-tuning regression.
