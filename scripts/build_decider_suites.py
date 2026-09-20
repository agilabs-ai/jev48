#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from jev48.decider_bridge import (
    DECIDER_COMMIT,
    DECIDER_MODEL,
    DECIDER_MODEL_REVISION,
    DECIDER_REPO,
    build_regression_rows,
    build_replay_rows,
    build_transfer_rows,
)
from jev48.io import file_sha256, write_jsonl


FROZEN_DEFAULT_SHA256 = {
    "transfer": "1a16921655885d88ea082a2e2e65c5c628d730a8faed2150865bf852a2565be4",
    "regression": "2e22f6a1cdd5bc77f6a1c37ec1736206a80396326d6a8f79679a696c1364c719",
    "replay": "aca350d6583b9d2fbc0ff1fa8637418fef9403450c0f2f7347d230fabc44dd7e",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build pinned decider transfer, regression, and replay suites.")
    ap.add_argument("--out-dir", type=Path, default=Path("data/decider_suites"))
    ap.add_argument("--calibration-per-task", type=int, default=10)
    ap.add_argument("--test-per-task", type=int, default=40)
    ap.add_argument("--regression-per-task", type=int, default=75)
    ap.add_argument("--replay-per-task", type=int, default=500)
    args = ap.parse_args()
    if args.out_dir.exists():
        ap.error("out-dir must not already exist; benchmark/training manifests are immutable")

    import decider.data as D
    from jev48.decider_bridge import TRANSFER_TASKS
    not_heldout = [name for name in TRANSFER_TASKS if not D.TASKS.get(name, {}).get("heldout", False)]
    if not_heldout:
        raise ValueError(f"pinned transfer tasks are no longer marked heldout upstream: {not_heldout}")

    transfer = build_transfer_rows(D.load_task, args.calibration_per_task, args.test_per_task)
    regression = build_regression_rows(D.load_task, args.regression_per_task)
    replay = build_replay_rows(D.load_task, args.replay_per_task)
    args.out_dir.mkdir(parents=True)
    paths = {
        "transfer": args.out_dir / "transfer.jsonl",
        "regression": args.out_dir / "regression.jsonl",
        "replay": args.out_dir / "replay.jsonl",
    }
    write_jsonl(paths["transfer"], transfer)
    write_jsonl(paths["regression"], regression)
    write_jsonl(paths["replay"], replay)
    manifest = {
        "upstream": {"repo": DECIDER_REPO, "commit": DECIDER_COMMIT, "model": DECIDER_MODEL, "model_revision": DECIDER_MODEL_REVISION},
        "counts": {k: len(v) for k, v in {"transfer": transfer, "regression": regression, "replay": replay}.items()},
        "split_counts": {
            "transfer": dict(Counter(x.split for x in transfer)),
            "regression": dict(Counter(x.split for x in regression)),
            "replay": dict(Counter(x.split for x in replay)),
        },
        "data_quality": {
            "selected_duplicate_source_rows_collapsed": {
                k: sum(max(0, int(x.metadata.get("upstream_duplicate_count", 1)) - 1) for x in v)
                for k, v in {"transfer": transfer, "regression": regression, "replay": replay}.items()
            },
            "selected_conflicting_duplicate_rows_preserved": {
                k: sum(bool(x.metadata.get("upstream_conflicting_duplicate", False)) for x in v)
                for k, v in {"transfer": transfer, "regression": regression, "replay": replay}.items()
            },
        },
        "sha256": {k: file_sha256(p) for k, p in paths.items()},
        "source_revision_note": "The pinned Mapika registry does not disclose immutable revisions for every underlying dataset; default builds fail on frozen-output hash drift.",
    }
    if (args.calibration_per_task, args.test_per_task, args.regression_per_task, args.replay_per_task) == (10, 40, 75, 500):
        if manifest["sha256"] != FROZEN_DEFAULT_SHA256:
            raise RuntimeError(
                "underlying dataset output drifted from the audited frozen build; "
                f"expected {FROZEN_DEFAULT_SHA256}, got {manifest['sha256']}"
            )
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
