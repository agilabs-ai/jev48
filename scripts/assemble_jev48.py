#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jev48.io import file_sha256, read_jsonl, write_jsonl


def unique(rows, label):
    ids = [r.id for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate IDs in {label}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble immutable Jev48 train/dev/final benchmark files.")
    ap.add_argument("--mtbench", required=True)
    ap.add_argument("--decider-dir", required=True)
    ap.add_argument("--out-dir", default="data/jev48_v1")
    args = ap.parse_args()
    out = Path(args.out_dir)
    if out.exists():
        ap.error("out-dir must not already exist")
    out.mkdir(parents=True)

    mt = read_jsonl(args.mtbench)
    ddir = Path(args.decider_dir)
    transfer = read_jsonl(ddir / "transfer.jsonl")
    regression = read_jsonl(ddir / "regression.jsonl")
    replay = read_jsonl(ddir / "replay.jsonl")

    mt_train = [x for x in mt if x.split == "train"]
    mt_dev = [x for x in mt if x.split == "dev"]
    mt_locked = [x for x in mt if x.split in {"calibration", "test"}]
    dev = unique(mt_dev + regression, "dev")
    final = unique(mt_locked + transfer, "final")

    files = {
        "mtbench_train": mt_train,
        "replay": replay,
        "dev": dev,
        "final": final,
    }
    paths = {}
    for name, rows in files.items():
        p = out / f"{name}.jsonl"
        write_jsonl(p, rows)
        paths[name] = p
    manifest = {
        "version": "jev48-v1",
        "policy": {
            "candidate_selection": "dev only",
            "temperature_fit": "calibration only after candidate selection",
            "locked_evaluation": ["test", "ood"],
            "jev_query_default": ["calibration", "test", "ood"],
            "jev_outputs_used_for_training": False,
        },
        "counts": {name: len(rows) for name, rows in files.items()},
        "sha256": {name: file_sha256(path) for name, path in paths.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
