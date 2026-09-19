#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jev48.io import write_jsonl
from jev48.mtbench import MTBENCH_REVISION, aggregate_human_votes, manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Build frozen MT-Bench human-vote distributions.")
    ap.add_argument("--out", type=Path, default=Path("data/mtbench_human.jsonl"))
    ap.add_argument("--manifest", type=Path, default=Path("data/mtbench_human.manifest.json"))
    args = ap.parse_args()
    if args.out.exists() or args.manifest.exists():
        ap.error("output paths must not already exist; benchmark artifacts are immutable")
    from datasets import load_dataset

    ds = load_dataset(
        "lmsys/mt_bench_human_judgments",
        split="human",
        revision=MTBENCH_REVISION,
    )
    examples = aggregate_human_votes(list(ds))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, examples)
    args.manifest.write_text(json.dumps(manifest(examples), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.manifest.read_text(), end="")


if __name__ == "__main__":
    main()
