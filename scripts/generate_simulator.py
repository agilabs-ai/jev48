#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from openjev.io import dump_json, write_jsonl
from openjev.simulators import generate_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/simulator")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--seen-per-domain", type=int, default=500)
    p.add_argument("--ood-count", type=int, default=500)
    args = p.parse_args()

    out = Path(args.output)
    examples = generate_dataset(args.seed, args.seen_per_domain, args.ood_count)
    counts = Counter()
    for split in ["train", "dev", "calibration", "test", "ood"]:
        rows = [x for x in examples if x.split == split]
        write_jsonl(out / f"{split}.jsonl", rows)
        counts[split] = len(rows)
    dump_json(out / "manifest.json", {
        "seed": args.seed,
        "seen_per_domain": args.seen_per_domain,
        "ood_count": args.ood_count,
        "counts": dict(counts),
        "note": "Known-probability simulator data. OOD is an entirely held-out equipment-failure domain.",
    })
    print(dict(counts))


if __name__ == "__main__":
    main()
