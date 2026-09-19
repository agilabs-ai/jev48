#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import random

from jev48.io import read_jsonl
from jev48.metrics import accuracy, brier


def read_preds(path):
    return {r["id"]: r for r in (json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip())}


def quantile(xs, q):
    xs = sorted(xs)
    pos = (len(xs) - 1) * q
    lo = int(pos); hi = min(len(xs) - 1, lo + 1)
    f = pos - lo
    return xs[lo] * (1 - f) + xs[hi] * f


def split_report(examples, a, b, split, reps, seed):
    exs = [e for e in examples if e.split == split]
    groups = defaultdict(list)
    for e in exs:
        groups[e.family_id].append(e)
    keys = sorted(groups)
    if not keys:
        return None

    def stats(sample_keys):
        sampled = [e for k in sample_keys for e in groups[k]]
        ta = [a[e.id]["probabilities"] for e in sampled]
        tb = [b[e.id]["probabilities"] for e in sampled]
        ys = [e.target_probs for e in sampled]
        return accuracy(ta, ys) - accuracy(tb, ys), brier(ta, ys) - brier(tb, ys)

    point = stats(keys)
    rng = random.Random(seed)
    da, db = [], []
    for _ in range(reps):
        sample = [rng.choice(keys) for _ in keys]
        x, y = stats(sample); da.append(x); db.append(y)
    return {
        "n_rows": len(exs),
        "n_clusters": len(keys),
        "accuracy_difference_a_minus_b": {"point": point[0], "ci95": [quantile(da, .025), quantile(da, .975)]},
        "brier_difference_a_minus_b": {"point": point[1], "ci95": [quantile(db, .025), quantile(db, .975)]},
    }


def main():
    ap = argparse.ArgumentParser(description="Paired cluster bootstrap for two systems on identical Jev48 rows.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=48)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    examples = read_jsonl(args.data)
    a, b = read_preds(args.a), read_preds(args.b)
    needed = {e.id for e in examples if e.split in {"test", "ood"}}
    if not needed <= set(a) or not needed <= set(b):
        raise ValueError("prediction files do not cover all locked rows")
    result = {
        "a": args.name_a,
        "b": args.name_b,
        "interpretation": "positive accuracy difference favors A; negative Brier difference favors A",
        "reps": args.reps,
        "seed": args.seed,
        "by_split": {
            split: split_report(examples, a, b, split, args.reps, args.seed + i)
            for i, split in enumerate(("test", "ood"))
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
