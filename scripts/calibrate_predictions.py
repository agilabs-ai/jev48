#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jev48.io import dump_json, file_sha256, read_jsonl
from jev48.predictions import fit_temperature_from_probs, temperature_transform
from jev48.reporting import sliced_summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit one scalar temperature on calibration rows and score locked rows.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fit-split", default="calibration")
    ap.add_argument("--eval-splits", default="test,ood")
    args = ap.parse_args()

    examples = read_jsonl(args.data)
    by_id = {e.id: e for e in examples}
    rows = [json.loads(x) for x in Path(args.predictions).read_text(encoding="utf-8").splitlines() if x.strip()]
    if set(r["id"] for r in rows) != set(by_id):
        missing = set(by_id) - set(r["id"] for r in rows)
        extra = set(r["id"] for r in rows) - set(by_id)
        raise ValueError(f"prediction IDs do not match benchmark: missing={len(missing)} extra={len(extra)}")
    rows.sort(key=lambda r: examples.index(by_id[r["id"]]))

    fit_rows = [r for r in rows if by_id[r["id"]].split == args.fit_split]
    if not fit_rows:
        raise ValueError(f"no rows in fit split {args.fit_split}")
    T = fit_temperature_from_probs(
        [r["probabilities"] for r in fit_rows],
        [by_id[r["id"]].target_probs for r in fit_rows],
    )
    eval_splits = set(args.eval_splits.split(","))
    eval_rows = [r for r in rows if by_id[r["id"]].split in eval_splits]
    eval_examples = [by_id[r["id"]] for r in eval_rows]
    raw = [r["probabilities"] for r in eval_rows]
    calibrated = [temperature_transform(p, T) for p in raw]
    targets = [e.target_probs for e in eval_examples]

    out_rows = []
    for r, p in zip(rows, [temperature_transform(r["probabilities"], T) for r in rows]):
        out_rows.append({**r, "raw_probabilities": r["probabilities"], "probabilities": p, "fitted_temperature": T})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows), encoding="utf-8")
    summary = {
        "benchmark_sha256": file_sha256(args.data),
        "predictions_sha256": file_sha256(args.predictions),
        "fit_split": args.fit_split,
        "eval_splits": sorted(eval_splits),
        "temperature": T,
        "raw": sliced_summary(raw, targets, eval_examples),
        "calibrated": sliced_summary(calibrated, targets, eval_examples),
    }
    dump_json(out.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
