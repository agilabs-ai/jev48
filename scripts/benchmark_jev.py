#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from jev48.io import dump_json, file_sha256, read_jsonl
from jev48.jev import JevClient
from jev48.reporting import sliced_summary


def main() -> None:
    p = argparse.ArgumentParser(description="Run the frozen Jev48 suite against live TypeSafe Jev.")
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="results/jev_predictions.jsonl")
    p.add_argument("--provider", choices=["auto", "typesafe", "openrouter"], default="auto")
    p.add_argument("--model", default="")
    p.add_argument("--endpoint", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--splits", default="calibration,test,ood", help="Never query dev/train by default.")
    p.add_argument("--sleep", type=float, default=0.0, help="Optional delay between successful API calls.")
    args = p.parse_args()

    data = Path(args.data)
    all_examples = read_jsonl(data)
    keep = set(args.splits.split(",")) if args.splits else None
    examples = [e for e in all_examples if keep is None or e.split in keep]
    if args.limit:
        examples = examples[: args.limit]
    if not examples:
        raise ValueError("no benchmark examples selected")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict] = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[row["id"]] = row
        unknown = set(existing) - {e.id for e in examples}
        if unknown:
            raise ValueError(f"resume file contains {len(unknown)} IDs outside selected benchmark")
        print(f"[resume] {len(existing)} existing rows", flush=True)

    client = JevClient(provider=args.provider, endpoint=args.endpoint or None, model=args.model or None)
    try:
        for i, ex in enumerate(examples, 1):
            if ex.id in existing:
                continue
            row = client.score(ex)
            row.update({"target_probs": ex.target_probs, "domain": ex.domain, "split": ex.split})
            existing[ex.id] = row
            # Rewrite the small receipt atomically-ish after every success so an interrupted run resumes.
            ordered = [existing[e.id] for e in examples if e.id in existing]
            tmp = out.with_suffix(out.suffix + ".tmp")
            tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered), encoding="utf-8")
            tmp.replace(out)
            print(f"{len(existing)}/{len(examples)} {ex.id} {row['latency_ms']:.1f}ms", flush=True)
            if args.sleep:
                time.sleep(args.sleep)
    finally:
        client.close()

    if len(existing) != len(examples):
        raise RuntimeError(f"incomplete Jev run: {len(existing)}/{len(examples)}")
    rows = [existing[e.id] for e in examples]
    summary = {
        "benchmark_sha256": file_sha256(data),
        "model": rows[0].get("model") if rows else args.model,
        "gateway": rows[0].get("gateway") if rows else args.provider,
        "queried_splits": sorted(keep) if keep else "all",
        "metrics": sliced_summary(
            [r["probabilities"] for r in rows],
            [r["target_probs"] for r in rows],
            examples,
        ),
        "mean_latency_ms": sum(r["latency_ms"] for r in rows) / len(rows),
        "jev_outputs_used_for_training": False,
    }
    dump_json(out.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
