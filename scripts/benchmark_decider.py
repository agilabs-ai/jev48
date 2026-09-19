#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from jev48.io import dump_json, file_sha256, read_jsonl
from jev48.reporting import sliced_summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark a Mapika/decider-compatible model on Jev48 JSONL.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="Mapika/decider-2b")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--temperature", type=float, default=1.0, help="Use 1.0 for raw logits; calibrate afterward.")
    ap.add_argument("--graphs", action="store_true", help="Use upstream CUDA graph engine; quality runs default eager.")
    ap.add_argument("--splits", default="", help="Comma-separated subset, e.g. dev,calibration,test,ood")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from decider.infer import Decider

    data_path = Path(args.data)
    examples = read_jsonl(data_path)
    if args.splits:
        keep = set(args.splits.split(","))
        examples = [e for e in examples if e.split in keep]
    if args.limit:
        examples = examples[: args.limit]
    if not examples:
        raise ValueError("no benchmark examples selected")

    d = Decider(
        args.model,
        device=args.device,
        dtype=torch.bfloat16 if str(args.device).startswith("cuda") else torch.float32,
        temperature=args.temperature,
        use_graphs=args.graphs,
    )
    rows = []
    for i, ex in enumerate(examples, 1):
        questions = {
            "decision": {
                "type": "choice",
                "instructions": ex.question,
                "criteria": {c.id: c.text for c in ex.candidates},
            }
        }
        start = time.perf_counter()
        answer = d.system_one(ex.state, questions, independent=True)["answers"]["decision"]
        latency_ms = (time.perf_counter() - start) * 1000
        probs = [float(answer["probabilities"][c.id]) for c in ex.candidates]
        rows.append({
            "id": ex.id,
            "split": ex.split,
            "domain": ex.domain,
            "probabilities": probs,
            "target_probs": ex.target_probs,
            "latency_ms": latency_ms,
            "model": args.model,
            "temperature": args.temperature,
        })
        if i == 1 or i % 25 == 0 or i == len(examples):
            print(f"{i}/{len(examples)} {ex.id} {latency_ms:.1f}ms", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    probs = [r["probabilities"] for r in rows]
    targets = [r["target_probs"] for r in rows]
    summary = {
        "benchmark_sha256": file_sha256(data_path),
        "model": args.model,
        "temperature": args.temperature,
        "mean_latency_ms": sum(r["latency_ms"] for r in rows) / len(rows),
        "metrics": sliced_summary(probs, targets, examples),
    }
    dump_json(out.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
