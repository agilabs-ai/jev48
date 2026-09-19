#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from open_system_one.evaluation import predict_examples
from open_system_one.io import dump_json, file_sha256, read_jsonl
from open_system_one.reporting import sliced_summary
from open_system_one.serving import load_engine


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark an Open System One checkpoint on frozen decision rows.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="results/checkpoint_predictions.jsonl")
    p.add_argument("--batch-questions", type=int, default=1)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    data = Path(args.data)
    examples = read_jsonl(data)
    if args.limit:
        examples = examples[: args.limit]
    engine = load_engine(args.checkpoint, args.device)

    started = time.perf_counter()
    _metrics, _logits, targets, records = predict_examples(
        engine.model,
        engine.tokenizer,
        examples,
        device=engine.device,
        max_length=engine.max_length,
        batch_questions=args.batch_questions,
        temperature=engine.temperature,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    probs = [r["probabilities"] for r in records]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    summary = {
        "model": f"Open System One ({Path(args.checkpoint).name})",
        "benchmark_sha256": file_sha256(data),
        "metrics": sliced_summary(probs, targets, examples)["overall"],
        "slices": sliced_summary(probs, targets, examples),
        "batch_wall_ms": elapsed_ms,
        "questions": len(examples),
        "questions_per_second": (1000.0 * len(examples) / elapsed_ms) if elapsed_ms > 0 else None,
        "latency_scope": "persistent in-process model after load; includes tokenization+tensor assembly+GPU inference",
        "checkpoint": str(args.checkpoint),
        "temperature": engine.temperature,
    }
    dump_json(out.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
