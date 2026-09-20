#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from datasets import load_dataset
from decider.infer import Decider

from jev48.io import dump_json
from jev48.typed_decisions import DATASET, REVISION, answer_from_choice, normalize_question, score_predictions, state_text


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Jev48 on the pinned public typed-decisions test benchmark.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    dataset = load_dataset(DATASET, "all", split="test", revision=REVISION)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    model = Decider(args.model, device=args.device, dtype=torch.bfloat16, temperature=1.0, use_graphs=False)
    rows, case_latencies = [], []
    for i, source in enumerate(dataset, 1):
        questions = json.loads(source["questions"])
        gold = json.loads(source["gold"])
        state = state_text(source["state"])
        predictions = {}
        started = time.perf_counter()
        for qid, question in questions.items():
            ids, criteria = normalize_question(question)
            request = {"decision": {"type": "choice", "instructions": question["instructions"], "criteria": criteria}}
            answer = model.system_one(state, request, independent=True)["answers"]["decision"]
            predictions[qid] = answer_from_choice(question, {k: answer["probabilities"][k] for k in ids})
        elapsed_ms = (time.perf_counter() - started) * 1000
        case_latencies.append(elapsed_ms)
        rows.append({"id": source["id"], "workflow": source["workflow"], "questions": questions, "gold": gold, "predictions": predictions, "latency_ms": elapsed_ms})
        if i == 1 or i % 25 == 0 or i == len(dataset):
            print(f"{i}/{len(dataset)} {source['id']} {elapsed_ms:.1f}ms", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "benchmark": DATASET,
        "benchmark_revision": REVISION,
        "mode": "zero-shot general model; no typed-decisions training rows used",
        "model": args.model,
        "metrics": score_predictions(rows),
        "mean_case_latency_ms": sum(case_latencies) / len(case_latencies),
        "jev_reference": {"source": "published dataset card; aggregate only, not paired predictions", "model": "TypeSafe Jev 1.13.0", "accuracy": 0.727, "soft_accuracy": 0.580, "brier": 0.148, "kl": 1.442, "ece_15": 0.144, "score_mae": 0.391, "within_one_level": 0.952, "median_case_latency_ms": 710},
    }
    dump_json(output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
