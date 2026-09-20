#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

import numpy as np
import torch
from decider.infer import Decider

from jev48.benchmark_metrics import paired_cluster_bootstrap_delta, wilson_interval
from jev48.io import dump_json


REVISION = "e105a48f8cdb7f3babb3594424f73e5d7bdc97b9"
BASE = f"https://raw.githubusercontent.com/fstandhartinger/jevbench/{REVISION}/"
FILES = ["datasets/public/easy.jsonl", "datasets/public/original.jsonl", "datasets/public/hard.jsonl"]
OUTCOMES = "results/v1.2/jevbench-v1.2-per-task.json"


def fetch(path: str) -> bytes:
    return urllib.request.urlopen(BASE + path, timeout=120).read()


def request_for(question: dict) -> tuple[list[str], dict]:
    kind = question["type"]
    criteria = question.get("criteria") or {}
    if kind == "noul":
        labels = ["no", "yes"]
        mapped = {"no": criteria.get("false", "No"), "yes": criteria.get("true", "Yes")}
    elif kind == "score":
        labels = [str(i) for i in range(len(criteria))]
        mapped = dict(zip(labels, criteria))
    else:
        labels = list(criteria)
        mapped = {k: (v or k) for k, v in criteria.items()}
    native = dict(question)
    return labels, {"decision": native}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    for name in ("run-name", "model-sha256", "release-manifest-sha256", "repo-commit", "script-sha256"):
        ap.add_argument(f"--{name}", required=True)
    args = ap.parse_args()
    rows, hashes = [], {}
    for path in FILES:
        raw = fetch(path)
        hashes[path] = hashlib.sha256(raw).hexdigest()
        tier = Path(path).stem
        rows.extend({**json.loads(line), "tier": tier} for line in raw.decode().splitlines() if line.strip())
    evidence_raw = fetch(OUTCOMES)
    hashes[OUTCOMES] = hashlib.sha256(evidence_raw).hexdigest()
    evidence = json.loads(evidence_raw)
    jev_outcomes = dict(evidence["systems"]["jev-1.13.0"]["public_tasks"])
    if len(rows) != 231 or set(x["id"] for x in rows) != set(jev_outcomes):
        raise ValueError("public task/evidence mismatch")
    if args.limit:
        rows = rows[:args.limit]
    model = Decider(args.model, device=args.device, dtype=torch.bfloat16, temperature=1.0, use_graphs=False)
    predictions, latencies = [], []
    for i, row in enumerate(rows, 1):
        labels, request = request_for(row["question"])
        state = row["state"] if isinstance(row["state"], str) else json.dumps(row["state"], ensure_ascii=False)
        started = time.perf_counter()
        answer = model.system_one(state, request, independent=True)["answers"]["decision"]
        latencies.append((time.perf_counter() - started) * 1000)
        if row["question"]["type"] == "noul":
            probs = {"no": 1.0 - float(answer["noul"]), "yes": float(answer["noul"])}
        else:
            probs = {k: float(answer["probabilities"][k]) for k in labels}
        predicted = labels[int(np.argmax([probs[k] for k in labels]))]
        correct = predicted == str(row["expected"])
        jev_code, jev_conf = jev_outcomes[row["id"]]
        predictions.append({"id": row["id"], "tier": row["tier"], "family": row["family"], "expected": row["expected"], "predicted": predicted, "correct": correct, "probabilities": probs, "latency_ms": latencies[-1], "jev_correct": jev_code == "c", "jev_outcome": jev_code, "jev_confidence": jev_conf})
        if i == 1 or i % 25 == 0 or i == len(rows):
            print(f"jevbench {i}/{len(rows)} {row['id']} {latencies[-1]:.1f}ms", flush=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in predictions), encoding="utf-8")
    by_tier = {}
    for tier in ("easy", "original", "hard"):
        subset = [x for x in predictions if x["tier"] == tier]
        n = len(subset); wins = sum(x["correct"] for x in subset); jwins = sum(x["jev_correct"] for x in subset)
        by_tier[tier] = {"n": n, "jev48_accuracy": wins / n, "jev48_accuracy_wilson_ci95": wilson_interval(wins, n), "jev_accuracy": jwins / n, "jev_accuracy_wilson_ci95": wilson_interval(jwins, n), "paired_delta": paired_cluster_bootstrap_delta((x["correct"] for x in subset), (x["jev_correct"] for x in subset), (x["family"] for x in subset))}
    n = len(predictions); wins = sum(x["correct"] for x in predictions); jwins = sum(x["jev_correct"] for x in predictions)
    summary = {"benchmark": "JevBench v1.2.2 public tasks", "benchmark_revision": REVISION, "file_sha256": hashes, "mode": "native JevBench question types on all 231 public tasks; no benchmark rows used in Jev48 fine-tuning or model selection; no favorable subset selection", "model": {"path": args.model, "sha256": args.model_sha256, "release_manifest_sha256": args.release_manifest_sha256, "run_name": args.run_name, "repo_commit": args.repo_commit, "script_sha256": args.script_sha256}, "metrics": {"n": n, "jev48_accuracy": wins / n, "jev48_accuracy_wilson_ci95": wilson_interval(wins, n), "jev_accuracy": jwins / n, "jev_accuracy_wilson_ci95": wilson_interval(jwins, n), "paired_delta": paired_cluster_bootstrap_delta((x["correct"] for x in predictions), (x["jev_correct"] for x in predictions), (x["family"] for x in predictions)), "by_tier": by_tier}, "mean_latency_ms": sum(latencies) / len(latencies), "comparison": "paired per-task correctness against source-published Jev outcomes using native question types; public subset only, not the 534-task leaderboard score"}
    dump_json(out.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
