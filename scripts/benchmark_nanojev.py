#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time

import httpx

from openjev.io import dump_json, file_sha256, read_jsonl
from openjev.metrics import summarize
from openjev.reporting import sliced_summary
from openjev.nanojev import build_nanojev_request, parse_nanojev_response


def run_cli(repo: Path, checkpoint: Path, request: dict, *, precision: str, batch_questions: int) -> tuple[dict, float]:
    with tempfile.TemporaryDirectory(prefix="oso-nanojev-") as td:
        td = Path(td)
        inp = td / "request.json"
        out = td / "response.json"
        inp.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
        cmd = [
            "python", str(repo / "scripts" / "predict_toy_decisions.py"),
            "--checkpoint-dir", str(checkpoint),
            "--input", str(inp),
            "--output", str(out),
            "--precision", precision,
            "--batch-questions", str(batch_questions),
        ]
        started = time.perf_counter()
        subprocess.run(cmd, cwd=repo, check=True)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return json.loads(out.read_text(encoding="utf-8")), elapsed_ms


def run_service(url: str, request: dict, *, timeout: float) -> tuple[dict, float]:
    endpoint = url.rstrip("/") + "/api/evaluate"
    with httpx.Client(timeout=timeout) as client:
        # Warm one tiny call so the measured request does not include cold startup if
        # the server itself is persistent.
        tiny = {"states": request["states"][:1]}
        warm = client.post(endpoint, json=tiny)
        warm.raise_for_status()
        started = time.perf_counter()
        response = client.post(endpoint, json=request)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.raise_for_status()
        return response.json(), elapsed_ms


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark the public NanoJev checkpoint on our frozen decision benchmark.")
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="results/nanojev_predictions.jsonl")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--service-url")
    mode.add_argument("--checkpoint-dir")
    p.add_argument("--nanojev-repo", default="vendor/NanoJev", help="Required with --checkpoint-dir")
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--batch-questions", type=int, default=0, help="0 = all benchmark questions in one NanoJev forward batch")
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--model-label", default="NanoJev")
    args = p.parse_args()

    data = Path(args.data)
    examples = read_jsonl(data)
    if args.limit:
        examples = examples[: args.limit]
    request = build_nanojev_request(examples)

    if args.service_url:
        raw, wall_ms = run_service(args.service_url, request, timeout=args.timeout)
        latency_scope = "persistent HTTP request after one warm-up; includes tokenization+transfer+GPU inference"
    else:
        repo = Path(args.nanojev_repo)
        checkpoint = Path(args.checkpoint_dir)
        if not (repo / "scripts" / "predict_toy_decisions.py").exists():
            raise FileNotFoundError(f"NanoJev repo not found at {repo}")
        raw, wall_ms = run_cli(repo, checkpoint, request, precision=args.precision, batch_questions=args.batch_questions)
        latency_scope = "CLI wall time INCLUDING checkpoint/model load; not comparable to persistent serving latency"

    probs = parse_nanojev_response(raw, examples)
    targets = [ex.target_probs for ex in examples]
    rows = []
    for ex, row in zip(examples, probs):
        rows.append({
            "id": ex.id,
            "domain": ex.domain,
            "candidate_ids": [c.id for c in ex.candidates],
            "probabilities": row,
            "target_probs": ex.target_probs,
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    execution = raw.get("execution", {}) if isinstance(raw, dict) else {}
    summary = {
        "model": args.model_label,
        "benchmark_sha256": file_sha256(data),
        "metrics": summarize(probs, targets),
        "slices": sliced_summary(probs, targets, examples),
        "batch_wall_ms": wall_ms,
        "questions": len(examples),
        "questions_per_second": (1000.0 * len(examples) / wall_ms) if wall_ms > 0 else None,
        "latency_scope": latency_scope,
        "nanojev_execution": execution,
        "notes": "Quality metrics are directly comparable on the same frozen rows. Latency is comparable only when scopes match.",
    }
    dump_json(out.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
