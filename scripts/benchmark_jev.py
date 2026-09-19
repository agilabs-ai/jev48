#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from openjev.io import dump_json, file_sha256, read_jsonl
from openjev.jev import JevClient
from openjev.metrics import summarize


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True)
    p.add_argument("--output",default="results/jev_predictions.jsonl")
    p.add_argument("--model",default="jev-latest")
    p.add_argument("--base-url",default="https://api.typesafe.ai")
    p.add_argument("--limit",type=int,default=0)
    args=p.parse_args()
    data=Path(args.data)
    examples=read_jsonl(data)
    if args.limit: examples=examples[:args.limit]
    client=JevClient(base_url=args.base_url,model=args.model)
    rows=[]
    for i,ex in enumerate(examples,1):
        row=client.score(ex)
        row["target_probs"]=ex.target_probs
        row["domain"]=ex.domain
        rows.append(row)
        print(f"{i}/{len(examples)} {ex.id} {row['latency_ms']:.1f}ms",flush=True)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf-8")
    metrics=summarize([r["probabilities"] for r in rows],[r["target_probs"] for r in rows])
    dump_json(out.with_suffix(".summary.json"),{
        "benchmark_sha256":file_sha256(data),"model":args.model,"metrics":metrics,
        "mean_latency_ms":sum(r["latency_ms"] for r in rows)/len(rows),
    })
    print(json.dumps(metrics,indent=2))


if __name__=="__main__": main()
