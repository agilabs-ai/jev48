#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def entropy(p):
    return -sum(x*math.log(max(x,1e-12)) for x in p)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--predictions",required=True)
    p.add_argument("--source-data",required=True)
    p.add_argument("--output",default="data/hard_cases.jsonl")
    p.add_argument("--top",type=int,default=5000)
    args=p.parse_args()
    preds={r["id"]:r for r in (json.loads(x) for x in Path(args.predictions).read_text().splitlines() if x.strip())}
    source={r["id"]:r for r in (json.loads(x) for x in Path(args.source_data).read_text().splitlines() if x.strip())}
    scored=[]
    for ex_id,row in preds.items():
        pvec=row["probabilities"]
        target=row.get("target_probs") or source[ex_id].get("target_probs")
        margin=sorted(pvec,reverse=True)[0]-sorted(pvec,reverse=True)[1] if len(pvec)>1 else 1
        wrong=int(max(range(len(pvec)),key=pvec.__getitem__) != max(range(len(target)),key=target.__getitem__))
        score=2.0*wrong + entropy(pvec) + (1.0-margin)
        scored.append((score,ex_id))
    chosen=sorted(scored,reverse=True)[:args.top]
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",encoding="utf-8") as f:
        for score,ex_id in chosen:
            row=dict(source[ex_id]); row.setdefault("metadata",{})["hard_case_score"]=score
            f.write(json.dumps(row,ensure_ascii=False)+"\n")
    print(f"wrote {len(chosen)} hard cases")


if __name__=="__main__": main()
