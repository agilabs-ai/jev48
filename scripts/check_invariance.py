#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import torch

from open_system_one.schema import DecisionRequest
from open_system_one.serving import load_engine


def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--data",required=True); p.add_argument("--limit",type=int,default=100); args=p.parse_args()
    from open_system_one.io import read_jsonl
    engine=load_engine(args.checkpoint)
    rows=read_jsonl(args.data)[:args.limit]
    rng=random.Random(17); max_delta=0.0
    for ex in rows:
        req=DecisionRequest(state=ex.state,question=ex.question,candidates=ex.candidates)
        base=engine.predict(req).probabilities
        order=list(range(len(ex.candidates))); rng.shuffle(order)
        req2=DecisionRequest(state=ex.state,question=ex.question,candidates=[ex.candidates[i] for i in order])
        perm=engine.predict(req2).probabilities
        delta=max(abs(base[k]-perm[k]) for k in base)
        max_delta=max(max_delta,delta)
    print({"examples":len(rows),"max_probability_delta":max_delta,"pass":max_delta < 1e-4})
    if max_delta >= 1e-4: raise SystemExit(1)


if __name__=="__main__": main()
