#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import time

from open_system_one.schema import Candidate,DecisionRequest
from open_system_one.serving import load_engine


def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--repeats",type=int,default=10); args=p.parse_args()
    engine=load_engine(args.checkpoint)
    base_state="A customer reports that their production integration has stopped processing payments after a credential rotation. "
    for k in [2,4,8,16,32,64]:
        candidates=[Candidate(id=f"c{i}",text=f"Possible routing destination number {i}, responsible for category {i}.") for i in range(k)]
        req=DecisionRequest(state=base_state,question="Which destination best fits this case?",candidates=candidates)
        engine.predict(req)
        vals=[]
        for _ in range(args.repeats):
            t=time.perf_counter(); engine.predict(req); vals.append((time.perf_counter()-t)*1000)
        print({"candidates":k,"median_ms":statistics.median(vals),"p95_ms":sorted(vals)[max(0,int(.95*len(vals))-1)],"prefix_sharing":False})


if __name__=="__main__": main()
