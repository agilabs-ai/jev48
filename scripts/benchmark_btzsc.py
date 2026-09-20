#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, random, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from decider.infer import Decider

from jev48.benchmark_metrics import wilson_interval
from jev48.io import dump_json

REVISION = "fef2a2ac62b69c58670047dddf045c53d7c3cb5e"
SOURCE_REVISION = "0d610cc53e79bcbec691312b0c4adb4a0e371642"
SPECS = (("agnews", "topic"), ("emotiondair", "emotion"), ("banking77", "intent"))
JEV = {"agnews": .91, "emotiondair": .48, "banking77": .87}

def balanced(targets, limit, seed):
    groups = defaultdict(list)
    for i, target in enumerate(targets): groups[target].append(i)
    rng = random.Random(seed)
    for values in groups.values(): rng.shuffle(values)
    out = []
    while len(out) < min(limit, len(targets)):
        changed = False
        for key in sorted(groups):
            if groups[key] and len(out) < limit:
                out.append(groups[key].pop()); changed = True
        if not changed: break
    return sorted(out)

def examples():
    out = []
    for offset, (name, task) in enumerate(SPECS):
        rows = load_dataset("btzsc/btzsc", name=name, split="test", revision=REVISION)
        binary, texts = list(map(int, rows["labels"])), list(map(str, rows["text"]))
        nclasses = next(i for i in range(1, len(texts)) if texts[i] != texts[0])
        labels = [str(rows[i]["hypothesis"]) for i in range(nclasses)]
        valid, targets = [], []
        for sample in range(len(rows) // nclasses):
            values = binary[sample*nclasses:(sample+1)*nclasses]
            if sum(values) == 1: valid.append(sample); targets.append(values.index(1))
        for pos in balanced(targets, 100, 20260917 + offset):
            sample = valid[pos]
            out.append({"dataset": name, "task": task, "id": f"{name}:{sample}", "text": texts[sample*nclasses], "labels": labels, "target": targets[pos]})
    return out

def main():
    ap=argparse.ArgumentParser()
    for x in ("model","output","run-name","model-sha256","release-manifest-sha256","repo-commit","script-sha256"): ap.add_argument("--"+x, required=True)
    ap.add_argument("--device",default="cuda"); ap.add_argument("--limit",type=int,default=0); a=ap.parse_args()
    rows=examples(); rows=rows[:a.limit] if a.limit else rows
    model=Decider(a.model,device=a.device,dtype=torch.bfloat16,temperature=1.0,use_graphs=False)
    results=[]; lat=[]
    for i,row in enumerate(rows,1):
        criteria={f"c{j:03d}": label for j,label in enumerate(row["labels"])}
        request={"decision":{"type":"choice","instructions":"Which single label best describes the input text?","criteria":criteria}}
        start=time.perf_counter(); ans=model.system_one(row["text"],request,independent=True)["answers"]["decision"]; lat.append((time.perf_counter()-start)*1000)
        probs=[float(ans["probabilities"][key]) for key in criteria]; predicted=int(np.argmax(probs))
        results.append({**row,"predicted":predicted,"correct":predicted==row["target"],"probabilities":probs,"latency_ms":lat[-1]})
        if i==1 or i%25==0 or i==len(rows): print(f"btzsc {i}/{len(rows)} {row['id']}",flush=True)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in results))
    by={}
    for name,_ in SPECS:
        part=[x for x in results if x["dataset"]==name]; wins=sum(x["correct"] for x in part); n=len(part)
        by[name]={"n":n,"jev48_accuracy":wins/n,"jev48_accuracy_wilson_ci95":wilson_interval(wins,n),"jev_accuracy":JEV[name]}
    wins=sum(x["correct"] for x in results); n=len(results)
    summary={"benchmark":"BTZSC pilot v1","benchmark_revision":SOURCE_REVISION,"dataset_revision":REVISION,"mode":"exact preregistered 300-row pilot manifest and native single-choice protocol","model":{"run_name":a.run_name,"sha256":a.model_sha256,"release_manifest_sha256":a.release_manifest_sha256,"repo_commit":a.repo_commit,"script_sha256":a.script_sha256},"metrics":{"n":n,"jev48_accuracy":wins/n,"jev48_accuracy_wilson_ci95":wilson_interval(wins,n),"jev_accuracy":sum(JEV.values())/3,"by_dataset":by},"mean_latency_ms":sum(lat)/len(lat),"comparison":"unpaired aggregate Jev reference; identical deterministic sampling protocol"}
    dump_json(out.with_suffix(".summary.json"),summary); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
