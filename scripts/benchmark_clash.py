#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, time, urllib.request
from pathlib import Path
import numpy as np, torch
from decider.infer import Decider
from jev48.benchmark_metrics import wilson_interval
from jev48.io import dump_json

REVISION="e1bbdc7d4f4ab16b30d7c11faca55ed9db50bd9a"
URL=f"https://raw.githubusercontent.com/AIPI-mvoronovych/JEVBenchmark-Contradiction-Detection/{REVISION}/data/synthetic_captions_test_filtered.json"
INSTRUCTION='You are given a description of an image (image_description) and a caption written for the same image (caption). Use both to answer the multiple choice question as accurately as possible. Choose the option best supported by both sources. If they contradict, or either lacks key information, select "Conflicting information – cannot answer."'
CONFLICT="Conflicting information – cannot answer."

def main():
 ap=argparse.ArgumentParser()
 for x in ("model","output","run-name","model-sha256","release-manifest-sha256","repo-commit","script-sha256"): ap.add_argument("--"+x,required=True)
 ap.add_argument("--device",default="cuda"); ap.add_argument("--limit",type=int,default=0); a=ap.parse_args()
 raw=urllib.request.urlopen(URL,timeout=120).read(); samples=json.loads(raw)["samples"]; samples=samples[:a.limit] if a.limit else samples
 model=Decider(a.model,device=a.device,dtype=torch.bfloat16,temperature=1.0,use_graphs=False); results=[]; lat=[]
 for i,s in enumerate(samples,1):
  answers=s["answers"]; values=[answers[k] for k in ("image_only","text_only","irrelevant_but_plausible")]+[CONFLICT]
  criteria={f"c{j}":v for j,v in enumerate(values)}
  state=json.dumps({"image_description":s["original_caption"],"caption":s["conflicting_caption"],"question":s["question"]},ensure_ascii=False)
  start=time.perf_counter(); ans=model.system_one(state,{"decision":{"type":"choice","instructions":INSTRUCTION,"criteria":criteria}},independent=True)["answers"]["decision"]; lat.append((time.perf_counter()-start)*1000)
  probs={k:float(ans["probabilities"][k]) for k in criteria}; pred=max(probs,key=probs.get); correct=pred=="c3"
  results.append({"sample_idx":i-1,"image_id":s["image_id"],"predicted":pred,"correct":correct,"probabilities":probs,"latency_ms":lat[-1]})
  if i==1 or i%100==0 or i==len(samples): print(f"clash {i}/{len(samples)}",flush=True)
 out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in results)); n=len(results); wins=sum(x["correct"] for x in results)
 summary={"benchmark":"CLASH text-vs-text contradiction detection","benchmark_revision":REVISION,"source_url":URL,"source_sha256":hashlib.sha256(raw).hexdigest(),"mode":"published Jev main condition: original COCO caption as image description, conflicting caption, native four-way choice","model":{"run_name":a.run_name,"sha256":a.model_sha256,"release_manifest_sha256":a.release_manifest_sha256,"repo_commit":a.repo_commit,"script_sha256":a.script_sha256},"metrics":{"n":n,"jev48_accuracy":wins/n,"jev48_accuracy_wilson_ci95":wilson_interval(wins,n),"jev_accuracy":.9858603568657874},"mean_latency_ms":sum(lat)/len(lat),"comparison":"same public rows and main protocol; aggregate Jev reference"}
 dump_json(out.with_suffix(".summary.json"),summary); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
