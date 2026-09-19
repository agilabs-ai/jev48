#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import torch

from open_system_one.baselines import BatchedOptionLikelihoodScorer, NaiveOptionLikelihoodScorer
from open_system_one.formatting import candidate_path
from open_system_one.io import dump_json, file_sha256, read_jsonl
from open_system_one.metrics import summarize
from open_system_one.reporting import sliced_summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--data",required=True)
    p.add_argument("--output",default="results/option_likelihood.jsonl")
    p.add_argument("--device",default="auto")
    p.add_argument("--norm",choices=["sum","mean"],default="mean")
    p.add_argument("--limit",type=int,default=0)
    p.add_argument("--implementation",choices=["batched","naive"],default="batched")
    args=p.parse_args()
    try:
        from transformers import AutoModelForCausalLM,AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("pip install -e '.[train]'") from exc
    device=args.device
    if device=="auto": device="cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available() else "cpu")
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(args.model,trust_remote_code=False).to(device)
    scorer=(BatchedOptionLikelihoodScorer if args.implementation=="batched" else NaiveOptionLikelihoodScorer)(model,tok,device)
    examples=read_jsonl(args.data)
    if args.limit: examples=examples[:args.limit]
    rows=[]; probs=[]; targets=[]; times=[]
    for i,ex in enumerate(examples,1):
        # Shared semantic prefix; only the final candidate text is treated as the option continuation.
        context=f"<STATE>\n{ex.state}\n</STATE>\n<QUESTION>\n{ex.question}\n</QUESTION>\n<CANDIDATE>\n"
        options=[c.text+"\n</CANDIDATE>" for c in ex.candidates]
        t=time.perf_counter(); result=scorer.score(context,options,args.norm); ms=(time.perf_counter()-t)*1000
        rows.append({"id":ex.id,"domain":ex.domain,"probabilities":result.probabilities,"scores":result.scores,"target_probs":ex.target_probs,"latency_ms":ms})
        probs.append(result.probabilities); targets.append(ex.target_probs); times.append(ms)
        print(f"{i}/{len(examples)} {ms:.1f}ms",flush=True)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text("".join(json.dumps(r)+"\n" for r in rows))
    summary={"model":f"{args.model} option likelihood ({args.implementation})","norm":args.norm,"implementation":args.implementation,"benchmark_sha256":file_sha256(Path(args.data)),"metrics":summarize(probs,targets),"slices":sliced_summary(probs,targets,examples),"mean_latency_ms":sum(times)/len(times),"questions_per_second":1000.0/(sum(times)/len(times)),"prefix_sharing":False}
    dump_json(out.with_suffix(".summary.json"),summary); print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
