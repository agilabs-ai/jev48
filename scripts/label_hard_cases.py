#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from open_system_one.schema import DecisionExample
from open_system_one.teachers import AnthropicTeacher, OpenAITeacher, ensemble_judgments


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output",default="data/hard_cases_labeled.jsonl")
    p.add_argument("--openai-model",default=None)
    p.add_argument("--anthropic-model",default=None)
    p.add_argument("--samples-each",type=int,default=2)
    p.add_argument("--limit",type=int,default=0)
    p.add_argument("--service-tier",default=None)
    args=p.parse_args()
    teachers=[]
    if args.openai_model: teachers.append(OpenAITeacher(args.openai_model,service_tier=args.service_tier))
    if args.anthropic_model: teachers.append(AnthropicTeacher(args.anthropic_model))
    if not teachers: raise SystemExit("Pass --openai-model and/or --anthropic-model")
    rows=[DecisionExample.model_validate_json(x) for x in Path(args.input).read_text().splitlines() if x.strip()]
    if args.limit: rows=rows[:args.limit]
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",encoding="utf-8") as f:
        for i,ex in enumerate(rows,1):
            probs,audit=ensemble_judgments(ex,teachers,args.samples_each,seed=17+i)
            labeled=ex.model_copy(update={"target_probs":probs,"target_kind":"teacher_distribution","metadata":{**ex.metadata,"teacher_audit":audit}})
            f.write(labeled.model_dump_json()+"\n")
            print(f"{i}/{len(rows)} {ex.id}",flush=True)


if __name__=="__main__": main()
