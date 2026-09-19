#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from openjev.evaluation import predict_examples
from openjev.io import read_jsonl
from openjev.serving import load_engine


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--data",required=True)
    p.add_argument("--batch-questions",type=int,default=8)
    p.add_argument("--device",default=None)
    args=p.parse_args()
    engine=load_engine(args.checkpoint,args.device)
    examples=read_jsonl(args.data)
    metrics,_,_,_=predict_examples(
        engine.model,engine.tokenizer,examples,device=engine.device,max_length=engine.max_length,
        batch_questions=args.batch_questions,temperature=engine.temperature,
    )
    print(json.dumps(metrics,indent=2))


if __name__=="__main__":
    main()
