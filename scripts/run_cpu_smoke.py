#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from open_system_one.formatting import paths_for_example
from open_system_one.io import dump_json, read_jsonl
from open_system_one.metrics import summarize


def flatten(examples):
    texts=[]; ys=[]; groups=[]
    for ex in examples:
        paths=paths_for_example(ex)
        texts.extend(paths)
        ys.extend(ex.target_probs)
        groups.append(len(paths))
    return texts, np.asarray(ys,dtype=np.float64), groups


def regroup(values, groups):
    out=[]; off=0
    for k in groups:
        row=np.asarray(values[off:off+k],dtype=np.float64)
        row=np.clip(row,1e-6,None)
        row=row/row.sum()
        out.append(row.tolist()); off+=k
    return out


def evaluate(model, vectorizer, examples):
    texts,_,groups=flatten(examples)
    raw=model.predict(vectorizer.transform(texts))
    probs=regroup(raw,groups)
    targets=[x.target_probs for x in examples]
    return summarize(probs,targets)


def uniform(examples):
    probs=[[1/len(x.candidates)]*len(x.candidates) for x in examples]
    return summarize(probs,[x.target_probs for x in examples])


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",default="data/simulator")
    p.add_argument("--output",default="results/cpu_smoke.json")
    args=p.parse_args()
    d=Path(args.data_dir)
    train=read_jsonl(d/"train.jsonl")
    dev=read_jsonl(d/"dev.jsonl")
    test=read_jsonl(d/"test.jsonl")
    ood=read_jsonl(d/"ood.jsonl")
    texts,y,_=flatten(train)
    vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=50000,sublinear_tf=True)
    x=vec.fit_transform(texts)
    model=Ridge(alpha=2.0).fit(x,y)
    result={
        "purpose":"CPU pipeline smoke test only; not a Jev-quality baseline",
        "train_examples":len(train),
        "dev":evaluate(model,vec,dev),
        "test":evaluate(model,vec,test),
        "ood":evaluate(model,vec,ood),
        "uniform_test":uniform(test),
        "uniform_ood":uniform(ood),
    }
    dump_json(args.output,result)
    print(result)


if __name__=="__main__":
    main()
