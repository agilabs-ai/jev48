#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import random
import re
import time

import torch
from torch import nn

from open_system_one.batching import collate_examples
from open_system_one.calibration import TemperatureScaler
from open_system_one.evaluation import predict_examples
from open_system_one.formatting import paths_for_example
from open_system_one.io import dump_json, read_jsonl
from open_system_one.model import DynamicDecisionModel, soft_target_loss


TOKEN_RE=re.compile(r"[A-Za-z0-9_$%+.-]+|[^\s]")


class SimpleTokenizer:
    def __init__(self, texts, max_vocab=6000):
        counts=Counter(tok.lower() for text in texts for tok in TOKEN_RE.findall(text))
        self.vocab={"<pad>":0,"<unk>":1}
        for token,_ in counts.most_common(max_vocab-2): self.vocab[token]=len(self.vocab)
        self.pad_token_id=0

    def __call__(self,texts,padding=True,truncation=False,return_tensors="pt",add_special_tokens=True):
        rows=[]
        for text in texts:
            ids=[self.vocab.get(t.lower(),1) for t in TOKEN_RE.findall(text)] or [1]
            rows.append(ids)
        width=max(map(len,rows))
        input_ids=torch.zeros((len(rows),width),dtype=torch.long)
        mask=torch.zeros_like(input_ids)
        for i,row in enumerate(rows):
            input_ids[i,:len(row)]=torch.tensor(row); mask[i,:len(row)]=1
        return {"input_ids":input_ids,"attention_mask":mask}


class TinyMeanBackbone(nn.Module):
    def __init__(self,vocab,hidden=64):
        super().__init__(); self.config=SimpleNamespace(hidden_size=hidden); self.emb=nn.Embedding(vocab,hidden,padding_idx=0); self.proj=nn.Sequential(nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,hidden))
    def forward(self,input_ids,attention_mask,use_cache=False):
        x=self.emb(input_ids)*attention_mask.unsqueeze(-1)
        cum=x.cumsum(1)
        denom=attention_mask.cumsum(1).clamp_min(1).unsqueeze(-1)
        h=self.proj(cum/denom)
        return SimpleNamespace(last_hidden_state=h)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-dir",default="data/simulator"); p.add_argument("--output",default="results/tiny_cpu_train.json"); p.add_argument("--steps",type=int,default=250); p.add_argument("--seed",type=int,default=17); args=p.parse_args()
    rng=random.Random(args.seed); torch.manual_seed(args.seed)
    d=Path(args.data_dir)
    train=read_jsonl(d/"train.jsonl"); dev=read_jsonl(d/"dev.jsonl"); cal=read_jsonl(d/"calibration.jsonl"); test=read_jsonl(d/"test.jsonl"); ood=read_jsonl(d/"ood.jsonl")
    all_train_paths=[p for ex in train for p in paths_for_example(ex)]
    tok=SimpleTokenizer(all_train_paths)
    model=DynamicDecisionModel(TinyMeanBackbone(len(tok.vocab)),set_attention=True,set_dim=64)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=0.001)
    best=1e9; best_state=None; logs=[]; start=time.perf_counter()
    for step in range(1,args.steps+1):
        group=rng.sample(train,16)
        batch=collate_examples(group,tok,192,permute_candidates=True,rng=rng)
        model.train(); opt.zero_grad(set_to_none=True)
        out=model(batch.input_ids,batch.attention_mask,batch.group_sizes)
        loss,parts=soft_target_loss(out.logits,out.valid_mask,batch.targets,ce_weight=1,brier_weight=.3)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
        if step%25==0 or step==args.steps:
            metrics,_,_,_=predict_examples(model,tok,dev,device="cpu",max_length=192,batch_questions=32)
            if metrics["nll"]<best:
                best=metrics["nll"]; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            logs.append({"step":step,**parts,"dev":metrics}); print(step,metrics["accuracy"],metrics["brier"],flush=True)
    model.load_state_dict(best_state)
    _,zs,ys,_=predict_examples(model,tok,cal,device="cpu",max_length=192,batch_questions=32)
    scaler=TemperatureScaler(); temp=scaler.fit(zs,[torch.tensor(y) for y in ys])
    test_m,_,_,_=predict_examples(model,tok,test,device="cpu",max_length=192,batch_questions=32,temperature=temp)
    ood_m,_,_,_=predict_examples(model,tok,ood,device="cpu",max_length=192,batch_questions=32,temperature=temp)
    result={"purpose":"End-to-end CPU validation of the exact dynamic-head training path; not a language-model benchmark","steps":args.steps,"vocab":len(tok.vocab),"temperature":temp,"test":test_m,"ood":ood_m,"seconds":time.perf_counter()-start,"logs":logs}
    dump_json(args.output,result); print(json.dumps({k:v for k,v in result.items() if k!="logs"},indent=2))


if __name__=="__main__": main()
