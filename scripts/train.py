#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import time

import torch
import yaml

from open_system_one.batching import collate_examples
from open_system_one.calibration import TemperatureScaler
from open_system_one.evaluation import predict_examples
from open_system_one.io import dump_json, read_jsonl
from open_system_one.model import DynamicDecisionModel, load_hf_backbone, save_head_checkpoint, soft_target_loss


def resolve_device(value: str) -> str:
    if value != "auto":
        return value
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_config(path: str) -> dict:
    cfg=yaml.safe_load(Path(path).read_text())
    required=["base_model","data_dir","output_dir"]
    missing=[x for x in required if x not in cfg]
    if missing:
        raise ValueError(f"missing config keys: {missing}")
    return cfg


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/train_head.yaml")
    args=ap.parse_args()
    cfg=load_config(args.config)
    if cfg.get("train_mode","head_only") != "head_only":
        raise ValueError("v0 trainer intentionally supports head_only first. Run the decision gate before adding LoRA/full tuning.")

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install training deps: pip install -e '.[train]'") from exc

    seed=int(cfg.get("seed",17))
    rng=random.Random(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device=resolve_device(cfg.get("device","auto"))
    dtype=cfg.get("dtype","bf16" if device=="cuda" else "auto")
    max_length=int(cfg.get("max_length",512))
    batch_questions=int(cfg.get("batch_questions",8))
    output=Path(cfg["output_dir"]); output.mkdir(parents=True,exist_ok=True)
    data=Path(cfg["data_dir"])

    train=read_jsonl(data/"train.jsonl")
    dev=read_jsonl(data/"dev.jsonl")
    calibration=read_jsonl(data/"calibration.jsonl")
    test=read_jsonl(data/"test.jsonl")
    ood=read_jsonl(data/"ood.jsonl") if (data/"ood.jsonl").exists() else []
    if not all([train,dev,calibration,test]):
        raise ValueError("train/dev/calibration/test splits must all be nonempty")

    tokenizer=AutoTokenizer.from_pretrained(cfg["base_model"],revision=cfg.get("revision"),trust_remote_code=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    backbone=load_hf_backbone(cfg["base_model"],device=device,dtype=dtype,revision=cfg.get("revision"))
    for p in backbone.parameters(): p.requires_grad_(False)
    backbone.eval()
    model=DynamicDecisionModel(backbone,set_attention=bool(cfg.get("set_attention",True)),set_dim=int(cfg.get("set_dim",128))).to(device)
    head=[p for name,p in model.named_parameters() if not name.startswith("backbone.")]
    optimizer=torch.optim.AdamW(head,lr=float(cfg.get("head_lr",2e-4)),weight_decay=float(cfg.get("weight_decay",0.01)))

    metadata={
        "format_version":"open-system-one-head-v1",
        "base_model":cfg["base_model"],"revision":cfg.get("revision"),
        "train_mode":"head_only","set_attention":bool(cfg.get("set_attention",True)),"set_dim":int(cfg.get("set_dim",128)),
        "max_length":max_length,"inference_dtype":dtype,"temperature":1.0,
        "prefix_sharing":False,"zero_output_decoding":True,"seed":seed,
        "train_examples":len(train),"dev_examples":len(dev),"calibration_examples":len(calibration),"test_examples":len(test),"ood_examples":len(ood),
    }
    dump_json(output/"run_config.json",{**cfg,**metadata})

    steps=int(cfg.get("steps",300)); eval_every=int(cfg.get("eval_every",25))
    ce_weight=float(cfg.get("ce_weight",1.0)); brier_weight=float(cfg.get("brier_weight",0.25))
    logs=[]; best=float("inf"); best_step=None; start=time.perf_counter()

    for step in range(1,steps+1):
        group=rng.sample(train,min(batch_questions,len(train)))
        model.train(); model.backbone.eval(); optimizer.zero_grad(set_to_none=True)
        batch=collate_examples(group,tokenizer,max_length,permute_candidates=bool(cfg.get("permute_candidates",True)),rng=rng).to(device)
        out=model(batch.input_ids,batch.attention_mask,batch.group_sizes)
        loss,parts=soft_target_loss(out.logits,out.valid_mask,batch.targets,ce_weight=ce_weight,brier_weight=brier_weight)
        if not torch.isfinite(loss): raise RuntimeError("nonfinite loss")
        loss.backward(); torch.nn.utils.clip_grad_norm_(head,float(cfg.get("grad_clip",1.0))); optimizer.step()
        item={"step":step,**parts,"elapsed_seconds":time.perf_counter()-start}
        if step % eval_every == 0 or step == steps:
            dev_metrics,_,_,_=predict_examples(model,tokenizer,dev,device=device,max_length=max_length,batch_questions=batch_questions)
            item["dev"]=dev_metrics
            score=dev_metrics["nll"]
            if score < best:
                best=score; best_step=step
                save_head_checkpoint(output,model,metadata)
        logs.append(item)
        if step==1 or step%max(1,eval_every)==0:
            print(json.dumps(item),flush=True)

    if best_step is None: raise RuntimeError("no best checkpoint selected")
    from safetensors.torch import load_file
    model.load_head_state_dict(load_file(str(output/"decision_head.safetensors")))

    _,cal_logits,cal_targets,_=predict_examples(model,tokenizer,calibration,device=device,max_length=max_length,batch_questions=batch_questions)
    scaler=TemperatureScaler(1.0)
    temperature=scaler.fit(cal_logits,[torch.tensor(x,dtype=torch.float32) for x in cal_targets])
    metadata["temperature"]=temperature
    save_head_checkpoint(output,model,metadata)

    test_metrics,_,_,test_records=predict_examples(model,tokenizer,test,device=device,max_length=max_length,batch_questions=batch_questions,temperature=temperature)
    ood_metrics=None; ood_records=[]
    if ood:
        ood_metrics,_,_,ood_records=predict_examples(model,tokenizer,ood,device=device,max_length=max_length,batch_questions=batch_questions,temperature=temperature)
    summary={
        "best_step":best_step,"selected_on":"dev_nll","temperature":temperature,
        "test":test_metrics,"ood":ood_metrics,"training_seconds":time.perf_counter()-start,
        "device":device,"zero_output_decoding":True,"prefix_sharing":False,
    }
    dump_json(output/"summary.json",summary)
    dump_json(output/"train_log.json",logs)
    (output/"test_predictions.jsonl").write_text("".join(json.dumps(x)+"\n" for x in test_records))
    if ood_records:
        (output/"ood_predictions.jsonl").write_text("".join(json.dumps(x)+"\n" for x in ood_records))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
