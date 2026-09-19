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
    if value != "auto": return value
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available(): return "mps"
    return "cpu"


def load_config(path: str) -> dict:
    cfg=yaml.safe_load(Path(path).read_text())
    for key in ["base_model","data_dir","output_dir"]:
        if key not in cfg: raise ValueError(f"missing config key: {key}")
    if cfg.get("train_mode") != "lora": raise ValueError("LoRA trainer requires train_mode: lora")
    return cfg


def trainable_state(model):
    # LoRA + decision head only. This is small enough to keep an in-memory best checkpoint.
    return {name:p.detach().cpu().clone() for name,p in model.named_parameters() if p.requires_grad}


def restore_trainable_state(model,state):
    params=dict(model.named_parameters())
    for name,value in state.items():
        params[name].data.copy_(value.to(device=params[name].device,dtype=params[name].dtype))


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/train_lora_public_fast.yaml"); args=p.parse_args()
    cfg=load_config(args.config)
    try:
        from transformers import AutoTokenizer
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("Install training deps: pip install -e '.[train]'") from exc

    seed=int(cfg.get("seed",17)); rng=random.Random(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device=resolve_device(cfg.get("device","auto")); dtype=cfg.get("dtype","bf16" if device=="cuda" else "auto")
    max_length=int(cfg.get("max_length",2048)); batch_questions=int(cfg.get("batch_questions",1))
    output=Path(cfg["output_dir"]); output.mkdir(parents=True,exist_ok=True); data=Path(cfg["data_dir"])
    train=read_jsonl(data/"train.jsonl"); dev=read_jsonl(data/"dev.jsonl"); calibration=read_jsonl(data/"calibration.jsonl")
    test=read_jsonl(data/"test.jsonl"); ood=read_jsonl(data/"ood.jsonl") if (data/"ood.jsonl").exists() else []

    tok=AutoTokenizer.from_pretrained(cfg["base_model"],revision=cfg.get("revision"),trust_remote_code=False)
    tok.padding_side="right"
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    backbone=load_hf_backbone(cfg["base_model"],device=device,dtype=dtype,revision=cfg.get("revision"))
    if bool(cfg.get("gradient_checkpointing",True)) and hasattr(backbone,"gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
    targets=cfg.get("lora_target_modules",["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    lora=LoraConfig(
        r=int(cfg.get("lora_r",8)), lora_alpha=int(cfg.get("lora_alpha",16)),
        lora_dropout=float(cfg.get("lora_dropout",0.0)), target_modules=targets,
        bias="none",
    )
    backbone=get_peft_model(backbone,lora)
    model=DynamicDecisionModel(backbone,set_attention=bool(cfg.get("set_attention",True)),set_dim=int(cfg.get("set_dim",128))).to(device)
    trainable=[p for p in model.parameters() if p.requires_grad]
    if not trainable: raise RuntimeError("no trainable parameters after LoRA setup")
    optimizer=torch.optim.AdamW(trainable,lr=float(cfg.get("lr",1e-4)),weight_decay=float(cfg.get("weight_decay",0.01)))

    metadata={
        "format_version":"open-system-one-lora-v1","base_model":cfg["base_model"],"revision":cfg.get("revision"),
        "train_mode":"lora","adapter_path":"adapter","set_attention":bool(cfg.get("set_attention",True)),
        "set_dim":int(cfg.get("set_dim",128)),"max_length":max_length,"inference_dtype":dtype,"temperature":1.0,
        "prefix_sharing":False,"zero_output_decoding":True,"seed":seed,"lora_r":int(cfg.get("lora_r",8)),
        "lora_alpha":int(cfg.get("lora_alpha",16)),"lora_target_modules":targets,
        "trainable_parameters":sum(p.numel() for p in trainable),"total_parameters":sum(p.numel() for p in model.parameters()),
    }
    dump_json(output/"run_config.json",{**cfg,**metadata})
    print(json.dumps({"trainable":metadata["trainable_parameters"],"total":metadata["total_parameters"]}),flush=True)

    steps=int(cfg.get("steps",150)); eval_every=int(cfg.get("eval_every",25)); ce_weight=float(cfg.get("ce_weight",1.0)); brier_weight=float(cfg.get("brier_weight",0.25))
    logs=[]; best=float("inf"); best_step=None; best_state=None; started=time.perf_counter()
    for step in range(1,steps+1):
        group=rng.sample(train,min(batch_questions,len(train)))
        model.train(); optimizer.zero_grad(set_to_none=True)
        batch=collate_examples(group,tok,max_length,permute_candidates=bool(cfg.get("permute_candidates",True)),rng=rng).to(device)
        out=model(batch.input_ids,batch.attention_mask,batch.group_sizes)
        loss,parts=soft_target_loss(out.logits,out.valid_mask,batch.targets,ce_weight=ce_weight,brier_weight=brier_weight)
        if not torch.isfinite(loss): raise RuntimeError("nonfinite loss")
        loss.backward(); torch.nn.utils.clip_grad_norm_(trainable,float(cfg.get("grad_clip",1.0))); optimizer.step()
        item={"step":step,**parts,"elapsed_seconds":time.perf_counter()-started}
        if step%eval_every==0 or step==steps:
            dev_metrics,_,_,_=predict_examples(model,tok,dev,device=device,max_length=max_length,batch_questions=1)
            item["dev"]=dev_metrics; score=dev_metrics["nll"]
            if score<best:
                best=score; best_step=step; best_state=trainable_state(model)
        logs.append(item)
        if step==1 or step%eval_every==0: print(json.dumps(item),flush=True)

    if best_state is None: raise RuntimeError("no checkpoint selected")
    restore_trainable_state(model,best_state)
    # Save the PEFT adapter and decision head as separate, explicit artifacts.
    model.backbone.save_pretrained(output/"adapter")
    save_head_checkpoint(output,model,metadata)

    _,cal_logits,cal_targets,_=predict_examples(model,tok,calibration,device=device,max_length=max_length,batch_questions=1)
    scaler=TemperatureScaler(1.0); temperature=scaler.fit(cal_logits,[torch.tensor(x,dtype=torch.float32) for x in cal_targets])
    metadata["temperature"]=temperature; save_head_checkpoint(output,model,metadata)
    test_metrics,_,_,test_records=predict_examples(model,tok,test,device=device,max_length=max_length,batch_questions=1,temperature=temperature)
    ood_metrics=None; ood_records=[]
    if ood: ood_metrics,_,_,ood_records=predict_examples(model,tok,ood,device=device,max_length=max_length,batch_questions=1,temperature=temperature)
    summary={"best_step":best_step,"temperature":temperature,"test":test_metrics,"ood":ood_metrics,"training_seconds":time.perf_counter()-started,"device":device}
    dump_json(output/"summary.json",summary); dump_json(output/"train_log.json",logs)
    (output/"test_predictions.jsonl").write_text("".join(json.dumps(x)+"\n" for x in test_records))
    if ood_records: (output/"ood_predictions.jsonl").write_text("".join(json.dumps(x)+"\n" for x in ood_records))
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
