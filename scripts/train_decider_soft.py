#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import torch
import torch.nn.functional as F

from jev48.decider_bridge import DECIDER_MODEL
from jev48.io import read_jsonl


def _load_rows(paths: list[str], preference_repeat: int, mode: str):
    rows = []
    for path in paths:
        rows.extend(e for e in read_jsonl(path) if e.split == "train")
    if not rows:
        raise ValueError("no training rows")
    out = []
    for e in rows:
        repeat = preference_repeat if e.target_kind == "empirical_distribution" else 1
        for _ in range(repeat):
            out.append(e)
    return out


def _target_for(ex, perm, mode: str):
    p = list(ex.target_probs)
    if mode == "hard" and ex.target_kind == "empirical_distribution":
        best = max(range(len(p)), key=p.__getitem__)
        p = [float(i == best) for i in range(len(p))]
    return [p[i] for i in perm]


def _make_items(rows, tok, seed: int, max_ctx: int, schema_first_prob: float, mode: str):
    import decider.data as D
    from decider.prompt import build

    rng = random.Random(seed)
    items = []
    for e in rows:
        opts = [f"{c.id}: {c.text}" for c in e.candidates]
        upstream = D.Example(e.state, [D.Q(e.question, opts, e.gold_index)], e.domain)
        layout = "schema_first" if rng.random() < schema_first_prob else "state_first"
        it = build(upstream, tok, rng, max_options=255, max_ctx_tokens=max_ctx, layout=layout)
        if len(it["perms"]) != 1:
            raise AssertionError("Jev48 trainer expects exactly one question per row")
        it["target_probs"] = _target_for(e, it["perms"][0], mode)
        it["source_id"] = e.id
        it["target_kind"] = e.target_kind
        items.append(it)
    return items


def _soft_loss(logits: torch.Tensor, items: list[dict], brier_w: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = torch.zeros_like(logits, dtype=torch.float32)
    for i, it in enumerate(items):
        p = torch.tensor(it["target_probs"], device=logits.device, dtype=torch.float32)
        target[i, : len(p)] = p
    logp = F.log_softmax(logits.float(), dim=-1)
    # Avoid 0 * -inf for masked options.
    ce_rows = -(torch.where(target > 0, target * logp, torch.zeros_like(logp))).sum(-1)
    probs = torch.softmax(logits.float(), dim=-1)
    brier_rows = ((probs - target) ** 2).sum(-1)
    loss_rows = ce_rows + brier_w * brier_rows
    return loss_rows.mean(), ce_rows.mean().detach(), brier_rows.mean().detach()


def _copy_decider_config(source: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    local = Path(source) / "decider_config.json"
    if local.exists():
        shutil.copy2(local, destination / "decider_config.json")
        return
    try:
        from huggingface_hub import hf_hub_download
        cfg = hf_hub_download(source, "decider_config.json")
        shutil.copy2(cfg, destination / "decider_config.json")
    except Exception as exc:
        print(f"[warn] could not copy decider_config.json: {exc}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Continue decider-2b on hard or empirical soft decision targets.")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--model", default=DECIDER_MODEL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["hard", "soft"], required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--brier-w", type=float, default=0.2)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-ctx", type=int, default=16384)
    ap.add_argument("--schema-first-prob", type=float, default=0.5)
    ap.add_argument("--preference-repeat", type=int, default=3)
    ap.add_argument("--seed", type=int, default=48)
    args = ap.parse_args()
    if args.lr <= 0 or args.epochs <= 0 or args.accum <= 0 or args.preference_repeat <= 0:
        ap.error("invalid training hyperparameters")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be new/empty")
    out.mkdir(parents=True, exist_ok=True)

    from decider.model import DecisionModel, collate
    from decider.train import batches_by_tokens

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = DecisionModel(args.model, dtype=torch.bfloat16, grad_ckpt=True).cuda()
    tok = model.tok
    rows = _load_rows(args.data, args.preference_repeat, args.mode)
    items = _make_items(rows, tok, args.seed, args.max_ctx, args.schema_first_prob, args.mode)
    token_count = sum(len(x["ids"]) for x in items)
    print(f"[data] {len(items)} rows, {token_count/1e6:.2f}M tokens", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    batches0 = batches_by_tokens(items, args.max_tokens, random.Random(args.seed))
    steps_per_epoch = math.ceil(len(batches0) / args.accum)
    total_steps = max(1, int(math.ceil(steps_per_epoch * args.epochs)))

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / max(1, args.warmup)
        frac = min(1.0, (step - args.warmup) / max(1, total_steps - args.warmup))
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * frac))

    opt.zero_grad(set_to_none=True)
    started = time.time()
    step = 0
    micro = 0
    epoch = 0
    log = []
    while step < total_steps:
        order_rng = random.Random(args.seed + epoch)
        for bidx in batches_by_tokens(items, args.max_tokens, order_rng):
            batch_items = [items[i] for i in bidx]
            b = collate(batch_items, tok.pad_token_id)
            b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
            logits = model(b)
            loss, ce, br = _soft_loss(logits, batch_items, args.brier_w)
            (loss / args.accum).backward()
            micro += 1
            if micro % args.accum != 0:
                continue
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            record = {
                "step": step,
                "epoch": epoch,
                "loss": float(loss.detach()),
                "ce": float(ce),
                "brier": float(br),
                "grad_norm": float(gn),
                "lr": lr_at(step),
                "elapsed_s": time.time() - started,
            }
            log.append(record)
            if step == 1 or step % 10 == 0 or step == total_steps:
                print("[train] " + json.dumps(record), flush=True)
            if step >= total_steps:
                break
        epoch += 1

    model_dir = out / "model"
    model.lm.save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    _copy_decider_config(args.model, model_dir)
    config = {
        **vars(args),
        "base_model": args.model,
        "training_rows_after_repeat": len(rows),
        "rendered_items": len(items),
        "tokens": token_count,
        "total_steps": total_steps,
        "training_seconds": time.time() - started,
        "gpu": torch.cuda.get_device_name(0),
        "max_gpu_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "objective": "soft cross entropy + categorical Brier" if args.mode == "soft" else "majority-hard cross entropy + categorical Brier",
        "jev_outputs_used_for_training": False,
    }
    (out / "train_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "train_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"done": str(out), **config}, indent=2), flush=True)


if __name__ == "__main__":
    main()
