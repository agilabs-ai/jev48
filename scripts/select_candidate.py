#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jev48.metrics import summarize


def load(path: str):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty predictions: {path}")
    return rows


def summarize_dev(rows):
    pref = [r for r in rows if r["domain"] == "mtbench_human_preference"]
    reg = [r for r in rows if r["domain"] != "mtbench_human_preference"]
    if not pref or not reg:
        raise ValueError("dev predictions must contain MT-Bench preference and regression domains")
    fn = lambda rs: summarize([r["probabilities"] for r in rs], [r["target_probs"] for r in rs])
    return fn(pref), fn(reg)


def parse_candidate(text: str):
    # name|dev_predictions|model_path
    parts = text.split("|", 2)
    if len(parts) != 3:
        raise ValueError("candidate must be name|dev_predictions|model_path")
    return parts


def main() -> None:
    ap = argparse.ArgumentParser(description="Select Jev48 candidate using dev only; never read locked test/OOD.")
    ap.add_argument("--base", required=True, help="Base-model dev predictions")
    ap.add_argument("--base-model", default="Mapika/decider-2b")
    ap.add_argument("--candidate", action="append", default=[], help="name|dev_predictions|model_path")
    ap.add_argument("--max-regression-drop", type=float, default=0.015)
    ap.add_argument("--min-brier-improvement", type=float, default=0.002)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    base_pref, base_reg = summarize_dev(load(args.base))
    records = [{"name": "base", "model": args.base_model, "preference": base_pref, "regression": base_reg, "eligible": True}]
    eligible = []
    for spec in args.candidate:
        name, pred_path, model_path = parse_candidate(spec)
        pref, reg = summarize_dev(load(pred_path))
        reg_drop = base_reg["accuracy"] - reg["accuracy"]
        brier_gain = base_pref["brier"] - pref["brier"]
        ok = reg_drop <= args.max_regression_drop and brier_gain >= args.min_brier_improvement
        rec = {
            "name": name,
            "model": model_path,
            "preference": pref,
            "regression": reg,
            "regression_accuracy_drop": reg_drop,
            "preference_brier_improvement": brier_gain,
            "eligible": ok,
        }
        records.append(rec)
        if ok:
            eligible.append(rec)

    if eligible:
        winner = min(eligible, key=lambda x: (x["preference"]["brier"], x["preference"]["nll"]))
        reason = "lowest preference-dev Brier among candidates passing the predeclared regression and improvement gates"
    else:
        winner = records[0]
        reason = "no fine-tuned candidate passed both the dev Brier improvement and regression guard; preserve base"
    result = {
        "selection_uses_locked_test_or_ood": False,
        "gates": {
            "max_regression_accuracy_drop": args.max_regression_drop,
            "min_preference_brier_improvement": args.min_brier_improvement,
        },
        "winner": winner,
        "reason": reason,
        "all": records,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
