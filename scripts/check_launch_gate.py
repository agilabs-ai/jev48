#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metric(summary: dict, slice_name: str | None, key: str):
    if slice_name is None:
        return summary["metrics"][key]
    return summary.get("slices", {}).get("by_split", {}).get(slice_name, {}).get(key)


def main() -> None:
    p = argparse.ArgumentParser(description="Apply predeclared marketing gates to Open System One vs NanoJev.")
    p.add_argument("--ours", required=True)
    p.add_argument("--nanojev", required=True)
    p.add_argument("--output", default="results/launch_gate.json")
    p.add_argument("--accuracy-noninferiority-pp", type=float, default=2.0)
    p.add_argument("--accuracy-win-pp", type=float, default=5.0)
    p.add_argument("--brier-regression-tolerance", type=float, default=0.02)
    args = p.parse_args()

    ours, nano = load(args.ours), load(args.nanojev)
    oh, nh = ours.get("benchmark_sha256"), nano.get("benchmark_sha256")
    if oh and nh and oh != nh:
        raise ValueError("benchmark hash mismatch; refusing launch-gate comparison")

    oa, na = float(metric(ours, None, "accuracy")), float(metric(nano, None, "accuracy"))
    ob, nb = float(metric(ours, None, "brier")), float(metric(nano, None, "brier"))
    acc_noninferior = oa >= na - args.accuracy_noninferiority_pp / 100.0
    lower_brier = ob < nb
    accuracy_clear_win = oa >= na + args.accuracy_win_pp / 100.0
    calibration_not_catastrophic = ob <= nb + args.brier_regression_tolerance
    gate_a = (lower_brier and acc_noninferior) or (accuracy_clear_win and calibration_not_catastrophic)

    ood_ours = ours.get("slices", {}).get("by_split", {}).get("ood")
    ood_nano = nano.get("slices", {}).get("by_split", {}).get("ood")
    gate_b = None
    if ood_ours and ood_nano:
        gate_b = (
            float(ood_ours["brier"]) < float(ood_nano["brier"])
            and float(ood_ours["accuracy"]) >= float(ood_nano["accuracy"]) - 0.02
        )

    result = {
        "benchmark_sha256": oh or nh,
        "gate_a_semantic_generality": {
            "pass": gate_a,
            "rule": "lower Brier + accuracy within 2pp, OR >=5pp accuracy win with Brier regression <=0.02",
            "ours": {"accuracy": oa, "brier": ob},
            "nanojev": {"accuracy": na, "brier": nb},
        },
        "gate_b_ood": {
            "pass": gate_b,
            "rule": "lower OOD Brier + OOD accuracy within 2pp",
            "ours": ood_ours,
            "nanojev": ood_nano,
        },
        "headline_better_than_nanojev_allowed": bool(gate_a or gate_b),
        "note": "This gate only covers quality. Any speed claim requires matched persistent-serving hardware/scope.",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
