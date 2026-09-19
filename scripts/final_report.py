#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fmt(v):
    return "—" if v is None else f"{float(v):.4f}"


def calibrated_row(name, summary):
    cal = summary["calibrated"]
    test = cal.get("by_split", {}).get("test", {})
    ood = cal.get("by_split", {}).get("ood", {})
    return {
        "model": name,
        "probability_mode": "post-hoc scalar calibrated on Jev48 calibration split",
        "temperature": summary.get("temperature"),
        "mtbench_accuracy": test.get("accuracy"),
        "mtbench_brier": test.get("brier"),
        "mtbench_ece": test.get("ece_15"),
        "transfer_accuracy": ood.get("accuracy"),
        "transfer_brier": ood.get("brier"),
        "transfer_ece": ood.get("ece_15"),
    }


def native_row(name, summary):
    metrics = summary["metrics"]
    test = metrics.get("by_split", {}).get("test", {})
    ood = metrics.get("by_split", {}).get("ood", {})
    return {
        "model": name,
        "probability_mode": "native provider probabilities",
        "temperature": None,
        "mtbench_accuracy": test.get("accuracy"),
        "mtbench_brier": test.get("brier"),
        "mtbench_ece": test.get("ece_15"),
        "transfer_accuracy": ood.get("accuracy"),
        "transfer_brier": ood.get("brier"),
        "transfer_ece": ood.get("ece_15"),
    }


def main():
    ap = argparse.ArgumentParser(description="Create the locked Jev48 head-to-head table.")
    ap.add_argument("--base", required=True, help="Calibrated base summary")
    ap.add_argument("--ours", required=True, help="Calibrated Jev48 summary")
    ap.add_argument("--selection", required=True)
    ap.add_argument("--jev-native", help="Raw/native Jev benchmark summary")
    ap.add_argument("--jev-calibrated", help="Optional post-hoc Jev calibration diagnostic")
    ap.add_argument("--output", default="results/final_report.md")
    args = ap.parse_args()

    selection = read(args.selection)
    base_summary, ours_summary = read(args.base), read(args.ours)
    rows = [
        calibrated_row("decider-2b (public starting point)", base_summary),
        calibrated_row("Jev48 / ChatGPT build", ours_summary),
    ]
    hashes = {base_summary["benchmark_sha256"], ours_summary["benchmark_sha256"]}
    diagnostics = {}
    if args.jev_native:
        native = read(args.jev_native)
        rows.append(native_row("Jev", native))
        hashes.add(native["benchmark_sha256"])
    if args.jev_calibrated:
        diag = read(args.jev_calibrated)
        hashes.add(diag["benchmark_sha256"])
        diagnostics["jev_posthoc_calibration"] = diag
    if len(hashes) != 1:
        raise ValueError("final summaries do not share one benchmark hash")

    lines = [
        "# Jev48 locked comparison",
        "",
        f"Selected on dev only: `{selection['winner']['name']}`. Locked test/OOD were not used for model selection.",
        "",
        "| Model | Human preference acc ↑ | Human preference Brier ↓ | Human preference ECE ↓ | Transfer acc ↑ | Transfer Brier ↓ | Transfer ECE ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {fmt(r['mtbench_accuracy'])} | {fmt(r['mtbench_brier'])} | {fmt(r['mtbench_ece'])} | "
            f"{fmt(r['transfer_accuracy'])} | {fmt(r['transfer_brier'])} | {fmt(r['transfer_ece'])} |"
        )
    lines += [
        "",
        "Human preference = locked MT-Bench expert-vote groups over real model outputs. Transfer = pinned tasks held out by the public starting model's training registry.",
        "The open base and Jev48 each receive one scalar temperature fitted on the same calibration rows. Jev's primary row uses the probabilities returned natively by the provider; any extra Jev temperature fit is retained only as a diagnostic.",
        "Jev outputs are never used for training or candidate selection.",
        "",
    ]
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({
        "benchmark_sha256": next(iter(hashes)),
        "selection": selection,
        "rows": rows,
        "diagnostics": diagnostics,
    }, indent=2) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
