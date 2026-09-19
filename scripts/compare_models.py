#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract(path: Path) -> dict:
    raw = load(path)
    metrics = raw.get("metrics") or raw.get("test") or raw
    name = raw.get("model") or path.stem.replace(".summary", "")
    return {
        "model": str(name),
        "accuracy": metrics.get("accuracy"),
        "brier": metrics.get("brier"),
        "nll": metrics.get("nll"),
        "ece_15": metrics.get("ece_15"),
        "total_variation": metrics.get("total_variation"),
        "questions_per_second": raw.get("questions_per_second"),
        "benchmark_sha256": raw.get("benchmark_sha256"),
        "test_accuracy": raw.get("slices", {}).get("by_split", {}).get("test", {}).get("accuracy"),
        "test_brier": raw.get("slices", {}).get("by_split", {}).get("test", {}).get("brier"),
        "ood_accuracy": raw.get("slices", {}).get("by_split", {}).get("ood", {}).get("accuracy"),
        "ood_brier": raw.get("slices", {}).get("by_split", {}).get("ood", {}).get("brier"),
        "source": str(path),
    }


def fmt(x, digits=4):
    return "—" if x is None else f"{float(x):.{digits}f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("summaries", nargs="+")
    p.add_argument("--output", default="results/model_comparison.md")
    args = p.parse_args()
    rows = [extract(Path(x)) for x in args.summaries]
    hashes = {r["benchmark_sha256"] for r in rows if r["benchmark_sha256"]}
    warning = None
    if len(hashes) > 1:
        warning = "WARNING: summary files contain different benchmark hashes; quality numbers may not be comparable."

    lines = ["# Model comparison", ""]
    if warning:
        lines += [f"> {warning}", ""]
    lines += [
        "| Model | Overall acc ↑ | Overall Brier ↓ | Seen acc ↑ | Seen Brier ↓ | OOD acc ↑ | OOD Brier ↓ | Q/s ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {fmt(r['accuracy'])} | {fmt(r['brier'])} | "
            f"{fmt(r['test_accuracy'])} | {fmt(r['test_brier'])} | {fmt(r['ood_accuracy'])} | "
            f"{fmt(r['ood_brier'])} | {fmt(r['questions_per_second'], 2)} |"
        )
    lines += ["", "Latency/throughput is only valid when execution scopes and hardware are matched.", ""]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"rows": rows, "warning": warning}, indent=2) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
