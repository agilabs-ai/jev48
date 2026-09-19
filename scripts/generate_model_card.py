#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def f(x):
    return "—" if x is None else f"{float(x):.4f}"


def main():
    ap = argparse.ArgumentParser(description="Generate a derivative model card from frozen Jev48 receipts.")
    ap.add_argument("--selection", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    sel, summary, source = read(args.selection), read(args.summary), read(args.source)
    cal = summary["calibrated"]
    test = cal.get("by_split", {}).get("test", {})
    ood = cal.get("by_split", {}).get("ood", {})
    text = f'''# Jev48 derivative model\n\nThis checkpoint is the model selected by the **Jev48** weekend reproduction experiment.\n\nIt is a derivative of **Mapika/decider-2b**. The upstream architecture and starting weights are not Jev48 inventions. The Jev48 change tested empirical soft human-vote distributions against hard-majority preference labels while retaining replay data.\n\n## Lineage\n\n- Starting model: `{source["base_model"]}`\n- Upstream repository: `{source["upstream_repo"]}`\n- Upstream commit: `{source["upstream_commit"]}`\n- Selected trial: `{sel["winner"]["name"]}`\n- Benchmark SHA-256: `{summary["benchmark_sha256"]}`\n- Fitted temperature: `{summary["temperature"]}`\n\n## Locked evaluation\n\n| Slice | Accuracy ↑ | Brier ↓ | ECE ↓ |\n|---|---:|---:|---:|\n| MT-Bench expert-vote test | {f(test.get("accuracy"))} | {f(test.get("brier"))} | {f(test.get("ece_15"))} |\n| Held-out transfer OOD | {f(ood.get("accuracy"))} | {f(ood.get("brier"))} | {f(ood.get("ece_15"))} |\n\nTemperature was fitted only on the frozen calibration split after model selection. Locked test/OOD were not used for candidate selection. Jev outputs were not used for training.\n\n## Intended use\n\nTyped, generation-free decision scoring through the upstream `decider` System One-compatible interface. Validate calibration on your own outcome-labelled workload before automating consequential decisions.\n\n## Attribution and licenses\n\nJev48 orchestration/evaluation code is MIT. This model is derived from `Mapika/decider-2b`; preserve all applicable Apache-2.0 and underlying Qwen/model/dataset obligations. See the Jev48 repository `THIRD_PARTY_NOTICES.md`.\n'''
    out=Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(text, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
