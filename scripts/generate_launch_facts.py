#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pct(x):
    return "—" if x is None else f"{100*float(x):.1f}%"


def num(x):
    return "—" if x is None else f"{float(x):.4f}"


def signed_pp(a, b):
    if a is None or b is None:
        return "—"
    return f"{100*(float(a)-float(b)):+.1f} pp"


def signed(a, b):
    if a is None or b is None:
        return "—"
    return f"{float(a)-float(b):+.4f}"


def main():
    ap = argparse.ArgumentParser(description="Generate factual launch bullets from the locked Jev48 report.")
    ap.add_argument("--report", default="results/final_report.json")
    ap.add_argument("--run-manifest", default="results/run_manifest.json")
    ap.add_argument("--output", default="results/LAUNCH_FACTS.md")
    args = ap.parse_args()

    report = read(args.report)
    manifest = read(args.run_manifest) if Path(args.run_manifest).exists() else {}
    rows = {r["model"]: r for r in report["rows"]}
    ours = rows["Jev48 / ChatGPT build"]
    base = next(r for n, r in rows.items() if n.startswith("decider-2b"))
    jev = rows.get("Jev")
    winner = report["selection"]["winner"]

    lines = [
        "# Jev48 launch facts",
        "",
        "Machine-generated from the locked result receipts. Do not edit numbers by hand.",
        "",
        f"- Selected reproduction: **{winner['name']}**.",
        f"- Public starting point: **Mapika/decider-2b** at pinned commit `{manifest.get('upstream_commit', 'see run manifest')}`.",
        f"- Locked human-preference test: Jev48 **{pct(ours['mtbench_accuracy'])} accuracy**, **{num(ours['mtbench_brier'])} Brier**.",
        f"- Locked transfer OOD: Jev48 **{pct(ours['transfer_accuracy'])} accuracy**, **{num(ours['transfer_brier'])} Brier**.",
        f"- Versus public starting point on human preference: **{signed_pp(ours['mtbench_accuracy'], base['mtbench_accuracy'])} accuracy**, **{signed(ours['mtbench_brier'], base['mtbench_brier'])} Brier** (negative Brier delta is better).",
        f"- Versus public starting point on transfer: **{signed_pp(ours['transfer_accuracy'], base['transfer_accuracy'])} accuracy**, **{signed(ours['transfer_brier'], base['transfer_brier'])} Brier**.",
        "- Jev outputs used for training/model selection: **0**.",
        f"- External GPU experiment elapsed time: **{manifest.get('elapsed_seconds', 'see run manifest')} seconds**.",
        f"- Estimated {manifest.get('gpu', manifest.get('modal_gpu_requested', 'GPU'))} list cost: **${float(manifest['estimated_gpu_list_cost_usd']):.2f}** (list-price estimate only; see run manifest for exclusions)." if manifest.get("estimated_gpu_list_cost_usd") is not None else "- Estimated GPU list cost: **see run manifest after cloud execution**.",
    ]
    if jev:
        lines += [
            f"- Native Jev on the same locked human-preference test: **{pct(jev['mtbench_accuracy'])} accuracy**, **{num(jev['mtbench_brier'])} Brier**.",
            f"- Native Jev on the same locked transfer OOD: **{pct(jev['transfer_accuracy'])} accuracy**, **{num(jev['transfer_brier'])} Brier**.",
            f"- Jev48 minus Jev human-preference accuracy: **{signed_pp(ours['mtbench_accuracy'], jev['mtbench_accuracy'])}**.",
            f"- Jev48 minus Jev transfer accuracy: **{signed_pp(ours['transfer_accuracy'], jev['transfer_accuracy'])}**.",
            f"- Jev48 minus Jev human-preference Brier: **{signed(ours['mtbench_brier'], jev['mtbench_brier'])}**.",
            f"- Jev48 minus Jev transfer Brier: **{signed(ours['transfer_brier'], jev['transfer_brier'])}**.",
        ]
    else:
        lines.append("- Live Jev comparison: **not run yet**.")
    lines += [
        "",
        "## Claim guardrails",
        "",
        "- Say: `I gave ChatGPT 48 hours/a weekend to recreate Jev using anything publicly available.`",
        "- Do not say: `from scratch`.",
        "- Do not imply the upstream architecture or weights were created by Jev48.",
        "- If a metric is not in this file or the locked report, do not invent it.",
    ]
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
