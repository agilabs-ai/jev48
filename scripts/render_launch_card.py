#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", type=Path, default=Path("receipts"))
    parser.add_argument("--output", type=Path, default=Path("launch/benchmark-card.html"))
    args = parser.parse_args()

    report = read(args.receipts / "results/final_report.json")
    public = read(args.receipts / "results/typed-decisions.summary.json")
    bootstrap = read(args.receipts / "results/bootstrap-jev48-vs-base.json")
    run = read(args.receipts / "results/run_manifest.json")
    base, ours = report["rows"][:2]
    jev = public["jev_reference"]
    public_ours = public["metrics"]
    ci = bootstrap["by_split"]["test"]["brier_difference_a_minus_b"]
    gap = (jev["accuracy"] - public_ours["accuracy"]) * 100
    improvement = -ci["point"]
    minutes = round(run["elapsed_seconds"] / 60)

    body = f"""<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box}}html,body{{margin:0;width:1080px;height:1350px;overflow:hidden}}
body{{background:#090909;color:#f5f3ed;font-family:Inter,Arial,sans-serif}}
.card{{height:100%;padding:72px 72px 78px;display:flex;flex-direction:column;background:radial-gradient(circle at 90% 2%,#2b190b 0,transparent 30%),linear-gradient(160deg,#111 0,#080808 70%)}}
.eyebrow,.label,.source{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;text-transform:uppercase;letter-spacing:.12em}}
.eyebrow{{font-size:18px;color:#f38b32}}h1{{font-size:82px;line-height:.94;letter-spacing:-.055em;margin:30px 0 24px;max-width:870px}}.lede{{font-size:25px;line-height:1.42;color:#bdbab1;max-width:850px;margin:0 0 52px}}
.score{{border-top:1px solid #3a3834;border-bottom:1px solid #3a3834;padding:34px 0 30px}}
.label{{font-size:15px;color:#8d8a83;margin-bottom:24px}}.barrow{{display:grid;grid-template-columns:185px 1fr 110px;align-items:center;gap:18px;margin:18px 0;font-size:24px}}.track{{height:20px;background:#262522;border-radius:99px;overflow:hidden}}.fill{{height:100%;border-radius:99px}}.jev{{background:#f38b32}}.ours{{background:#eeeae0}}.val{{font-size:28px;text-align:right;font-weight:700}}
.gap{{margin-top:23px;color:#f38b32;font-size:18px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
.proof{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:28px}}.proof>div{{background:#151513;border:1px solid #302f2b;border-radius:16px;padding:24px}}.big{{font-size:38px;font-weight:750;letter-spacing:-.035em;margin:8px 0}}.small{{font-size:16px;line-height:1.45;color:#aaa79f}}
.footer{{margin-top:auto;padding-top:28px;border-top:1px solid #302f2b;display:flex;justify-content:space-between;gap:30px;align-items:flex-end}}.source{{font-size:12px;line-height:1.55;color:#77746e;max-width:690px}}.brand{{font-size:19px;font-weight:750;white-space:nowrap}}
</style></head><body><main class="card">
<div class="eyebrow">JEV48 · VERIFIED BENCHMARK</div>
<h1>We got closer.<br>Jev still won.</h1>
<p class="lede">An AI agent started from an open 2B decision model, trained on public data, and evaluated the result on an independent benchmark.</p>
<section class="score"><div class="label">Independent teacher-agreement accuracy</div>
<div class="barrow"><span>TypeSafe Jev</span><div class="track"><div class="fill jev" style="width:{jev['accuracy']*100}%"></div></div><span class="val">{pct(jev['accuracy'])}</span></div>
<div class="barrow"><span>Jev48</span><div class="track"><div class="fill ours" style="width:{public_ours['accuracy']*100}%"></div></div><span class="val">{pct(public_ours['accuracy'])}</span></div>
<div class="gap">Jev leads by {gap:.1f} percentage points · aggregate, unpaired</div></section>
<section class="proof"><div><div class="label">What improved</div><div class="big">Brier ↓ {improvement:.4f}</div><div class="small">vs. the untouched open base on locked human-preference rows. 95% paired bootstrap CI: {ci['ci95'][0]:.4f} to {ci['ci95'][1]:.4f}.</div></div>
<div><div class="label">The run</div><div class="big">{minutes} min · ${run['estimated_gpu_list_cost_usd']:.2f}</div><div class="small">Pinned H200 run. Selected on development data only. No Jev outputs used for training or model selection.</div></div></section>
<footer class="footer"><div class="source">SOURCE: LOCALLLAMA/TYPED-DECISIONS @ {html.escape(public['benchmark_revision'][:10])} · JEV ROW SOURCE-PUBLISHED · SYNTHETIC TEACHER TARGETS · FULL RECEIPTS: GITHUB.COM/AGILABS-AI/JEV48</div><div class="brand">AGI LABS</div></footer>
</main></body></html>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(body, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
