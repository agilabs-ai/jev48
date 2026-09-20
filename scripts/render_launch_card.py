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
    parser.add_argument("--variant", choices=("scoreboard", "gap", "calibration", "ranking"), default="scoreboard")
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
    if args.variant == "gap":
        body = f"""<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box}}html,body{{margin:0;width:1080px;height:1350px;overflow:hidden}}body{{background:#eee9dd;color:#111;font-family:Inter,Arial,sans-serif}}main{{height:100%;padding:70px 72px;display:flex;flex-direction:column;background:linear-gradient(130deg,rgba(243,139,50,.17),transparent 37%)}}.mono{{font:700 17px ui-monospace,SFMono-Regular,monospace;letter-spacing:.13em;text-transform:uppercase}}.orange{{color:#e66800}}h1{{font-size:174px;line-height:.82;letter-spacing:-.085em;margin:76px 0 24px}}h2{{font-size:57px;line-height:.98;letter-spacing:-.05em;margin:0;max-width:820px}}.rule{{border-top:2px solid #171717;margin:74px 0 36px}}.rows{{display:grid;gap:20px}}.row{{display:grid;grid-template-columns:1fr auto;align-items:end;border-bottom:1px solid #aaa399;padding-bottom:18px}}.name{{font-size:28px}}.num{{font-size:62px;font-weight:800;letter-spacing:-.05em}}.note{{font:16px/1.55 ui-monospace,SFMono-Regular,monospace;color:#625f59;margin-top:28px;max-width:830px}}footer{{margin-top:auto;border-top:1px solid #aaa399;padding-top:24px;display:flex;justify-content:space-between;align-items:flex-end}}.source{{font:12px/1.5 ui-monospace,SFMono-Regular,monospace;max-width:720px;color:#6f6a62;text-transform:uppercase;letter-spacing:.1em}}.brand{{font-size:20px;font-weight:800}}
</style></head><body><main><div class="mono">JEV48 · THE HONEST RESULT</div><h1 class="orange">{gap:.1f}</h1><h2>percentage points short of Jev.</h2><div class="rule"></div><section class="rows"><div class="row"><span class="name">TypeSafe Jev</span><span class="num">{pct(jev['accuracy'])}</span></div><div class="row"><span class="name">Jev48</span><span class="num">{pct(public_ours['accuracy'])}</span></div></section><p class="note">Independent synthetic teacher-agreement benchmark · aggregate, unpaired comparison · selected on development data only · {minutes}-minute pinned H200 run · ${run['estimated_gpu_list_cost_usd']:.2f} estimated GPU list cost.</p><footer><div class="source">LOCALLLAMA/TYPED-DECISIONS @ {html.escape(public['benchmark_revision'][:10])} · FULL RECEIPTS: GITHUB.COM/AGILABS-AI/JEV48</div><div class="brand">AGI LABS</div></footer></main></body></html>"""
    elif args.variant == "calibration":
        body = f"""<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box}}html,body{{margin:0;width:1080px;height:1350px;overflow:hidden}}body{{background:#0b0b0a;color:#f3f0e8;font-family:Inter,Arial,sans-serif}}main{{height:100%;padding:72px;display:flex;flex-direction:column;background:radial-gradient(circle at 10% 95%,#23180e,transparent 34%)}}.mono{{font:700 17px ui-monospace,SFMono-Regular,monospace;letter-spacing:.13em;text-transform:uppercase;color:#f38b32}}h1{{font-size:90px;line-height:.92;letter-spacing:-.06em;margin:45px 0 24px}}.lede{{font-size:26px;line-height:1.45;color:#b8b4ac;max-width:850px}}.delta{{font-size:144px;font-weight:850;letter-spacing:-.08em;color:#f38b32;margin:55px 0 0}}.metric{{font:18px ui-monospace,SFMono-Regular,monospace;text-transform:uppercase;letter-spacing:.1em;color:#aaa69e}}.compare{{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:28px;margin-top:48px}}.box{{border-top:1px solid #4a4741;padding-top:22px}}.box b{{display:block;font-size:54px;letter-spacing:-.05em}}.box span{{font-size:18px;color:#aaa69e}}.arrow{{font-size:40px;color:#f38b32}}.ci{{margin-top:38px;padding:22px;border:1px solid #383632;border-radius:14px;font:17px/1.5 ui-monospace,SFMono-Regular,monospace;color:#bbb7af}}.reality{{margin-top:36px;border-left:4px solid #f38b32;padding-left:20px;font-size:24px;line-height:1.4}}footer{{margin-top:auto;border-top:1px solid #35332f;padding-top:24px;display:flex;justify-content:space-between;align-items:flex-end}}.source{{font:12px/1.5 ui-monospace,SFMono-Regular,monospace;max-width:720px;color:#77736b;text-transform:uppercase;letter-spacing:.1em}}.brand{{font-size:20px;font-weight:800}}
</style></head><body><main><div class="mono">JEV48 · LOCKED PREFERENCE TEST</div><h1>The win was<br>calibration.</h1><p class="lede">Training on distributions of human votes improved probability quality against the untouched open base.</p><div class="delta">↓ {improvement:.4f}</div><div class="metric">Brier score improvement</div><div class="compare"><div class="box"><b>{base['mtbench_brier']:.4f}</b><span>Open base</span></div><div class="arrow">→</div><div class="box"><b>{ours['mtbench_brier']:.4f}</b><span>Jev48</span></div></div><div class="ci">95% paired bootstrap CI for Jev48 − base: {ci['ci95'][0]:.4f} to {ci['ci95'][1]:.4f}. Negative is better.</div><div class="reality">On the independent public benchmark, Jev still led accuracy {pct(jev['accuracy'])} to {pct(public_ours['accuracy'])}.</div><footer><div class="source">PINNED RUN {html.escape(run['repo_commit'][:10])} · NO JEV OUTPUTS USED FOR TRAINING · FULL RECEIPTS: GITHUB.COM/AGILABS-AI/JEV48</div><div class="brand">AGI LABS</div></footer></main></body></html>"""
    elif args.variant == "ranking":
        phishing = read(args.receipts / "results/phishing.summary.json")
        pm, pj = phishing["metrics"], phishing["jev_reference"]
        auroc_lead = pm["auroc"] - pj["auroc"]
        accuracy_gap = pj["accuracy"] - pm["accuracy"]
        body = f"""<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box}}html,body{{margin:0;width:1080px;height:1350px;overflow:hidden}}body{{background:#0a0a09;color:#f4f1e9;font-family:Inter,Arial,sans-serif}}main{{height:100%;padding:72px;display:flex;flex-direction:column;background:radial-gradient(circle at 88% 12%,#302010,transparent 32%)}}.mono{{font:700 17px ui-monospace,SFMono-Regular,monospace;letter-spacing:.13em;text-transform:uppercase;color:#f38b32}}h1{{font-size:84px;line-height:.94;letter-spacing:-.058em;margin:42px 0 22px;max-width:900px}}.lede{{font-size:25px;line-height:1.42;color:#bbb7af;max-width:850px}}.hero{{font-size:150px;font-weight:850;letter-spacing:-.085em;color:#f38b32;margin:45px 0 0}}.metric{{font:18px ui-monospace,SFMono-Regular,monospace;text-transform:uppercase;letter-spacing:.1em;color:#aaa69e}}.rows{{margin-top:42px;border-top:1px solid #413e38}}.row{{display:grid;grid-template-columns:1fr 180px 180px;gap:20px;padding:22px 0;border-bottom:1px solid #302e2a;align-items:center}}.head{{font:14px ui-monospace,SFMono-Regular,monospace;color:#8e8a82;text-transform:uppercase;letter-spacing:.08em}}.name{{font-size:24px}}.num{{font-size:31px;font-weight:750;text-align:right}}.warn{{margin-top:32px;border-left:4px solid #f38b32;padding:2px 0 2px 20px;font-size:22px;line-height:1.42;color:#d2cec5}}footer{{margin-top:auto;border-top:1px solid #35332f;padding-top:24px;display:flex;justify-content:space-between;align-items:flex-end}}.source{{font:12px/1.5 ui-monospace,SFMono-Regular,monospace;max-width:730px;color:#77736b;text-transform:uppercase;letter-spacing:.1em}}.brand{{font-size:20px;font-weight:800}}
</style></head><body><main><div class="mono">JEV48 · 2,000-EMAIL PHISHING BENCHMARK</div><h1>Higher AUROC.<br>Worse decisions.</h1><p class="lede">Same public benchmark: Jev48 {pm['auroc']:.3f} vs source-published Jev {pj['auroc']:.3f}. Aggregate and unpaired; no difference interval.</p><div class="hero">+{auroc_lead:.3f}</div><div class="metric">Descriptive AUROC lead for Jev48</div><section class="rows"><div class="row head"><span>Metric</span><span>TypeSafe Jev</span><span>Jev48</span></div><div class="row"><span class="name">AUROC</span><span class="num">{pj['auroc']:.3f}</span><span class="num">{pm['auroc']:.3f}</span></div><div class="row"><span class="name">Accuracy @ 0.5</span><span class="num">{pct(pj['accuracy'])}</span><span class="num">{pct(pm['accuracy'])}</span></div></section><div class="warn">The accuracy gap is {accuracy_gap*100:.1f} points against Jev48. Ranking quality is not calibrated decision quality; this is not a parity claim.</div><footer><div class="source">PHISHNCHIPS V5.2 · EXACT NINE-QUESTION REQUEST · NO FINE-TUNING OR SELECTION ON THESE ROWS · AGGREGATE JEV REFERENCE · FULL RECEIPTS: GITHUB.COM/AGILABS-AI/JEV48</div><div class="brand">AGI LABS</div></footer></main></body></html>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(body, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
