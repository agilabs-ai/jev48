#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def fmt(x, pct=False):
    if x is None: return "—"
    return f"{100*x:.1f}%" if pct else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser(description="Render the measured Jev48 results into a shareable static page.")
    ap.add_argument("--report-json", default="results/final_report.json")
    ap.add_argument("--output", default="site/index.html")
    ap.add_argument("--public-comparison")
    args = ap.parse_args()
    data = json.loads(Path(args.report_json).read_text(encoding="utf-8"))
    rows = data["rows"]
    jev = next((r for r in rows if r["model"] == "Jev"), None)
    ours = next(r for r in rows if r["model"] == "Jev48 / ChatGPT build")
    base = next(r for r in rows if r["model"].startswith("decider"))
    selection = data["selection"]
    public = json.loads(Path(args.public_comparison).read_text(encoding="utf-8")) if args.public_comparison else None

    def cards(r):
        return f'''<div class="metrics">
          <div><b>{fmt(r['mtbench_accuracy'], True)}</b><span>human-pref accuracy</span></div>
          <div><b>{fmt(r['mtbench_brier'])}</b><span>human-pref Brier ↓</span></div>
          <div><b>{fmt(r['transfer_accuracy'], True)}</b><span>transfer accuracy</span></div>
          <div><b>{fmt(r['transfer_brier'])}</b><span>transfer Brier ↓</span></div>
        </div>'''

    jev_block = cards(jev) if jev else '<p class="pending">No paired live run. See the independent public benchmark below.</p>'
    rows_html = "".join(
        f"<tr><td>{html.escape(r['model'])}</td><td>{fmt(r['mtbench_accuracy'], True)}</td><td>{fmt(r['mtbench_brier'])}</td><td>{fmt(r['transfer_accuracy'], True)}</td><td>{fmt(r['transfer_brier'])}</td></tr>"
        for r in rows
    )
    public_html = ""
    if public:
        pm, pj = public["metrics"], public["jev_reference"]
        public_html = f'''<h3>Independent public Jev benchmark</h3><p class="pending">Aggregate, unpaired comparison on <code>LocalLLaMA/typed-decisions</code>. The maintainers published the Jev row; Jev48 ran separately, zero-shot, on the pinned test split.</p><table><thead><tr><th>System</th><th>Accuracy</th><th>Brier ↓</th><th>KL ↓</th><th>ECE ↓</th></tr></thead><tbody><tr><td>TypeSafe Jev 1.13.0 (published)</td><td>{fmt(pj['accuracy'], True)}</td><td>{fmt(pj['brier'])}</td><td>{fmt(pj['kl'])}</td><td>{fmt(pj['ece_15'])}</td></tr><tr><td>Jev48 (zero-shot)</td><td>{fmt(pm['accuracy'], True)}</td><td>{fmt(pm['brier'])}</td><td>{fmt(pm['kl'])}</td><td>{fmt(pm['ece_15'])}</td></tr></tbody></table><p class="pending">Jev leads accuracy and Brier. Jev48 has lower reported KL and ECE. This is not a paired run on Jev48's original rows.</p>'''
    body = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Jev48 — Jev vs ChatGPT in 48 hours</title><style>
:root{{--bg:#080808;--fg:#f6f6f3;--muted:#aaa;--line:#303030}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}}main{{max-width:1080px;margin:auto;padding:72px 28px 100px}}.eyebrow{{font:600 13px ui-monospace,SFMono-Regular,monospace;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}}h1{{font-size:clamp(48px,8vw,96px);line-height:.94;letter-spacing:-.055em;margin:24px 0 18px;max-width:900px}}.sub{{font-size:22px;color:#c8c8c2;max-width:760px;line-height:1.45}}.versus{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:64px 0}}.panel{{border:1px solid var(--line);border-radius:20px;padding:28px}}.panel h2{{font-size:30px;margin:0 0 4px}}.panel .tag{{color:var(--muted);margin-bottom:30px}}.metrics{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}.metrics div{{border-top:1px solid var(--line);padding:16px 0}}.metrics b{{display:block;font-size:29px;letter-spacing:-.04em}}.metrics span{{font-size:12px;color:var(--muted)}}h3{{margin-top:70px;font-size:30px}}table{{width:100%;border-collapse:collapse;font-size:15px}}th,td{{padding:15px 10px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}.audit{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.audit div{{border:1px solid var(--line);padding:18px;border-radius:14px}}code{{font-family:ui-monospace,SFMono-Regular,monospace;color:#ddd}}.pending{{color:var(--muted)}}footer{{margin-top:80px;color:#888;font-size:13px;line-height:1.5}}@media(max-width:720px){{.versus{{grid-template-columns:1fr}}.audit{{grid-template-columns:1fr}}}}
</style></head><body><main>
<div class="eyebrow">JEV48 / measured experiment</div><h1>Jev got years.<br>ChatGPT got a weekend.</h1>
<p class="sub">How close can an AI agent get to a newly launched decision model in 48 hours if it can use everything public on the internet? No “from scratch” claim. No hidden benchmark edits. Raw receipts included.</p>
<div class="versus"><section class="panel"><h2>Jev</h2><div class="tag">TypeSafe / closed reference</div>{jev_block}</section>
<section class="panel"><h2>ChatGPT build</h2><div class="tag">selected: {html.escape(selection['winner']['name'])}</div>{cards(ours)}</section></div>
<h3>Locked benchmark</h3><table><thead><tr><th>System</th><th>Human pref acc</th><th>Brier ↓</th><th>Transfer acc</th><th>Brier ↓</th></tr></thead><tbody>{rows_html}</tbody></table>
{public_html}
<h3>Integrity</h3><div class="audit"><div><b>Starting point disclosed</b><br><code>Mapika/decider-2b</code></div><div><b>Selection</b><br>dev only; locked test/OOD unseen</div><div><b>Jev leakage</b><br>0 outputs used for training</div></div>
<footer>Human preference: MT-Bench expert votes over real model outputs. Transfer: public tasks held out by the pinned open starting model. The open systems use a scalar calibration fit on frozen calibration rows. The independent Jev row comes from the cited public benchmark and is not a paired run on Jev48's locked rows. See the repository for hashes, provenance, exact commands and raw predictions.</footer>
</main></body></html>'''
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(body, encoding="utf-8")
    print(out)


if __name__ == "__main__": main()
