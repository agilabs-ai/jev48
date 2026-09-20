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
        public_html = f'''<h3>Independent public Jev benchmark</h3><p class="pending">Aggregate, unpaired comparison on <code>LocalLLaMA/typed-decisions</code>. The target is the mean of three samples from a separate teacher model, so this measures teacher agreement—not real-world correctness.</p><table><thead><tr><th>System</th><th>Accuracy</th></tr></thead><tbody><tr><td>TypeSafe Jev 1.13.0 (source-published)</td><td>{fmt(pj['accuracy'], True)}</td></tr><tr><td>Jev48 (zero-shot)</td><td>{fmt(pm['accuracy'], True)}</td></tr></tbody></table><p class="pending">Jev leads by {(pj['accuracy']-pm['accuracy'])*100:.1f} points. No paired Jev predictions or significance test are available; the upstream scorer for distribution metrics was not published.</p>'''
    body = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Jev48 — Jev vs ChatGPT in 48 hours</title><style>
:root{{--bg:#080808;--fg:#f6f6f3;--muted:#aaa;--line:#303030}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}}main{{max-width:1080px;margin:auto;padding:72px 28px 100px}}.eyebrow{{font:600 13px ui-monospace,SFMono-Regular,monospace;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}}h1{{font-size:clamp(48px,8vw,96px);line-height:.94;letter-spacing:-.055em;margin:24px 0 18px;max-width:900px}}.sub{{font-size:22px;color:#c8c8c2;max-width:760px;line-height:1.45}}.versus{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:64px 0}}.panel{{border:1px solid var(--line);border-radius:20px;padding:28px}}.panel h2{{font-size:30px;margin:0 0 4px}}.panel .tag{{color:var(--muted);margin-bottom:30px}}.metrics{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}.metrics div{{border-top:1px solid var(--line);padding:16px 0}}.metrics b{{display:block;font-size:29px;letter-spacing:-.04em}}.metrics span{{font-size:12px;color:var(--muted)}}h3{{margin-top:70px;font-size:30px}}table{{width:100%;border-collapse:collapse;font-size:15px}}th,td{{padding:15px 10px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}.audit{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.audit div{{border:1px solid var(--line);padding:18px;border-radius:14px}}code{{font-family:ui-monospace,SFMono-Regular,monospace;color:#ddd}}.pending{{color:var(--muted)}}footer{{margin-top:80px;color:#888;font-size:13px;line-height:1.5}}@media(max-width:720px){{.versus{{grid-template-columns:1fr}}.audit{{grid-template-columns:1fr}}}}
</style></head><body><main>
<div class="eyebrow">JEV48 / 48-HOUR PUBLIC-WEB EXPERIMENT</div><h1>Jev launched.<br>An AI agent got 48 hours.</h1>
<p class="sub">Starting from the public Mapika/decider-2b checkpoint, the agent selected a derivative and finished 15.0 accuracy points behind Jev on an independent synthetic teacher-agreement benchmark. The successful H200 run lasted 37 minutes; 48 hours was the agent challenge's maximum window, not elapsed GPU time.</p>
<p><a href="https://github.com/agilabs-ai/jev48">Source</a> · <a href="https://huggingface.co/agilabs-ai/jev48-2b">Model</a> · <a href="https://github.com/agilabs-ai/jev48/blob/main/RESULTS.md">Results</a> · <a href="https://github.com/agilabs-ai/jev48/tree/main/receipts">Receipts</a> · <a href="https://github.com/agilabs-ai/jev48/blob/main/PROJECT_SPEC.md">Methodology</a> · <a href="https://github.com/agilabs-ai/jev48/blob/main/README.md#run-it">Reproduce</a> · <a href="https://github.com/Mapika/decider/tree/b08acf787d5d1f718a8c36c4677960f43772c7be">Pinned upstream code</a> · <a href="https://huggingface.co/Mapika/decider-2b/tree/4a0e86782adfdb7393e04b8ec9f6b939dca09273">Pinned base weights</a> · <a href="https://github.com/agilabs-ai/jev48/blob/main/receipts/upstream/typed-decisions-README.md">Pinned benchmark evidence</a> · <a href="https://github.com/agilabs-ai/jev48/blob/main/THIRD_PARTY_NOTICES.md">Licenses</a> · <a href="https://github.com/agilabs-ai/jev48/blob/main/CITATION.cff">Citation</a></p>
<div class="versus"><section class="panel"><h2>Jev</h2><div class="tag">TypeSafe / closed reference</div>{jev_block}</section>
<section class="panel"><h2>ChatGPT build</h2><div class="tag">selected: {html.escape(selection['winner']['name'])}</div>{cards(ours)}</section></div>
<h3>Locked benchmark</h3><table><thead><tr><th>System</th><th>Human pref acc</th><th>Brier ↓</th><th>Transfer acc</th><th>Brier ↓</th></tr></thead><tbody>{rows_html}</tbody></table>
{public_html}
<h3>Integrity</h3><div class="audit"><div><b>Starting point disclosed</b><br><code>Mapika/decider-2b</code></div><div><b>Selection</b><br>dev only; locked test/OOD unseen</div><div><b>Jev leakage</b><br>0 outputs used for training</div></div>
<footer>Human preference: MT-Bench expert votes over real model outputs. Transfer: public tasks held out by the pinned open starting model. A pooled scalar calibration is fit on frozen calibration rows. The independent Jev row is unpaired. Hashed aggregate receipts are published; mixed-source row-level data is withheld pending a per-source license audit.</footer>
</main></body></html>'''
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(body, encoding="utf-8")
    print(out)


if __name__ == "__main__": main()
