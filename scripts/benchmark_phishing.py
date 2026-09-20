#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request

import torch
from decider.infer import Decider

from jev48.benchmark_metrics import binary_metrics, wilson_interval
from jev48.io import dump_json


DATA_REVISION = "89afcc39610084298c4679159cb2e27d9ffffa46"
DATA_URL = f"https://huggingface.co/datasets/AreLit/PhishNChips/resolve/{DATA_REVISION}/core_emails.csv"
DATA_SHA256 = "cebb407ff8630491a97400e37464b8db8dfc4299164fca51fcb4ac7eec8204ef"
SOURCE_BENCHMARK = "https://github.com/anisselbd/jev-phishing-bench"
SOURCE_COMMIT = "1d56e8c64d029a9554a0874e2ef2901ed196e230"
QUESTIONS = {"verdict": {
    "type": "choice",
    "instructions": (
        "This email arrived in the user's inbox and contains a link (`link_url`). "
        "Is it a phishing attempt, or a legitimate message whose link is safe to click?"
    ),
    "criteria": {
        "phishing": "The email or its link is a phishing attempt or otherwise malicious. The user should not click the link.",
        "legitimate": "A normal, legitimate message. The link is safe to click.",
    },
}, "is_phishing": {"type": "noul", "instructions": "This email is a phishing attempt and the user should not click the link in `link_url`.", "criteria": {"true": "The email or its link is a phishing attempt or otherwise malicious.", "false": "The email is a normal, legitimate message and the link is safe to click."}},
"sig_domain_mismatch": {"type": "noul", "instructions": "The domain of the sender address in `from` is different from the organization or domain that the link in `link_url` points to."},
"sig_free_hosting": {"type": "noul", "instructions": "The link in `link_url` points to a URL shortener or to a free hosting or file-sharing platform (for example bit.ly, an IPFS gateway, Firebase, GitHub Pages, Google Docs) rather than to the organization's own website."},
"sig_lure": {"type": "noul", "instructions": "The email asks the user to sign in, verify an account, or open a document or file through the link."},
"sig_urgency": {"type": "noul", "instructions": "The email pressures the user to act quickly, mentions a deadline, or threatens a consequence."},
"sig_generic_sender": {"type": "noul", "instructions": "The sender uses a generic webmail address (such as gmail, outlook, yahoo, hotmail) in `from` while presenting themselves as a company, team, department or organization."},
"verdict_alt_click": {"type": "choice", "instructions": "Should the user click the link in this email?", "criteria": {"do_not_click": "No. The link may be phishing or malicious.", "click": "Yes. The email and its link look legitimate."}},
"verdict_alt_minimal": {"type": "choice", "instructions": "Classify this email.", "criteria": {"phishing": None, "legitimate": None}}}
EMAIL_FIELDS = ["sender", "from", "subject", "body", "link_display_text", "link_url"]


def load_rows() -> tuple[list[dict], str]:
    raw = urllib.request.urlopen(DATA_URL, timeout=120).read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != DATA_SHA256:
        raise ValueError(f"dataset checksum mismatch: {digest}")
    parsed = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    rows = []
    for row in parsed:
        email = json.loads(row["email_content"])
        rows.append({"id": row["id"], "y": int(row["phish_label"]), "email": {k: email[k] for k in EMAIL_FIELDS}})
    if len(rows) != 2000 or sum(r["y"] for r in rows) != 1000:
        raise ValueError("unexpected PhishNChips composition")
    return rows, digest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    for name in ("run-name", "model-sha256", "release-manifest-sha256", "repo-commit", "script-sha256"):
        ap.add_argument(f"--{name}", required=True)
    args = ap.parse_args()
    rows, digest = load_rows()
    if args.limit:
        rows = rows[:args.limit]
    model = Decider(args.model, device=args.device, dtype=torch.bfloat16, temperature=1.0, use_graphs=False)
    predictions, latencies = [], []
    for i, row in enumerate(rows, 1):
        started = time.perf_counter()
        answer = model.system_one(row["email"], QUESTIONS, independent=False)["answers"]["verdict"]
        latencies.append((time.perf_counter() - started) * 1000)
        probs = {k: float(v) for k, v in answer["probabilities"].items()}
        predictions.append({"id": row["id"], "y": row["y"], "phishing_probability": probs["phishing"], "probabilities": probs, "latency_ms": latencies[-1]})
        if i == 1 or i % 100 == 0 or i == len(rows):
            print(f"phishing {i}/{len(rows)} {latencies[-1]:.1f}ms", flush=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in predictions), encoding="utf-8")
    metrics = binary_metrics((x["y"] for x in predictions), (x["phishing_probability"] for x in predictions))
    successes = round(float(metrics["accuracy"]) * int(metrics["n"]))
    summary = {
        "benchmark": "PhishNChips v5.2 / jev-phishing-bench verdict",
        "source_benchmark": SOURCE_BENCHMARK,
        "source_benchmark_commit": SOURCE_COMMIT,
        "dataset_revision": DATA_REVISION,
        "dataset_url": DATA_URL,
        "dataset_sha256": digest,
        "mode": "exact published nine-question request and object state; no benchmark rows used in Jev48 fine-tuning or model selection",
        "model": {"path": args.model, "sha256": args.model_sha256, "release_manifest_sha256": args.release_manifest_sha256, "run_name": args.run_name, "repo_commit": args.repo_commit, "script_sha256": args.script_sha256},
        "metrics": {**metrics, "accuracy_wilson_ci95": wilson_interval(successes, int(metrics["n"]))},
        "mean_latency_ms": sum(latencies) / len(latencies),
        "jev_reference": {"model": "Jev 1.13.0", "n": 2000, "accuracy": 0.626, "accuracy_ci95": [0.6045711453178447, 0.6469457410468261], "recall": 0.432, "false_positive_rate": 0.180, "auroc": 0.6885385, "auroc_ci95": [0.6666893911680324, 0.7109252895727851], "ece_10": 0.15361, "evidence_path": "results/metrics.json", "evidence_sha256": "8beb3f727dd48adc5163c398e73fae639bcc6eec568cfa1e243331843b334e69", "comparison": "same dataset, object state, nine-question batch, and headline verdict wording; aggregate/unpaired because Jev row-level outputs are not public; no confidence interval for the cross-system difference"},
    }
    dump_json(out.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
