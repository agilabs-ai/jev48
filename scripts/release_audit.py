#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jev48.decider_bridge import DECIDER_COMMIT, DECIDER_MODEL, DECIDER_MODEL_REVISION
from jev48.mtbench import MTBENCH_REVISION
from jev48.typed_decisions import REVISION as TYPED_DECISIONS_REVISION
from scripts.verify_release_bundle import verify as verify_release_bundle


SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"TYPESAFE_API_KEY\s*=\s*(?!\.\.\.|YOUR_KEY|YOUR_)[^\s`]+"),
    re.compile(r"MODAL_TOKEN_SECRET\s*=\s*[^\s`]+"),
    re.compile(r"OPENROUTER_API_KEY\s*=\s*(?!\.\.\.|YOUR_KEY|YOUR_)[^\s`]+"),
]

IGNORED_SCAN_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache"}


def scan_secrets(root: Path):
    hits = []
    for path in root.rglob("*"):
        if not path.is_file() or any(part in IGNORED_SCAN_DIRS for part in path.parts):
            continue
        if path.stat().st_size > 5_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pat in SECRET_PATTERNS:
            if pat.search(text):
                hits.append(str(path.relative_to(root)))
                break
    return hits


def main():
    ap = argparse.ArgumentParser(description="Fail-closed Jev48 release audit.")
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--final-results", type=Path, help="Path to downloaded Modal run; enables final receipt checks")
    ap.add_argument("--release-bundle", type=Path)
    ap.add_argument("--release-manifest", type=Path)
    args = ap.parse_args()
    root = args.root.resolve()
    errors = []
    secret_hits = scan_secrets(root)
    if secret_hits:
        errors.append(f"possible secrets found in: {secret_hits}")
    if (root / ".env").exists():
        errors.append(".env exists in release root")

    # Pins must remain explicit and stable.
    pins = {
        "decider_commit": DECIDER_COMMIT,
        "decider_model": DECIDER_MODEL,
        "decider_model_revision": DECIDER_MODEL_REVISION,
        "mtbench_revision": MTBENCH_REVISION,
    }
    if len(DECIDER_COMMIT) != 40 or len(DECIDER_MODEL_REVISION) != 40 or len(MTBENCH_REVISION) != 40:
        errors.append("upstream revisions are not immutable 40-char commits")

    final = None
    if args.final_results:
        final = args.final_results.resolve()
        required = [
            final / "data/jev48_v1/manifest.json",
            final / "data/jev48_v1/final.jsonl.manifest.json",
            final / "results/selection.json",
            final / "results/final/base.calibrated.summary.json",
            final / "results/final/jev48.calibrated.summary.json",
            final / "results/final_report.json",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            errors.append(f"missing final receipts: {missing}")
        else:
            sel = json.loads((final / "results/selection.json").read_text())
            if sel.get("selection_uses_locked_test_or_ood") is not False:
                errors.append("selection receipt does not affirm locked test/OOD exclusion")
            manifest = json.loads((final / "data/jev48_v1/final.jsonl.manifest.json").read_text())
            report = json.loads((final / "results/final_report.json").read_text())
            if report.get("benchmark_sha256") != manifest.get("sha256"):
                errors.append("final report hash does not match frozen benchmark manifest")
            public_summary = final / "results/public/typed-decisions.summary.json"
            public_predictions = final / "results/public/typed-decisions.jsonl"
            if public_summary.exists() or public_predictions.exists():
                if not public_summary.exists() or not public_predictions.exists():
                    errors.append("public comparison requires both predictions and summary")
                else:
                    public = json.loads(public_summary.read_text())
                    rows = [json.loads(line) for line in public_predictions.read_text().splitlines() if line]
                    if public.get("benchmark_revision") != TYPED_DECISIONS_REVISION:
                        errors.append("public comparison benchmark revision is not pinned")
                    if public.get("metrics", {}).get("n_decisions") != 2000 or len(rows) != 400:
                        errors.append("public comparison is incomplete")
                    if len({row.get("id") for row in rows}) != len(rows):
                        errors.append("public comparison contains duplicate case IDs")

    if bool(args.release_bundle) != bool(args.release_manifest):
        errors.append("release bundle and manifest must be supplied together")
    elif args.release_bundle:
        try:
            verify_release_bundle(args.release_bundle, args.release_manifest)
        except Exception as exc:
            errors.append(f"release bundle verification failed: {exc}")

    result = {"ok": not errors, "pins": pins, "secret_hits": secret_hits, "errors": errors}
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
