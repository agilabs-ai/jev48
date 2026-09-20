#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import re
import sys
import tarfile
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jev48.decider_bridge import DECIDER_COMMIT, DECIDER_MODEL, DECIDER_MODEL_REVISION
from jev48.mtbench import MTBENCH_REVISION
from jev48.typed_decisions import REVISION as TYPED_DECISIONS_REVISION
from jev48.benchmark_metrics import binary_metrics, paired_cluster_bootstrap_delta
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


def sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_public_run(name: str, summary: dict, rows: list[dict], expected_count: int,
                        id_fn, expected_model: str, manifest_sha: str, commit: str,
                        script_digest: str) -> list[str]:
    """Fail closed on headline metrics and immutable run provenance."""
    errors = []
    ids = [id_fn(row) for row in rows]
    if len(rows) != expected_count or len(set(ids)) != expected_count:
        errors.append(f"{name} predictions are incomplete or contain duplicate IDs")
    provenance = summary.get("model", {})
    expected = (expected_model, manifest_sha, "jev48-1789867911", commit, script_digest)
    actual = (provenance.get("sha256"), provenance.get("release_manifest_sha256"),
              provenance.get("run_name"), provenance.get("repo_commit"), provenance.get("script_sha256"))
    if actual != expected:
        errors.append(f"{name} immutable run provenance mismatch")
    if rows:
        accuracy = sum(bool(row.get("correct")) for row in rows) / len(rows)
        if abs(accuracy - float(summary.get("metrics", {}).get("jev48_accuracy", -1))) > 1e-12:
            errors.append(f"{name} headline accuracy mismatch")
    return errors


def audit_additional_benchmarks(final: Path, root: Path, errors: list[str]) -> None:
    expected_model = "20948bb0163f7230d3922e292e919a62cf6c0e0c7600718ab0d152b693227aec"
    manifest_path = final / "release/MANIFEST.json"
    manifest_sha = sha256(manifest_path)
    specs = {
        "phishing": (2000, root / "scripts/benchmark_phishing.py", "2f3ba4403672443ab407414119e7cc83c9845933", "579b3b34ebf94b962360e87de78b8f3c93955abb00891a535659ba54d63b9adc"),
        "jevbench": (231, root / "scripts/benchmark_jevbench_public.py", "c85fa24c8e6aece2dc5bfb1badc9ca3f844f7acf", "48a8fcf0bf840fc0ba8e5afb8570dd50a6d866f164a0e61e9da0e9f05249050c"),
    }
    loaded = {}
    for name, (count, script, commit, script_digest) in specs.items():
        summary_path = final / f"results/public/{name}.summary.json"
        predictions_path = final / f"results/public/{name}.jsonl"
        if not summary_path.exists() or not predictions_path.exists():
            errors.append(f"missing {name} summary or predictions")
            continue
        summary = json.loads(summary_path.read_text())
        rows = [json.loads(line) for line in predictions_path.read_text().splitlines() if line]
        loaded[name] = (summary, rows)
        if len(rows) != count or len({row.get("id") for row in rows}) != count:
            errors.append(f"{name} predictions are incomplete or contain duplicate IDs")
        provenance = summary.get("model", {})
        if provenance.get("sha256") != expected_model or provenance.get("release_manifest_sha256") != manifest_sha:
            errors.append(f"{name} model/release provenance mismatch")
        if provenance.get("run_name") != "jev48-1789867911" or provenance.get("repo_commit") != commit:
            errors.append(f"{name} run/code provenance is incomplete")
        if provenance.get("script_sha256") != script_digest or sha256(script) != script_digest:
            errors.append(f"{name} script hash does not match the benchmark receipt")
    if "phishing" in loaded:
        summary, rows = loaded["phishing"]
        raw = urlopen(summary["dataset_url"], timeout=120).read()
        if __import__("hashlib").sha256(raw).hexdigest() != "cebb407ff8630491a97400e37464b8db8dfc4299164fca51fcb4ac7eec8204ef":
            errors.append("phishing source download hash mismatch")
            expected = {}
        else:
            expected = {r["id"]: int(r["phish_label"]) for r in csv.DictReader(io.StringIO(raw.decode("utf-8")))}
        if any(expected.get(r["id"]) != r["y"] for r in rows) or set(expected) != {r["id"] for r in rows}:
            errors.append("phishing prediction IDs/labels do not match pinned source")
        got = binary_metrics((r["y"] for r in rows), (r["phishing_probability"] for r in rows))
        for key in ("accuracy", "recall", "false_positive_rate", "auroc", "ece_10", "brier"):
            if abs(float(got[key]) - float(summary["metrics"][key])) > 1e-12:
                errors.append(f"phishing metric mismatch: {key}")
        if summary.get("dataset_revision") != "89afcc39610084298c4679159cb2e27d9ffffa46" or summary.get("dataset_sha256") != "cebb407ff8630491a97400e37464b8db8dfc4299164fca51fcb4ac7eec8204ef":
            errors.append("phishing dataset pin/hash mismatch")
        if summary.get("jev_reference", {}).get("evidence_sha256") != "8beb3f727dd48adc5163c398e73fae639bcc6eec568cfa1e243331843b334e69":
            errors.append("phishing Jev reference evidence mismatch")
    if "jevbench" in loaded:
        summary, rows = loaded["jevbench"]
        revision = "e105a48f8cdb7f3babb3594424f73e5d7bdc97b9"
        expected_hashes = {
            "datasets/public/easy.jsonl": "231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b",
            "datasets/public/original.jsonl": "5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180",
            "datasets/public/hard.jsonl": "89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb",
            "results/v1.2/jevbench-v1.2-per-task.json": "0a8146ea994807b943b663edb34c0fdc05d35c702f9dfc792006f0edf30cbc7b",
        }
        if summary.get("file_sha256") != expected_hashes:
            errors.append("JevBench source hash map mismatch")
        tasks = {}
        evidence = None
        for path, digest in expected_hashes.items():
            raw = urlopen(f"https://raw.githubusercontent.com/fstandhartinger/jevbench/{revision}/{path}", timeout=120).read()
            if __import__("hashlib").sha256(raw).hexdigest() != digest:
                errors.append(f"JevBench source download hash mismatch: {path}")
                continue
            if path.endswith(".jsonl"):
                tier = Path(path).stem
                for line in raw.decode().splitlines():
                    item = json.loads(line)
                    tasks[item["id"]] = (item["expected"], item["family"], tier)
            else:
                evidence = json.loads(raw)
        jev_outcomes = dict(evidence["systems"]["jev-1.13.0"]["public_tasks"]) if evidence else {}
        if set(tasks) != {r["id"] for r in rows} or set(jev_outcomes) != set(tasks):
            errors.append("JevBench prediction/source ID mismatch")
        for r in rows:
            source = tasks.get(r["id"])
            outcome = jev_outcomes.get(r["id"])
            if not source or str(source[0]) != str(r["expected"]) or source[1:] != (r["family"], r["tier"]) or not outcome or (outcome[0] == "c") != r["jev_correct"] or outcome[0] != r["jev_outcome"]:
                errors.append(f"JevBench row provenance mismatch: {r['id']}")
                break
        ours = sum(bool(r["correct"]) for r in rows) / len(rows)
        jev = sum(bool(r["jev_correct"]) for r in rows) / len(rows)
        if abs(ours - summary["metrics"]["jev48_accuracy"]) > 1e-12 or abs(jev - summary["metrics"]["jev_accuracy"]) > 1e-12:
            errors.append("JevBench aggregate accuracy mismatch")
        delta = paired_cluster_bootstrap_delta((r["correct"] for r in rows), (r["jev_correct"] for r in rows), (r["family"] for r in rows))
        if delta != summary["metrics"]["paired_delta"]:
            errors.append("JevBench paired cluster bootstrap mismatch")
        if summary.get("benchmark_revision") != "e105a48f8cdb7f3babb3594424f73e5d7bdc97b9":
            errors.append("JevBench revision mismatch")

    newer = {
        "btzsc": (300, lambda r: r.get("id"), "9c242f5ad77300514d52784dd23f37d9211cd6d4", "8166ce95f49a4a81dfca7029583574230ba142c9cbac99110360ae4ae7086bef"),
        "code-review": (480, lambda r: (r.get("case_id"), r.get("rule")), "9c242f5ad77300514d52784dd23f37d9211cd6d4", "d3834231b8a00c440c9b1fa09a08ecbe8029ed08e2f73c3d1ca3098d9bd555a3"),
        "clash": (1289, lambda r: r.get("sample_idx"), "b600a6410569277cf8eac25c143058f00dd8c3f8", "ed0712870ae4af226a0fb0b6a7b560fe65ee326fea770465e1c84f7785505377"),
    }
    for name, (count, id_fn, commit, script_digest) in newer.items():
        sp, rp = final / f"results/public/{name}.summary.json", final / f"results/public/{name}.jsonl"
        if not sp.exists() or not rp.exists():
            errors.append(f"missing {name} summary or predictions")
            continue
        summary = json.loads(sp.read_text()); rows = [json.loads(x) for x in rp.read_text().splitlines() if x]
        errors.extend(validate_public_run(name, summary, rows, count, id_fn, expected_model, manifest_sha, commit, script_digest))
        if name == "btzsc":
            url = f"https://raw.githubusercontent.com/AbdelStark/jev-benchmarks/{summary.get('benchmark_revision')}/results/reports/btzsc-pilot-v1.json"
            raw = urlopen(url, timeout=120).read()
            if __import__("hashlib").sha256(raw).hexdigest() != "292db65ade535b9cd9c06477b6f7863e03a6106ec0a5fa56378a08164ee14598": errors.append("BTZSC reference report hash mismatch")
            reference = json.loads(raw)["results"]["jev"]
            for dataset, metrics in summary["metrics"]["by_dataset"].items():
                if abs(metrics["jev_accuracy"] - reference[dataset]["accuracy"]) > 1e-12: errors.append(f"BTZSC Jev reference mismatch: {dataset}")
            if summary.get("dataset_revision") != "fef2a2ac62b69c58670047dddf045c53d7c3cb5e": errors.append("BTZSC dataset revision mismatch")
        elif name == "code-review":
            raw = urlopen(f"https://raw.githubusercontent.com/gemanor/jev-code-review-benchmark/{summary.get('benchmark_revision')}/docs/results/decisions.csv", timeout=120).read()
            if __import__("hashlib").sha256(raw).hexdigest() != summary.get("reference_csv_sha256"): errors.append("code-review reference CSV hash mismatch")
            ref = [r for r in csv.DictReader(io.StringIO(raw.decode())) if r["model"] == "jev" and r["condition"] == "primary"]
            if abs(sum(r["correct"] == "True" for r in ref)/len(ref) - summary["metrics"]["jev_accuracy"]) > 1e-12: errors.append("code-review Jev reference mismatch")
        elif name == "clash":
            raw = urlopen(summary["source_url"], timeout=120).read()
            if __import__("hashlib").sha256(raw).hexdigest() != summary.get("source_sha256"): errors.append("CLASH source download hash mismatch")
            samples = json.loads(raw)["samples"]
            if len(samples) != len(rows) or any(rows[i]["image_id"] != sample["image_id"] for i, sample in enumerate(samples)): errors.append("CLASH rows do not match pinned source order")


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
            audit_additional_benchmarks(final, root, errors)

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
