"""Live Jev stage for an already completed Jev48 GPU experiment.

This is a separate Modal app on purpose: the main GPU experiment can run without
any TypeSafe/OpenRouter secret existing in the workspace.

Setup one provider locally (never paste the key into chat or source):
    modal secret create jev48-secrets TYPESAFE_API_KEY=...
# or
    modal secret create jev48-secrets OPENROUTER_API_KEY=...

Then:
    modal run modal_jev.py --run-name jev48-...
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

import modal

ROOT = Path(__file__).resolve().parent
app = modal.App("jev48-live")
artifacts = modal.Volume.from_name("jev48-artifacts", create_if_missing=True)
secret = modal.Secret.from_name("jev48-secrets")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy>=1.26,<2", "pydantic>=2.10", "httpx>=0.28", "pyyaml>=6")
    .add_local_dir(
        str(ROOT),
        remote_path="/workspace/jev48",
        ignore=["**/__pycache__/**", ".pytest_cache/**", "data/**", "results/**"],
    )
)


def run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def fresh_workdir() -> Path:
    src = Path("/workspace/jev48")
    dst = Path("/tmp/jev48")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


@app.function(
    image=image,
    timeout=2 * 60 * 60,
    secrets=[secret],
    volumes={"/artifacts": artifacts},
)
def live_jev(run_name: str) -> dict:
    artifacts.reload()
    source = Path("/artifacts") / run_name
    if not source.exists():
        raise ValueError(f"unknown artifact run {run_name}")
    work = fresh_workdir()
    shutil.copytree(source / "data", work / "data", dirs_exist_ok=True)
    shutil.copytree(source / "results", work / "results", dirs_exist_ok=True)

    run([
        "python", "scripts/benchmark_jev.py",
        "--data", "data/jev48_v1/final.jsonl",
        "--output", "results/final/jev.raw.jsonl",
        "--splits", "calibration,test,ood",
    ], work)
    run([
        "python", "scripts/calibrate_predictions.py",
        "--data", "data/jev48_v1/final.jsonl",
        "--predictions", "results/final/jev.raw.jsonl",
        "--output", "results/final/jev.calibrated.jsonl",
        "--fit-split", "calibration",
        "--eval-splits", "test,ood",
    ], work)
    run([
        "python", "scripts/final_report.py",
        "--base", "results/final/base.calibrated.summary.json",
        "--ours", "results/final/jev48.calibrated.summary.json",
        "--jev-native", "results/final/jev.raw.summary.json",
        "--jev-calibrated", "results/final/jev.calibrated.summary.json",
        "--selection", "results/selection.json",
        "--output", "results/final_report.md",
    ], work)
    run([
        "python", "scripts/paired_bootstrap.py",
        "--data", "data/jev48_v1/final.jsonl",
        "--a", "results/final/jev48.calibrated.jsonl",
        "--b", "results/final/jev.raw.jsonl",
        "--name-a", "Jev48", "--name-b", "Jev",
        "--output", "results/bootstrap-jev48-vs-jev.json",
    ], work)
    run([
        "python", "scripts/paired_bootstrap.py",
        "--data", "data/jev48_v1/final.jsonl",
        "--a", "results/final/jev48.calibrated.jsonl",
        "--b", "results/final/base.calibrated.jsonl",
        "--name-a", "Jev48", "--name-b", "decider-2b",
        "--output", "results/bootstrap-jev48-vs-base.json",
    ], work)
    run(["python", "scripts/render_site.py", "--report-json", "results/final_report.json", "--output", "site/index.html"], work)
    run(["python", "scripts/generate_launch_facts.py", "--report", "results/final_report.json", "--run-manifest", "results/run_manifest.json", "--output", "results/LAUNCH_FACTS.md"], work)
    run(["python", "scripts/release_audit.py", "--final-results", "."], work)

    shutil.copytree(work / "results", source / "results", dirs_exist_ok=True)
    shutil.copytree(work / "site", source / "site", dirs_exist_ok=True)
    manifest_path = source / "results/run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["jev_queried"] = True
    manifest["jev_completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts.commit()
    return {
        "run_name": run_name,
        "final_report": (work / "results/final_report.md").read_text(encoding="utf-8"),
        "jev_queried": True,
    }


@app.local_entrypoint()
def main(run_name: str):
    result = live_jev.remote(run_name)
    print(json.dumps(result, indent=2), flush=True)
    print("\\nDownload all receipts with:")
    print(f"modal volume get jev48-artifacts {run_name} ./jev48-modal-results")
