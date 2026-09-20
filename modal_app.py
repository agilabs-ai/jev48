"""Jev48: one-command Modal execution for the 48-hour Jev reproduction challenge.

The public story is a direct Jev challenge. Existing OSS is allowed as prior art:
we pin Mapika/decider as the strongest starting point found during the audit, then
run a controlled human-vote calibration experiment and a locked head-to-head.

Local setup:
    pip install 'modal>=1.5,<2'
    modal setup

Core experiment (no TypeSafe key required):
    modal run modal_app.py

The live Jev comparison is intentionally split into `modal_jev.py` so the GPU
experiment has zero dependency on an API secret.

Artifacts persist in Modal volume `jev48-artifacts`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from datetime import datetime, timezone

import modal

ROOT = Path(__file__).resolve().parent
APP_NAME = "jev48"
DECIDER_COMMIT = "b08acf787d5d1f718a8c36c4677960f43772c7be"
DECIDER_MODEL = "Mapika/decider-2b"
DECIDER_MODEL_REVISION = "4a0e86782adfdb7393e04b8ec9f6b939dca09273"
MODAL_GPU_REQUESTED = "H200"
MODAL_GPU_USD_PER_SECOND = 0.001261  # Modal public list price checked 2026-09-19
MODAL_PRICING_URL = "https://modal.com/pricing"

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("jev48-hf-cache", create_if_missing=True)
artifacts = modal.Volume.from_name("jev48-artifacts", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install("uv>=0.8")
    .run_commands(
        f"git clone https://github.com/Mapika/decider.git /opt/decider && cd /opt/decider && git checkout {DECIDER_COMMIT}",
        "cd /opt/decider && uv pip install --system -e '.[train,serve]'",
        "uv pip install --system 'pydantic>=2.10' 'pyyaml>=6' 'httpx>=0.28' 'pytest>=8' 'datasets>=3.6,<4'",
    )
    .env({
        "HF_HOME": "/cache/huggingface",
        "HF_HUB_CACHE": "/cache/huggingface/hub",
        # Keep datasets 3.6 metadata separate from caches written by newer
        # incompatible datasets majors during earlier/other Modal runs.
        "HF_DATASETS_CACHE": "/cache/huggingface/datasets-3.6",
        "HF_XET_HIGH_PERFORMANCE": "1",
        # The pinned upstream registry includes CogComp/trec, whose pinned
        # loader is a Hugging Face dataset script. datasets 3.6 requires this
        # explicit opt-in; datasets 4+ removed script support entirely.
        "HF_DATASETS_TRUST_REMOTE_CODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .add_local_dir(
        str(ROOT),
        remote_path="/workspace/jev48",
        ignore=["**/__pycache__/**", ".pytest_cache/**", "checkpoints/**", "data/**", "results/**"],
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


def persist(work: Path, run_name: str, paths: list[str]) -> str:
    target = Path("/artifacts") / run_name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    for rel in paths:
        source = work / rel
        if not source.exists():
            continue
        destination = target / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
    artifacts.commit()
    return str(target)


def build_data(work: Path) -> None:
    run(["python", "scripts/build_mtbench.py", "--out", "data/mtbench_human.jsonl", "--manifest", "data/mtbench_human.manifest.json"], work)
    run(["python", "scripts/build_decider_suites.py", "--out-dir", "data/decider_suites"], work)
    run(["python", "scripts/assemble_jev48.py", "--mtbench", "data/mtbench_human.jsonl", "--decider-dir", "data/decider_suites", "--out-dir", "data/jev48_v1"], work)
    run(["python", "scripts/freeze_benchmark.py", "data/jev48_v1/final.jsonl"], work)


def benchmark_dev(work: Path, model: str, label: str) -> Path:
    out = Path("results/dev") / f"{label}.jsonl"
    run([
        "python", "scripts/benchmark_decider.py",
        "--data", "data/jev48_v1/dev.jsonl",
        "--model", model,
        "--output", str(out),
        "--device", "cuda",
        "--temperature", "1.0",
        "--splits", "dev",
    ], work)
    return out


def train_trial(work: Path, base_model: str, name: str, mode: str, lr: float) -> tuple[Path, Path]:
    out = Path("runs") / name
    run([
        "python", "scripts/train_decider_soft.py",
        "--data", "data/jev48_v1/mtbench_train.jsonl", "data/jev48_v1/replay.jsonl",
        "--model", base_model,
        "--out", str(out),
        "--mode", mode,
        "--epochs", "2",
        "--lr", str(lr),
        "--brier-w", "0.2",
        "--warmup", "20",
        "--max-tokens", "16384",
        "--accum", "2",
        "--max-ctx", "16384",
        "--schema-first-prob", "0.5",
        "--preference-repeat", "3",
        "--seed", "48",
    ], work)
    pred = benchmark_dev(work, str(out / "model"), name)
    return out / "model", pred


def benchmark_final(work: Path, model: str, label: str) -> tuple[Path, Path]:
    raw = Path("results/final") / f"{label}.raw.jsonl"
    cal = Path("results/final") / f"{label}.calibrated.jsonl"
    run([
        "python", "scripts/benchmark_decider.py",
        "--data", "data/jev48_v1/final.jsonl",
        "--model", model,
        "--output", str(raw),
        "--device", "cuda",
        "--temperature", "1.0",
        "--splits", "calibration,test,ood",
    ], work)
    run([
        "python", "scripts/calibrate_predictions.py",
        "--data", "data/jev48_v1/final.jsonl",
        "--predictions", str(raw),
        "--output", str(cal),
        "--fit-split", "calibration",
        "--eval-splits", "test,ood",
    ], work)
    return raw, cal


@app.function(
    image=image,
    gpu=MODAL_GPU_REQUESTED,
    timeout=6 * 60 * 60,
    volumes={"/cache": hf_cache, "/artifacts": artifacts},
)
def experiment() -> dict:
    work = fresh_workdir()
    started = time.time()
    started_utc = datetime.now(timezone.utc).isoformat()
    repo_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=work, text=True).strip()
    upstream_sha = subprocess.check_output(["git", "-C", "/opt/decider", "rev-parse", "HEAD"], text=True).strip()
    if upstream_sha != DECIDER_COMMIT:
        raise RuntimeError(f"upstream pin mismatch: {upstream_sha} != {DECIDER_COMMIT}")
    from jev48.decider_bridge import resolve_model_snapshot
    base_model_path = resolve_model_snapshot(DECIDER_MODEL, DECIDER_MODEL_REVISION)
    run(["python", "-m", "pytest", "-q"], work)
    run(["python", "scripts/release_audit.py"], work)
    build_data(work)

    base_dev = benchmark_dev(work, base_model_path, "base")
    trials = [
        ("hard-lr1e-6", "hard", 1e-6),
        ("soft-lr3e-7", "soft", 3e-7),
        ("soft-lr1e-6", "soft", 1e-6),
        ("soft-lr3e-6", "soft", 3e-6),
    ]
    candidates = []
    for name, mode, lr in trials:
        model, pred = train_trial(work, base_model_path, name, mode, lr)
        candidates.append((name, model, pred))

    selection_path = work / "results" / "selection.json"
    cmd = [
        "python", "scripts/select_candidate.py",
        "--base", str(base_dev),
        "--base-model", DECIDER_MODEL,
        "--output", str(selection_path.relative_to(work)),
    ]
    for name, model, pred in candidates:
        cmd += ["--candidate", f"{name}|{pred}|{model}"]
    run(cmd, work)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    winner_model = selection["winner"]["model"]

    _, base_cal = benchmark_final(work, base_model_path, "base")
    if winner_model == DECIDER_MODEL:
        # Preserve a separate Jev48 receipt while making it explicit that the base survived selection.
        ours_raw = work / "results/final/jev48.raw.jsonl"
        ours_cal = work / "results/final/jev48.calibrated.jsonl"
        shutil.copy2(work / "results/final/base.raw.jsonl", ours_raw)
        shutil.copy2(work / "results/final/base.raw.summary.json", ours_raw.with_suffix(".summary.json"))
        shutil.copy2(work / "results/final/base.calibrated.jsonl", ours_cal)
        shutil.copy2(work / "results/final/base.calibrated.summary.json", ours_cal.with_suffix(".summary.json"))
    else:
        _, ours_cal_rel = benchmark_final(work, winner_model, "jev48")
        ours_cal = work / ours_cal_rel

    run([
        "python", "scripts/final_report.py",
        "--base", "results/final/base.calibrated.summary.json",
        "--ours", "results/final/jev48.calibrated.summary.json",
        "--selection", "results/selection.json",
        "--output", "results/final_report.md",
    ], work)
    run(["python", "scripts/render_site.py", "--report-json", "results/final_report.json", "--output", "site/index.html"], work)
    run(["python", "scripts/generate_launch_facts.py", "--report", "results/final_report.json", "--output", "results/LAUNCH_FACTS.md"], work)
    run(["python", "scripts/release_audit.py", "--final-results", "."], work)
    run(["python", "-m", "pip", "freeze"], work)
    env_text = subprocess.check_output(["python", "-m", "pip", "freeze"], cwd=work, text=True)
    (work / "results/environment.txt").write_text(env_text, encoding="utf-8")

    # Keep only the selected derivative weights. All candidate logs remain.
    release = work / "release"
    release.mkdir(exist_ok=True)
    if winner_model != DECIDER_MODEL:
        shutil.copytree(work / winner_model, release / "model")
        # Publish the exact temperature selected from the frozen calibration split.
        calibrated_summary = json.loads((work / "results/final/jev48.calibrated.summary.json").read_text(encoding="utf-8"))
        cfg_path = release / "model" / "decider_config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        cfg.update({
            "temperature": float(calibrated_summary["temperature"]),
            "version": "jev48-2b",
            "jev48_benchmark_sha256": calibrated_summary["benchmark_sha256"],
            "jev48_temperature_fit_split": calibrated_summary["fit_split"],
        })
        cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (release / "MODEL_SOURCE.json").write_text(json.dumps({
        "release_model_name": "jev48-2b" if winner_model != DECIDER_MODEL else None,
        "selected_name": selection["winner"]["name"],
        "selected_model": winner_model,
        "base_model": DECIDER_MODEL,
        "base_model_revision": DECIDER_MODEL_REVISION,
        "upstream_repo": "Mapika/decider",
        "upstream_commit": DECIDER_COMMIT,
    }, indent=2) + "\n", encoding="utf-8")
    if winner_model != DECIDER_MODEL:
        run([
            "python", "scripts/generate_model_card.py",
            "--selection", "results/selection.json",
            "--summary", "results/final/jev48.calibrated.summary.json",
            "--source", "release/MODEL_SOURCE.json",
            "--output", "release/model/README.md",
        ], work)
    for trial_dir in (work / "runs").glob("*"):
        model_dir = trial_dir / "model"
        if model_dir.exists():
            shutil.rmtree(model_dir)

    elapsed_seconds = time.time() - started
    run_manifest = {
        "repo_commit": repo_sha,
        "upstream_commit": upstream_sha,
        "base_model": DECIDER_MODEL,
        "base_model_revision": DECIDER_MODEL_REVISION,
        "started_utc": started_utc,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "modal_gpu_requested": MODAL_GPU_REQUESTED,
        "modal_gpu_list_price_usd_per_second": MODAL_GPU_USD_PER_SECOND,
        "modal_pricing_url": MODAL_PRICING_URL,
        "estimated_gpu_list_cost_usd": elapsed_seconds * MODAL_GPU_USD_PER_SECOND,
        "cost_note": "GPU list-price estimate only; excludes CPU/memory, region multipliers, credits, storage and API charges",
        "selection": selection,
        "gpu": subprocess.check_output(["python", "-c", "import torch; print(torch.cuda.get_device_name(0))"], text=True).strip(),
        "jev_queried": False,
    }
    (work / "results/run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Refresh facts now that exact runtime/GPU metadata exists.
    run(["python", "scripts/generate_launch_facts.py", "--report", "results/final_report.json", "--run-manifest", "results/run_manifest.json", "--output", "results/LAUNCH_FACTS.md"], work)

    run_name = f"jev48-{int(started)}"
    persisted = persist(work, run_name, [
        "data/mtbench_human.manifest.json",
        "data/decider_suites/manifest.json",
        "data/jev48_v1",
        "runs",
        "release",
        "results",
        "site",
    ])
    return {
        "run_name": run_name,
        "artifacts": persisted,
        "elapsed_seconds": time.time() - started,
        "selection": selection,
        "final_report": (work / "results/final_report.md").read_text(encoding="utf-8"),
        "base_source": {"repo": "Mapika/decider", "commit": DECIDER_COMMIT, "model": DECIDER_MODEL, "model_revision": DECIDER_MODEL_REVISION},
        "jev_queried": False,
    }



@app.local_entrypoint()
def main():
    result = experiment.remote()
    print(json.dumps(result, indent=2), flush=True)
    print("\nRun live Jev only after this selection is frozen:")
    print(f"modal run modal_jev.py --run-name {result['run_name']}")
    print("\nDownload all receipts with:")
    print(f"modal volume get jev48-artifacts {result['run_name']} ./jev48-modal-results")
