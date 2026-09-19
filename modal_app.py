"""One-command Modal runner for OpenJev.

OpenJev v0 intentionally starts from the public NanoJev checkpoint and changes
primarily the training distribution, not the architecture.

Usage:
    pip install 'modal>=1.1,<2'
    modal setup
    modal run modal_app.py

Artifacts are persisted to the Modal volume `openjev-artifacts`.
Model downloads are cached in `openjev-hf-cache`.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import time

import modal

ROOT = Path(__file__).resolve().parent
APP_NAME = "openjev"
NANOJEV_COMMIT = "71a513bb0163b5634467842b523ee0c0ed6fb1c7"
NANOJEV_MODEL = "C-Tianyu/NanoJev"

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("openjev-hf-cache", create_if_missing=True)
artifacts = modal.Volume.from_name("openjev-artifacts", create_if_missing=True)

# Match NanoJev's recorded dependency family as closely as practical. No H100 is required.
image = (
    modal.Image.debian_slim(python_version="3.14")
    .apt_install("git")
    .pip_install(
        "torch==2.14.0",
        "transformers==5.17.0",
        "safetensors==0.8.0",
        "numpy==2.5.3",
        "pydantic>=2.10",
        "pyyaml>=6.0",
        "httpx>=0.28",
        "huggingface_hub>=0.34",
        "datasets>=4.0",
        "sentencepiece>=0.2",
    )
    .env({
        "HF_HOME": "/cache/huggingface",
        "HF_HUB_CACHE": "/cache/huggingface/hub",
        "HF_XET_HIGH_PERFORMANCE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .add_local_dir(
        str(ROOT),
        remote_path="/workspace/openjev",
        ignore=[".git/**", "**/__pycache__/**", ".pytest_cache/**", "checkpoints/**"],
    )
)


def run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def fresh_workdir() -> Path:
    src = Path("/workspace/openjev")
    dst = Path("/tmp/openjev")
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


def download_nanojev_checkpoint(work: Path) -> Path:
    checkpoint = work / "checkpoints" / "nanojev-base"
    code = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id={NANOJEV_MODEL!r}, local_dir={str(checkpoint)!r}, "
        "allow_patterns=['best.safetensors','config.json','tokenizer/*','backbone_config/*'])"
    )
    run(["python", "-c", code], work)
    return checkpoint


def clone_nanojev(work: Path) -> Path:
    vendor = work / "vendor" / "NanoJev"
    vendor.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "https://github.com/TianyuCodings/NanoJev.git", str(vendor)], work)
    run(["git", "checkout", NANOJEV_COMMIT], vendor)
    return vendor


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60,
    volumes={"/cache": hf_cache, "/artifacts": artifacts},
)
def openjev_gpu() -> dict:
    """Fine-tune NanoJev into OpenJev and compare both on the frozen benchmark."""
    work = fresh_workdir()
    started = time.time()

    # 1) Build broader semantic data + known-probability simulator supervision.
    run([
        "python", "scripts/build_public_data.py",
        "--output", "data/public_decisions",
        "--benchmark-output", "data/benchmark/open_decision_bench.jsonl",
    ], work)
    run([
        "python", "scripts/merge_data_dirs.py",
        "--inputs", "data/simulator", "data/public_decisions",
        "--output", "data/combined",
    ], work)
    run(["python", "scripts/validate_dataset.py", "data/combined"], work)
    run(["python", "scripts/freeze_benchmark.py", "data/benchmark/open_decision_bench.jsonl"], work)
    run([
        "python", "scripts/export_nanojev_training_data.py",
        "--input-dir", "data/combined",
        "--output-dir", "data/openjev_training",
    ], work)

    # 2) Pin upstream source and public checkpoint.
    vendor = clone_nanojev(work)
    base = download_nanojev_checkpoint(work)

    # Validate our bridge using upstream NanoJev's own schema validator before GPU training.
    run([
        "python", str(vendor / "scripts" / "train_pipeline_decisions.py"),
        "--input", str(work / "data" / "openjev_training"),
        "--validate-only",
    ], vendor)

    # 3) Benchmark base NanoJev BEFORE OpenJev training on the frozen rows.
    run([
        "python", "scripts/benchmark_nanojev.py",
        "--data", "data/benchmark/open_decision_bench.jsonl",
        "--nanojev-repo", str(vendor),
        "--checkpoint-dir", str(base),
        "--output", "results/nanojev_open_decision.jsonl",
        "--batch-questions", "1",
    ], work)

    # 4) Minimal tweak: same architecture, short distribution-aware fine-tune.
    openjev_ckpt = work / "checkpoints" / "openjev"
    run([
        "python", str(vendor / "scripts" / "train_pipeline_decisions.py"),
        "--input", str(work / "data" / "openjev_training"),
        "--output-dir", str(openjev_ckpt),
        "--init-checkpoint", str(base),
        "--objective", "gold_distribution",
        "--loss", "brier",
        "--gradient-checkpointing",
        "--steps", "150",
        "--head-steps", "0",
        "--batch-questions", "1",
        "--microbatch-questions", "1",
        "--max-microbatch-tokens", "16384",
        "--eval-every", "25",
        "--max-length", "512",
        "--backbone-lr", "2e-5",
        "--head-lr", "2e-4",
        "--precision", "bf16",
        "--seed", "17",
    ], vendor)

    # 5) Benchmark OpenJev on exactly the same frozen rows.
    run([
        "python", "scripts/benchmark_nanojev.py",
        "--data", "data/benchmark/open_decision_bench.jsonl",
        "--nanojev-repo", str(vendor),
        "--checkpoint-dir", str(openjev_ckpt),
        "--output", "results/openjev_open_decision.jsonl",
        "--batch-questions", "1",
        "--model-label", "OpenJev",
    ], work)

    # 6) Report + predeclared launch gate.
    run([
        "python", "scripts/compare_models.py",
        "results/nanojev_open_decision.summary.json",
        "results/openjev_open_decision.summary.json",
        "--output", "results/openjev_vs_nanojev.md",
    ], work)
    run([
        "python", "scripts/check_launch_gate.py",
        "--ours", "results/openjev_open_decision.summary.json",
        "--nanojev", "results/nanojev_open_decision.summary.json",
        "--output", "results/openjev_gate.json",
    ], work)

    gate = json.loads((work / "results/openjev_gate.json").read_text())
    run_name = f"openjev-{int(started)}"
    persisted = persist(work, run_name, [
        "data/public_decisions/manifest.json",
        "data/benchmark/open_decision_bench.jsonl.manifest.json",
        "checkpoints/openjev",
        "results/nanojev_open_decision.jsonl",
        "results/nanojev_open_decision.summary.json",
        "results/openjev_open_decision.jsonl",
        "results/openjev_open_decision.summary.json",
        "results/openjev_vs_nanojev.md",
        "results/openjev_vs_nanojev.json",
        "results/openjev_gate.json",
    ])

    return {
        "run_name": run_name,
        "artifacts": persisted,
        "elapsed_seconds": time.time() - started,
        "launch_gate": gate,
        "comparison_markdown": (work / "results/openjev_vs_nanojev.md").read_text(),
        "disclosure": (
            "OpenJev v0 is derived from the public NanoJev checkpoint and pinned public source; "
            "the core architecture is intentionally unchanged."
        ),
    }


@app.local_entrypoint()
def main():
    result = openjev_gpu.remote()
    print(json.dumps(result, indent=2))
    print("\nDownload everything with:")
    print(f"modal volume get openjev-artifacts {result['run_name']} ./modal-results")
