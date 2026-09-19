"""One-command Modal runner for the first real de-risk experiment.

Usage (from repo root):
    pip install 'modal>=1.1'
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

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("openjev-hf-cache", create_if_missing=True)
artifacts = modal.Volume.from_name("openjev-artifacts", create_if_missing=True)

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
        "peft>=0.17",
        "sentencepiece>=0.2",
    )
    .env({
        "HF_HOME": "/cache/huggingface",
        "HF_HUB_CACHE": "/cache/huggingface/hub",
        "HF_XET_HIGH_PERFORMANCE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .add_local_dir(str(ROOT), remote_path="/workspace/openjev", ignore=[".git/**", "**/__pycache__/**", ".pytest_cache/**", "checkpoints/**"])
)


def run(cmd: list[str], cwd: Path, env: dict | None = None) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


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


@app.function(
    image=image,
    gpu="A10",
    timeout=60 * 60,
    volumes={"/cache": hf_cache, "/artifacts": artifacts},
)
def de_risk_gpu() -> dict:
    """Train the first semantic head and benchmark it against NanoJev + native Qwen."""
    work = fresh_workdir()
    started = time.time()

    # 1) Build real public semantic data and keep entirely unseen OOD domains.
    run(["python", "scripts/build_public_data.py", "--output", "data/public_decisions", "--benchmark-output", "data/benchmark/open_decision_bench.jsonl"], work)
    run(["python", "scripts/merge_data_dirs.py", "--inputs", "data/simulator", "data/public_decisions", "--output", "data/combined"], work)
    run(["python", "scripts/validate_dataset.py", "data/combined"], work)
    run(["python", "scripts/freeze_benchmark.py", "data/benchmark/open_decision_bench.jsonl"], work)

    # 2) Fast head-only training gate on the same Qwen3-0.6B family NanoJev reports.
    run(["python", "scripts/train.py", "--config", "configs/train_head_public_fast.yaml"], work)

    # 3) Evaluate ours on the frozen real-data benchmark with seen/OOD/domain slices.
    run([
        "python", "scripts/benchmark_checkpoint.py", "--checkpoint", "checkpoints/head-public-fast",
        "--data", "data/benchmark/open_decision_bench.jsonl",
        "--output", "results/ours_open_decision.jsonl", "--batch-questions", "1",
    ], work)
    ours_summary = json.loads((work / "results/ours_open_decision.summary.json").read_text())

    # 4) Untuned Qwen option-likelihood baseline.
    run([
        "python", "scripts/benchmark_option_likelihood.py", "--model", "Qwen/Qwen3-0.6B", "--data", "data/benchmark/open_decision_bench.jsonl",
        "--output", "results/qwen_option_likelihood.jsonl", "--device", "cuda", "--norm", "mean",
    ], work)

    # 5) Pin and run the strongest existing OSS baseline, NanoJev.
    vendor = work / "vendor" / "NanoJev"
    vendor.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "https://github.com/TianyuCodings/NanoJev.git", str(vendor)], work)
    run(["git", "checkout", NANOJEV_COMMIT], vendor)
    checkpoint = work / "checkpoints" / "nanojev"
    code = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id='C-Tianyu/NanoJev', local_dir={str(checkpoint)!r}, "
        "allow_patterns=['best.safetensors','config.json','tokenizer/*','backbone_config/*'])"
    )
    run(["python", "-c", code], work)
    run([
        "python", "scripts/benchmark_nanojev.py", "--data", "data/benchmark/open_decision_bench.jsonl",
        "--nanojev-repo", str(vendor), "--checkpoint-dir", str(checkpoint),
        "--output", "results/nanojev_open_decision.jsonl", "--batch-questions", "3",
    ], work)

    # 6) Put the quality metrics side by side. (NanoJev CLI latency includes model load; don't headline it.)
    run([
        "python", "scripts/compare_models.py",
        "results/ours_open_decision.summary.json",
        "results/qwen_option_likelihood.summary.json",
        "results/nanojev_open_decision.summary.json",
        "--output", "results/open_decision_comparison.md",
    ], work)
    run([
        "python", "scripts/check_launch_gate.py",
        "--ours", "results/ours_open_decision.summary.json",
        "--nanojev", "results/nanojev_open_decision.summary.json",
        "--output", "results/launch_gate.json",
    ], work)
    launch_gate = json.loads((work / "results/launch_gate.json").read_text())

    # 7) If a frozen backbone + learned head is too weak, spend one more cheap run on LoRA.
    # This preserves the independent architecture while allowing semantic representations to adapt.
    if not launch_gate["headline_better_than_nanojev_allowed"]:
        print("Head-only missed the predeclared NanoJev gate; running LoRA fallback.", flush=True)
        run(["python", "scripts/train_lora.py", "--config", "configs/train_lora_public_fast.yaml"], work)
        run([
            "python", "scripts/benchmark_checkpoint.py", "--checkpoint", "checkpoints/lora-public-fast",
            "--data", "data/benchmark/open_decision_bench.jsonl",
            "--output", "results/ours_lora_open_decision.jsonl", "--batch-questions", "1",
        ], work)
        run([
            "python", "scripts/check_launch_gate.py",
            "--ours", "results/ours_lora_open_decision.summary.json",
            "--nanojev", "results/nanojev_open_decision.summary.json",
            "--output", "results/launch_gate_lora.json",
        ], work)
        run([
            "python", "scripts/compare_models.py",
            "results/ours_open_decision.summary.json",
            "results/ours_lora_open_decision.summary.json",
            "results/qwen_option_likelihood.summary.json",
            "results/nanojev_open_decision.summary.json",
            "--output", "results/open_decision_comparison.md",
        ], work)
        lora_gate = json.loads((work / "results/launch_gate_lora.json").read_text())
    else:
        lora_gate = None

    run_name = f"derisk-{int(started)}"
    persisted = persist(work, run_name, [
        "data/public_decisions/manifest.json",
        "data/benchmark/open_decision_bench.jsonl.manifest.json",
        "checkpoints/head-public-fast",
        "results/ours_open_decision.summary.json",
        "results/qwen_option_likelihood.summary.json",
        "results/nanojev_open_decision.summary.json",
        "results/open_decision_comparison.md",
        "results/open_decision_comparison.json",
        "results/launch_gate.json",
        "checkpoints/lora-public-fast",
        "results/ours_lora_open_decision.summary.json",
        "results/launch_gate_lora.json",
    ])
    return {
        "run_name": run_name,
        "artifacts": persisted,
        "elapsed_seconds": time.time() - started,
        "ours": ours_summary,
        "comparison_markdown": (work / "results/open_decision_comparison.md").read_text(),
        "launch_gate": launch_gate,
        "lora_launch_gate": lora_gate,
    }


@app.local_entrypoint()
def main():
    result = de_risk_gpu.remote()
    print(json.dumps(result, indent=2))
    print("\nDownload everything with:")
    print(f"modal volume get openjev-artifacts {result['run_name']} ./modal-results")

@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60,
    volumes={"/cache": hf_cache, "/artifacts": artifacts},
)
def nanojev_plus_gpu() -> dict:
    """Plan B: explicitly fine-tune the public NanoJev checkpoint on our broader data.

    This is a derivative experiment, not the independent OpenJev v0. It is
    intentionally a separate Modal function so it is never run or marketed by accident.
    """
    work = fresh_workdir()
    started = time.time()

    run(["python", "scripts/build_public_data.py", "--output", "data/public_decisions", "--benchmark-output", "data/benchmark/open_decision_bench.jsonl"], work)
    run(["python", "scripts/merge_data_dirs.py", "--inputs", "data/simulator", "data/public_decisions", "--output", "data/combined"], work)
    run(["python", "scripts/validate_dataset.py", "data/combined"], work)
    run(["python", "scripts/freeze_benchmark.py", "data/benchmark/open_decision_bench.jsonl"], work)
    run(["python", "scripts/export_nanojev_training_data.py", "--input-dir", "data/combined", "--output-dir", "data/nanojev_bridge"], work)

    vendor = work / "vendor" / "NanoJev"
    vendor.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "https://github.com/TianyuCodings/NanoJev.git", str(vendor)], work)
    run(["git", "checkout", NANOJEV_COMMIT], vendor)
    checkpoint = work / "checkpoints" / "nanojev"
    code = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id='C-Tianyu/NanoJev', local_dir={str(checkpoint)!r}, "
        "allow_patterns=['best.safetensors','config.json','tokenizer/*','backbone_config/*'])"
    )
    run(["python", "-c", code], work)

    # Validate the bridge with NanoJev's own schema code before using GPU time.
    run([
        "python", str(vendor / "scripts" / "train_pipeline_decisions.py"),
        "--input", str(work / "data" / "nanojev_bridge"), "--validate-only",
    ], vendor)

    plus = work / "checkpoints" / "nanojev-plus"
    run([
        "python", str(vendor / "scripts" / "train_pipeline_decisions.py"),
        "--input", str(work / "data" / "nanojev_bridge"),
        "--output-dir", str(plus),
        "--init-checkpoint", str(checkpoint),
        "--objective", "gold_distribution", "--loss", "brier",
        "--gradient-checkpointing", "--steps", "150", "--head-steps", "0",
        "--batch-questions", "1", "--microbatch-questions", "1",
        "--max-microbatch-tokens", "16384", "--eval-every", "25",
        "--max-length", "512", "--backbone-lr", "2e-5", "--head-lr", "2e-4",
        "--precision", "bf16", "--seed", "17",
    ], vendor)

    run([
        "python", "scripts/benchmark_nanojev.py", "--data", "data/benchmark/open_decision_bench.jsonl",
        "--nanojev-repo", str(vendor), "--checkpoint-dir", str(plus),
        "--output", "results/nanojev_plus_open_decision.jsonl", "--batch-questions", "1",
    ], work)
    # Re-run base NanoJev on the same rows for a direct derivative-vs-base comparison.
    run([
        "python", "scripts/benchmark_nanojev.py", "--data", "data/benchmark/open_decision_bench.jsonl",
        "--nanojev-repo", str(vendor), "--checkpoint-dir", str(checkpoint),
        "--output", "results/nanojev_open_decision.jsonl", "--batch-questions", "1",
    ], work)
    run([
        "python", "scripts/compare_models.py", "results/nanojev_open_decision.summary.json",
        "results/nanojev_plus_open_decision.summary.json", "--output", "results/nanojev_plus_comparison.md",
    ], work)
    run([
        "python", "scripts/check_launch_gate.py", "--ours", "results/nanojev_plus_open_decision.summary.json",
        "--nanojev", "results/nanojev_open_decision.summary.json", "--output", "results/nanojev_plus_gate.json",
    ], work)

    run_name = f"nanojev-plus-{int(started)}"
    persisted = persist(work, run_name, [
        "data/public_decisions/manifest.json",
        "data/benchmark/open_decision_bench.jsonl.manifest.json",
        "checkpoints/nanojev-plus",
        "results/nanojev_open_decision.summary.json",
        "results/nanojev_plus_open_decision.summary.json",
        "results/nanojev_plus_comparison.md",
        "results/nanojev_plus_comparison.json",
        "results/nanojev_plus_gate.json",
    ])
    return {
        "run_name": run_name,
        "artifacts": persisted,
        "elapsed_seconds": time.time() - started,
        "comparison_markdown": (work / "results/nanojev_plus_comparison.md").read_text(),
        "launch_gate": json.loads((work / "results/nanojev_plus_gate.json").read_text()),
        "disclosure": "Derivative NanoJev++ branch initialized from C-Tianyu/NanoJev; not the independent v0.",
    }
