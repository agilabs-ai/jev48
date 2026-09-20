"""Run the selected Jev48 checkpoint on a pinned public benchmark with a published Jev row."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import modal

app = modal.App("jev48-public-benchmark")
ROOT = Path(__file__).resolve().parent
DECIDER_COMMIT = "b08acf787d5d1f718a8c36c4677960f43772c7be"
hf_cache = modal.Volume.from_name("jev48-hf-cache", create_if_missing=True)
artifacts = modal.Volume.from_name("jev48-artifacts", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install("uv>=0.8")
    .run_commands(
        f"git clone https://github.com/Mapika/decider.git /opt/decider && cd /opt/decider && git checkout {DECIDER_COMMIT}",
        "cd /opt/decider && uv pip install --system -e '.[train,serve]'",
        "uv pip install --system 'pydantic>=2.10' 'datasets>=3.6,<4'",
    )
    .env({"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub", "HF_DATASETS_CACHE": "/cache/huggingface/datasets-3.6", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(str(ROOT), remote_path="/workspace/jev48", ignore=["**/__pycache__/**", ".pytest_cache/**", "data/**", "results/**", "jev48-modal-results/**"])
)


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60,
    volumes={"/cache": hf_cache, "/artifacts": artifacts},
)
def run_public_benchmark(run_name: str, limit: int = 0) -> dict:
    artifacts.reload()
    source = Path("/artifacts") / run_name
    source_model = source / "release/model"
    if not source_model.exists():
        raise ValueError(f"selected derivative not found for {run_name}")
    work = Path("/tmp/jev48")
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree("/workspace/jev48", work)
    model = work / "selected-model"
    shutil.copytree(source_model, model)
    output = work / "results/public/typed-decisions.jsonl"
    cmd = [
        "python", "scripts/benchmark_typed_decisions.py",
        "--model", str(model),
        "--output", str(output),
        "--device", "cuda",
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    env = {**__import__("os").environ, "PYTHONPATH": str(work)}
    subprocess.run(cmd, cwd=work, env=env, check=True)
    target = source / "results/public"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, target / output.name)
    shutil.copy2(output.with_suffix(".summary.json"), target / "typed-decisions.summary.json")
    artifacts.commit()
    return json.loads((target / "typed-decisions.summary.json").read_text(encoding="utf-8"))


@app.local_entrypoint()
def main(run_name: str, limit: int = 0):
    print(json.dumps(run_public_benchmark.remote(run_name, limit), indent=2), flush=True)
