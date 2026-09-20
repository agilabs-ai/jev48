"""Run additional pinned public benchmarks on the selected Jev48 checkpoint."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import shutil
import subprocess
import time

import modal

app = modal.App("jev48-multi-public-benchmarks")
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
        "uv pip install --system 'pydantic>=2.10' 'numpy>=1.26,<2'",
    )
    .env({"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(str(ROOT), remote_path="/workspace/jev48", ignore=["**/__pycache__/**", ".pytest_cache/**", "data/**", "results/**", "jev48-modal-results/**"])
)


@app.function(image=image, gpu="L40S", timeout=2 * 60 * 60, volumes={"/cache": hf_cache, "/artifacts": artifacts})
def run_benchmark(run_name: str, benchmark: str, repo_commit: str, limit: int = 0) -> dict:
    artifacts.reload()
    source = Path("/artifacts") / run_name
    source_model = source / "release/model"
    release_manifest = source / "release/MANIFEST.json"
    if not source_model.exists():
        raise ValueError(f"selected derivative not found for {run_name}")
    manifest = json.loads(release_manifest.read_text())
    model_entry = next((x for x in manifest["files"] if x["path"] == "model.safetensors"), None)
    model_file = source_model / "model.safetensors"
    model_sha = hashlib.sha256(model_file.read_bytes()).hexdigest()
    if not model_entry or model_entry["sha256"] != model_sha or manifest.get("run_name") != run_name:
        raise ValueError("release manifest/model verification failed")
    manifest_sha = hashlib.sha256(release_manifest.read_bytes()).hexdigest()
    work = Path("/tmp/jev48")
    if work.exists(): shutil.rmtree(work)
    shutil.copytree("/workspace/jev48", work)
    model = work / "selected-model"
    shutil.copytree(source_model, model)
    scripts = {"phishing": "benchmark_phishing.py", "jevbench": "benchmark_jevbench_public.py"}
    if benchmark not in scripts: raise ValueError(f"unknown benchmark {benchmark}")
    output = work / f"results/public/{benchmark}.jsonl"
    script = work / "scripts" / scripts[benchmark]
    script_sha = hashlib.sha256(script.read_bytes()).hexdigest()
    if len(repo_commit) != 40:
        raise ValueError("repo_commit must be an immutable 40-character commit")
    cmd = ["python", str(script), "--model", str(model), "--output", str(output), "--device", "cuda", "--run-name", run_name, "--model-sha256", model_sha, "--release-manifest-sha256", manifest_sha, "--repo-commit", repo_commit, "--script-sha256", script_sha]
    if limit: cmd += ["--limit", str(limit)]
    started = time.monotonic()
    subprocess.run(cmd, cwd=work, env={**__import__("os").environ, "PYTHONPATH": str(work)}, check=True)
    target = source / "results/public"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, target / output.name)
    shutil.copy2(output.with_suffix(".summary.json"), target / f"{benchmark}.summary.json")
    summary = json.loads((target / f"{benchmark}.summary.json").read_text())
    summary["runtime_seconds"] = time.monotonic() - started
    (target / f"{benchmark}.summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    artifacts.commit()
    return summary


@app.local_entrypoint()
def main(run_name: str, benchmark: str, limit: int = 0):
    import subprocess
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    print(json.dumps(run_benchmark.remote(run_name, benchmark, commit, limit), indent=2), flush=True)
