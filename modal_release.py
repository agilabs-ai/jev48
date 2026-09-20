"""Stage and validate the private Jev48 model bundle before publication."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import modal


ROOT = Path(__file__).resolve().parent
DECIDER_COMMIT = "b08acf787d5d1f718a8c36c4677960f43772c7be"
app = modal.App("jev48-release-stage")
artifacts = modal.Volume.from_name("jev48-artifacts", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands(f"git clone https://github.com/Mapika/decider.git /opt/decider && cd /opt/decider && git checkout {DECIDER_COMMIT}")
    .add_local_dir(str(ROOT), remote_path="/workspace/jev48", ignore=[".git/**", ".venv/**", "jev48-modal-results/**", "**/__pycache__/**"])
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.function(image=image, timeout=60 * 60, volumes={"/artifacts": artifacts})
def stage(run_name: str) -> dict:
    artifacts.reload()
    run = Path("/artifacts") / run_name
    model = run / "release/model"
    if not (model / "model.safetensors").exists():
        raise ValueError(f"missing selected model for {run_name}")
    shutil.copy2("/workspace/jev48/MODEL_CARD.md", model / "README.md")
    shutil.copy2("/opt/decider/LICENSE", model / "LICENSE")
    shutil.copy2("/workspace/jev48/THIRD_PARTY_NOTICES.md", model / "THIRD_PARTY_NOTICES.md")
    receipts = model / "receipts"
    receipts.mkdir(exist_ok=True)
    sources = {
        "selection.json": run / "results/selection.json",
        "run_manifest.json": run / "results/run_manifest.json",
        "jev48.calibrated.summary.json": run / "results/final/jev48.calibrated.summary.json",
        "base.calibrated.summary.json": run / "results/final/base.calibrated.summary.json",
        "typed-decisions.summary.json": run / "results/public/typed-decisions.summary.json",
        "MODEL_SOURCE.json": run / "release/MODEL_SOURCE.json",
    }
    for name, source in sources.items():
        if not source.exists():
            raise ValueError(f"missing release receipt: {source}")
        shutil.copy2(source, receipts / name)
    files = []
    for path in sorted(p for p in model.rglob("*") if p.is_file()):
        files.append({"path": str(path.relative_to(model)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "release": "agilabs-ai/jev48-2b",
        "visibility_at_staging": "private/not-uploaded",
        "run_name": run_name,
        "upstream_commit": DECIDER_COMMIT,
        "files": files,
    }
    (run / "release/MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts.commit()
    return {"run_name": run_name, "files": len(files), "bytes": sum(item["bytes"] for item in files), "manifest": str(run / "release/MANIFEST.json")}


@app.local_entrypoint()
def main(run_name: str):
    print(json.dumps(stage.remote(run_name), indent=2), flush=True)
