"""Clean-container inference smoke test for the staged Jev48 checkpoint."""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parent
COMMIT = "b08acf787d5d1f718a8c36c4677960f43772c7be"
app = modal.App("jev48-release-smoke")
artifacts = modal.Volume.from_name("jev48-artifacts", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install("uv>=0.8")
    .run_commands(
        f"git clone https://github.com/Mapika/decider.git /opt/decider && cd /opt/decider && git checkout {COMMIT}",
        "cd /opt/decider && uv pip install --system -e '.[train]'",
    )
    .add_local_dir(str(ROOT), remote_path="/workspace/jev48", ignore=[".git/**", ".venv/**", "jev48-modal-results/**"])
)


@app.function(image=image, gpu="L40S", timeout=20 * 60, volumes={"/artifacts": artifacts})
def smoke(run_name: str) -> dict:
    import torch
    from decider.infer import Decider

    artifacts.reload()
    model_path = Path("/artifacts") / run_name / "release/model"
    model = Decider(str(model_path), device="cuda", dtype=torch.bfloat16,
                    temperature=None, use_graphs=False)
    questions = {
        "choice": {
            "type": "choice",
            "instructions": "Choose the safer release action.",
            "criteria": {"a": "Publish without tests", "b": "Verify, then publish"},
        }
    }
    output = model.system_one("The artifact has not yet been verified.", questions, independent=True)
    probabilities = output["answers"]["choice"]["probabilities"]
    values = [float(probabilities[key]) for key in ("a", "b")]
    if any(value < 0 or value > 1 for value in values) or abs(sum(values) - 1.0) > 1e-4:
        raise ValueError(f"invalid probabilities: {probabilities}")
    receipt = {
        "ok": True,
        "run_name": run_name,
        "tested_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(0),
        "dtype": "bfloat16",
        "upstream_commit": "b08acf787d5d1f718a8c36c4677960f43772c7be",
        "packages": {name: version(name) for name in ("decider", "torch", "transformers", "safetensors")},
        "probability_keys": sorted(probabilities),
        "probability_sum": sum(values),
    }
    path = Path("/artifacts") / run_name / "release/SMOKE_TEST.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts.commit()
    return receipt


@app.local_entrypoint()
def smoke_main(run_name: str):
    print(json.dumps(smoke.remote(run_name), indent=2, sort_keys=True))
