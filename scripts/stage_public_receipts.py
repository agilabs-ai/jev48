#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
from urllib.request import urlopen


UPSTREAM_CARD_URL = "https://huggingface.co/datasets/LocalLLaMA/typed-decisions/raw/ea9306458d6e9563628369a3d1e72e362fb381d2/README.md"
UPSTREAM_CARD_SHA256 = "b22b2b778f75574c5f7fc832e9d5b4cfe5b02e51e2a308913cd3cabfb0049a49"
README = """# Published receipts

This directory contains the aggregate, provenance, selection, calibration, and
bootstrap receipts safe to publish with Jev48. `SHA256SUMS` covers every file.

Row-level mixed-dataset inputs and predictions are intentionally withheld pending
a per-source redistribution audit. Their hashes, row counts, builders, and source
identifiers remain published. The complete private receipt tree is preserved in
Modal volume `jev48-artifacts`, run `jev48-1789858979`.

`upstream/typed-decisions-README.md` is the immutable Apache-2.0 benchmark card at
revision `ea9306458d6e9563628369a3d1e72e362fb381d2`. Jev48 does not possess Jev's
row-level predictions, so that comparison is descriptive and unpaired. Only
accuracy is treated as directly comparable; the source card ships no scorer code
with which to prove parity for its reported distribution metrics.
"""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "jev48-modal-results"
    target = root / "receipts"
    if target.exists():
        shutil.rmtree(target)
    files = {
        "MODEL_SOURCE.json": "MODEL_SOURCE.json",
        "SMOKE_TEST.json": "release/SMOKE_TEST.json",
        "data/mtbench_human.manifest.json": "data/mtbench_human.manifest.json",
        "data/decider_suites.manifest.json": "data/decider_suites/manifest.json",
        "data/jev48_v1.manifest.json": "data/jev48_v1/manifest.json",
        "data/final.jsonl.manifest.json": "data/jev48_v1/final.jsonl.manifest.json",
        "results/selection.json": "results/selection.json",
        "results/final_report.json": "results/final_report.json",
        "results/final_report.md": "results/final_report.md",
        "results/LAUNCH_FACTS.md": "results/LAUNCH_FACTS.md",
        "results/run_manifest.json": "results/run_manifest.json",
        "results/bootstrap-jev48-vs-base.json": "results/bootstrap-jev48-vs-base.json",
        "results/typed-decisions.summary.json": "results/public/typed-decisions.summary.json",
        "results/base.raw.summary.json": "results/final/base.raw.summary.json",
        "results/base.calibrated.summary.json": "results/final/base.calibrated.summary.json",
        "results/jev48.raw.summary.json": "results/final/jev48.raw.summary.json",
        "results/jev48.calibrated.summary.json": "results/final/jev48.calibrated.summary.json",
    }
    for name in ("base", "hard-lr1e-6", "soft-lr3e-7", "soft-lr1e-6", "soft-lr3e-6"):
        files[f"results/dev/{name}.summary.json"] = f"results/dev/{name}.summary.json"
    for destination, relative in files.items():
        src, dst = source / relative, target / destination
        if not src.exists():
            raise FileNotFoundError(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    upstream = target / "upstream/typed-decisions-README.md"
    upstream.parent.mkdir(parents=True, exist_ok=True)
    upstream.write_bytes(urlopen(UPSTREAM_CARD_URL, timeout=30).read())
    if digest(upstream) != UPSTREAM_CARD_SHA256:
        raise RuntimeError("pinned typed-decisions card hash mismatch")
    (target / "README.md").write_text(README, encoding="utf-8")
    published = sorted(path for path in target.rglob("*") if path.is_file())
    sums = "".join(f"{digest(path)}  {path.relative_to(target)}\n" for path in published)
    (target / "SHA256SUMS").write_text(sums, encoding="utf-8")


if __name__ == "__main__":
    main()
