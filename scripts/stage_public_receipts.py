#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
from urllib.request import urlopen


UPSTREAM_CARD_URL = "https://huggingface.co/datasets/LocalLLaMA/typed-decisions/raw/ea9306458d6e9563628369a3d1e72e362fb381d2/README.md"
UPSTREAM_CARD_SHA256 = "b22b2b778f75574c5f7fc832e9d5b4cfe5b02e51e2a308913cd3cabfb0049a49"
PHISH_BENCHMARK_COMMIT = "1d56e8c64d029a9554a0874e2ef2901ed196e230"
PHISH_METRICS_URL = f"https://raw.githubusercontent.com/anisselbd/jev-phishing-bench/{PHISH_BENCHMARK_COMMIT}/results/metrics.json"
PHISH_METRICS_SHA256 = "8beb3f727dd48adc5163c398e73fae639bcc6eec568cfa1e243331843b334e69"
PHISH_DATASET_REVISION = "89afcc39610084298c4679159cb2e27d9ffffa46"
PHISH_DATASET_README_URL = f"https://huggingface.co/datasets/AreLit/PhishNChips/raw/{PHISH_DATASET_REVISION}/README.md"
PHISH_DATASET_README_SHA256 = "5b4df6ac2f2e86b6df13b2b8ca063860491d811f0733807d98093d3c7a9f8d2d"
PHISH_LICENSES_URL = f"https://huggingface.co/datasets/AreLit/PhishNChips/raw/{PHISH_DATASET_REVISION}/SOURCE_LICENSES.md"
PHISH_LICENSES_SHA256 = "129e19acd6ae8fb243b7861fd7f7c3c63987f431b0f73626b7d21206e275f8b7"
README = """# Published receipts

This directory contains the aggregate, provenance, selection, calibration, and
bootstrap receipts safe to publish with Jev48. `SHA256SUMS` covers every file.

Row-level mixed-dataset inputs and predictions are intentionally withheld pending
a per-source redistribution audit. Their hashes, row counts, builders, and source
identifiers remain published. The complete private receipt tree is preserved in
Modal volume `jev48-artifacts`, run `jev48-1789867911`.

`upstream/typed-decisions-README.md` is the immutable Apache-2.0 benchmark card at
revision `ea9306458d6e9563628369a3d1e72e362fb381d2`. Jev48 does not possess Jev's
row-level predictions, so that comparison is descriptive and unpaired. Only
accuracy is treated as directly comparable; the source card ships no scorer code
with which to prove parity for its reported distribution metrics.

The additional public summaries cover all 2,000 PhishNChips v5.2 emails and all
231 public JevBench v1.2.2 tasks. Row-level predictions remain in the private Modal
receipt tree because both benchmarks carry source-specific redistribution terms.
Source revisions, input hashes, comparison limitations, and uncertainty intervals
are embedded in the summaries. Pinned upstream metric and mixed-source licensing
evidence is bundled under `upstream/`.
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
        "RELEASE_MANIFEST.json": "release/MANIFEST.json",
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
        "results/phishing.summary.json": "results/public/phishing.summary.json",
        "results/jevbench-public.summary.json": "results/public/jevbench.summary.json",
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
    upstream_files = {
        "upstream/jev-phishing-metrics.json": (PHISH_METRICS_URL, PHISH_METRICS_SHA256),
        "upstream/phishnchips-README.md": (PHISH_DATASET_README_URL, PHISH_DATASET_README_SHA256),
        "upstream/phishnchips-SOURCE_LICENSES.md": (PHISH_LICENSES_URL, PHISH_LICENSES_SHA256),
    }
    for relative, (url, expected) in upstream_files.items():
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urlopen(url, timeout=30).read())
        if digest(path) != expected:
            raise RuntimeError(f"pinned upstream hash mismatch: {relative}")
    (target / "README.md").write_text(README, encoding="utf-8")
    published = sorted(path for path in target.rglob("*") if path.is_file())
    sums = "".join(f"{digest(path)}  {path.relative_to(target)}\n" for path in published)
    (target / "SHA256SUMS").write_text(sums, encoding="utf-8")


if __name__ == "__main__":
    main()
