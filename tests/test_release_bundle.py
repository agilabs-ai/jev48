from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.verify_release_bundle import verify


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    bundle = tmp_path / "model"
    receipts = bundle / "receipts"
    receipts.mkdir(parents=True)
    base = {"base_model": "Mapika/decider-2b", "base_model_revision": "a" * 40}
    files = {
        "README.md": "card",
        "LICENSE": "license",
        "THIRD_PARTY_NOTICES.md": "notice",
        "model.safetensors": "fake-for-unit-test",
        "receipts/MODEL_SOURCE.json": json.dumps(base),
        "receipts/run_manifest.json": json.dumps({**base, "upstream_commit": "b" * 40}),
        "receipts/selection.json": json.dumps({"selection_uses_locked_test_or_ood": False}),
    }
    entries = []
    for rel, content in files.items():
        path = bundle / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        entries.append({"path": rel, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(json.dumps({"base_model": base["base_model"], "base_model_revision": "a" * 40, "upstream_commit": "b" * 40, "files": entries}))
    return bundle, manifest


def test_bundle_verifier_accepts_exact_bundle(tmp_path):
    bundle, manifest = _fixture(tmp_path)
    assert verify(bundle, manifest, check_safetensors=False)["ok"] is True


def test_bundle_verifier_rejects_tamper(tmp_path):
    bundle, manifest = _fixture(tmp_path)
    (bundle / "README.md").write_text("tampered")
    with pytest.raises(ValueError, match="mismatch"):
        verify(bundle, manifest, check_safetensors=False)


def test_bundle_verifier_rejects_unexpected_file(tmp_path):
    bundle, manifest = _fixture(tmp_path)
    (bundle / "extra.txt").write_text("surprise")
    with pytest.raises(ValueError, match="unexpected"):
        verify(bundle, manifest, check_safetensors=False)
