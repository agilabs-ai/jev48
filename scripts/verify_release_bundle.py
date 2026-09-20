#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


REQUIRED = {
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "model.safetensors",
    "receipts/MODEL_SOURCE.json",
    "receipts/run_manifest.json",
    "receipts/selection.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(bundle: Path, manifest_path: Path, *, check_safetensors: bool = True) -> dict:
    bundle, manifest_path = bundle.resolve(), manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest files must be a non-empty list")

    declared: dict[str, dict] = {}
    for entry in entries:
        rel = entry.get("path")
        pure = PurePosixPath(rel) if isinstance(rel, str) else None
        if pure is None or pure.is_absolute() or ".." in pure.parts or rel in declared:
            raise ValueError(f"unsafe or duplicate manifest path: {rel!r}")
        declared[rel] = entry

    actual: dict[str, Path] = {}
    for path in bundle.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"symlink forbidden in release bundle: {path}")
        if path.is_file():
            actual[path.relative_to(bundle).as_posix()] = path
    if set(actual) != set(declared):
        raise ValueError(
            f"bundle file set differs from manifest; missing={sorted(set(declared)-set(actual))}, "
            f"unexpected={sorted(set(actual)-set(declared))}"
        )
    missing_required = sorted(REQUIRED - set(actual))
    if missing_required:
        raise ValueError(f"required release files missing: {missing_required}")

    for rel, path in actual.items():
        entry = declared[rel]
        if entry.get("bytes") != path.stat().st_size:
            raise ValueError(f"size mismatch: {rel}")
        if entry.get("sha256") != sha256(path):
            raise ValueError(f"sha256 mismatch: {rel}")

    source = json.loads(actual["receipts/MODEL_SOURCE.json"].read_text())
    run = json.loads(actual["receipts/run_manifest.json"].read_text())
    selection = json.loads(actual["receipts/selection.json"].read_text())
    for receipt_name, receipt in (("MODEL_SOURCE", source), ("run_manifest", run)):
        if receipt.get("base_model") != manifest.get("base_model"):
            raise ValueError(f"{receipt_name} base model disagrees with manifest")
        if receipt.get("base_model_revision") != manifest.get("base_model_revision"):
            raise ValueError(f"{receipt_name} base revision disagrees with manifest")
    if run.get("upstream_commit") != manifest.get("upstream_commit"):
        raise ValueError("run manifest upstream commit disagrees with release manifest")
    if selection.get("selection_uses_locked_test_or_ood") is not False:
        raise ValueError("selection receipt does not affirm locked exclusion")

    if check_safetensors:
        from safetensors import safe_open

        with safe_open(actual["model.safetensors"], framework="numpy", device="cpu") as handle:
            if not handle.keys():
                raise ValueError("model.safetensors contains no tensors")
    return {"ok": True, "files": len(actual), "bytes": sum(p.stat().st_size for p in actual.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed verification of a Jev48 release bundle.")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.bundle, args.manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
