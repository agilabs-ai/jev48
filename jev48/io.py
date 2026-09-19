from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Iterable

from .schema import DecisionExample


def read_jsonl(path: str | Path) -> list[DecisionExample]:
    result: list[DecisionExample] = []
    seen: set[str] = set()
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            ex = DecisionExample.model_validate_json(line)
        except Exception as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
        if ex.id in seen:
            raise ValueError(f"duplicate example id: {ex.id}")
        seen.add(ex.id)
        result.append(ex)
    return result


def write_jsonl(path: str | Path, examples: Iterable[DecisionExample]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_json(path: str | Path, value) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
