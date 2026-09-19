#!/usr/bin/env python3
"""Export our unified decision rows into NanoJev's public training schema.

This is intentionally a Plan-B bridge. Using it creates a derivative NanoJev++
branch and must be disclosed separately from the independently built v0.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from open_system_one.io import read_jsonl


def convert(ex) -> dict:
    criteria = {c.id: c.text for c in ex.candidates}
    ids = [c.id for c in ex.candidates]
    probs = {cid: float(p) for cid, p in zip(ids, ex.target_probs)}
    row = {
        "id": ex.id,
        "state_id": ex.id,
        "family_id": ex.family_id,
        "split": ex.split,
        "state": ex.state,
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": ex.question,
                "criteria": criteria,
            }
        },
        "gold_probs": {"decision": probs},
        "metadata": {**ex.metadata, "source_domain": ex.domain, "source_target_kind": ex.target_kind},
    }
    if ex.target_kind == "known_distribution":
        row["gold_probs_kind"] = {"decision": "programmatic_conditional_distribution"}
        row["gold_label_kind"] = {"decision": "unobserved"}
    elif ex.target_kind in {"deterministic_truth", "observed_outcome"}:
        best = ids[ex.gold_index]
        row["gold"] = {"decision": best}
        if ex.target_kind == "deterministic_truth":
            row["gold_probs_kind"] = {"decision": "deterministic_truth"}
            row["gold_label_kind"] = {"decision": "deterministic_truth"}
        else:
            # Observed outcomes are trained as one-hot observations; NanoJev's
            # gold_distribution objective can still consume the explicit one-hot q.
            row["gold_probs_kind"] = {"decision": "deterministic_truth"}
            row["gold_label_kind"] = {"decision": "observed_outcome"}
    else:
        # Teacher distributions are distributions, not independently verified truth.
        # Export as a teacher target rather than mislabeling it as gold.
        row.pop("gold_probs")
        row["teacher"] = {
            "native_probs": {"decision": probs},
            "rounding": {"probabilityDecimals": 15},
        }
        row["gold_label_kind"] = {"decision": "unobserved"}
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="data/combined")
    p.add_argument("--output-dir", default="data/nanojev_bridge")
    args = p.parse_args()
    src, dst = Path(args.input_dir), Path(args.output_dir)
    dst.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ["train", "dev", "calibration", "test", "ood"]:
        path = src / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = [convert(ex) for ex in read_jsonl(path)]
        (dst / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
        counts[split] = len(rows)
    print(json.dumps({"output_dir": str(dst), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
