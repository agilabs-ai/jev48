from __future__ import annotations

from typing import Sequence

from .schema import DecisionExample


NANOJEV_GITHUB_COMMIT = "71a513bb0163b5634467842b523ee0c0ed6fb1c7"
NANOJEV_HF_REPO = "C-Tianyu/NanoJev"


def build_nanojev_request(examples: Sequence[DecisionExample]) -> dict:
    """Convert our frozen benchmark rows to NanoJev's public inference schema.

    Each benchmark example becomes one state with one choice question. Keeping one
    question per state makes result alignment unambiguous while NanoJev may still
    batch every candidate path into one backbone forward.
    """
    states = []
    for ex in examples:
        states.append({
            "id": ex.id,
            "state": ex.state,
            "questions": {
                "decision": {
                    "type": "choice",
                    "instructions": ex.question,
                    "criteria": {c.id: c.text for c in ex.candidates},
                }
            },
        })
    return {"states": states}


def parse_nanojev_response(payload: dict, examples: Sequence[DecisionExample]) -> list[list[float]]:
    by_id = {row["id"]: row for row in payload.get("states", [])}
    probs: list[list[float]] = []
    for ex in examples:
        if ex.id not in by_id:
            raise ValueError(f"NanoJev response missing state {ex.id}")
        answer = by_id[ex.id].get("answers", {}).get("decision")
        if not isinstance(answer, dict):
            raise ValueError(f"NanoJev response missing decision answer for {ex.id}")
        mapping = answer.get("probabilities")
        if not isinstance(mapping, dict):
            raise ValueError(f"NanoJev response missing probability mapping for {ex.id}")
        ids = [c.id for c in ex.candidates]
        if set(mapping) != set(ids):
            raise ValueError(f"NanoJev candidate mismatch for {ex.id}: {set(mapping)} != {set(ids)}")
        row = [float(mapping[cid]) for cid in ids]
        if abs(sum(row) - 1.0) > 1e-5:
            raise ValueError(f"NanoJev probabilities for {ex.id} do not sum to one")
        probs.append(row)
    return probs
