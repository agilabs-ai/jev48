from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from .metrics import summarize
from .schema import DecisionExample


def sliced_summary(
    probs: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    examples: Sequence[DecisionExample],
) -> dict:
    if not (len(probs) == len(targets) == len(examples)) or not examples:
        raise ValueError("probs, targets, and examples must be nonempty and aligned")

    by_split: dict[str, list[int]] = defaultdict(list)
    by_domain: dict[str, list[int]] = defaultdict(list)
    by_candidate_count: dict[str, list[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        by_split[ex.split].append(i)
        by_domain[ex.domain].append(i)
        k = len(ex.candidates)
        bucket = "2" if k == 2 else "3-8" if k <= 8 else "9-32" if k <= 32 else "33-64" if k <= 64 else "65+"
        by_candidate_count[bucket].append(i)

    def take(indices: list[int]) -> dict:
        return summarize([probs[i] for i in indices], [targets[i] for i in indices])

    return {
        "overall": summarize(probs, targets),
        "by_split": {key: take(ix) for key, ix in sorted(by_split.items())},
        "by_domain": {key: take(ix) for key, ix in sorted(by_domain.items())},
        "by_candidate_count": {key: take(ix) for key, ix in sorted(by_candidate_count.items())},
    }
