from __future__ import annotations

import math
from typing import Sequence
import numpy as np


def _arrays(probs: Sequence[Sequence[float]], targets: Sequence[Sequence[float]]):
    if len(probs) != len(targets) or not probs:
        raise ValueError("probs and targets must be nonempty and aligned")
    return [np.asarray(p, dtype=np.float64) for p in probs], [np.asarray(t, dtype=np.float64) for t in targets]


def accuracy(probs, targets) -> float:
    ps, ts = _arrays(probs, targets)
    return float(np.mean([int(p.argmax() == t.argmax()) for p, t in zip(ps, ts)]))


def brier(probs, targets) -> float:
    ps, ts = _arrays(probs, targets)
    return float(np.mean([np.square(p - t).sum() for p, t in zip(ps, ts)]))


def nll(probs, targets, eps: float = 1e-12) -> float:
    ps, ts = _arrays(probs, targets)
    vals = []
    for p, t in zip(ps, ts):
        p = np.clip(p, eps, 1.0)
        vals.append(float(-(t * np.log(p)).sum()))
    return float(np.mean(vals))


def total_variation(probs, targets) -> float:
    ps, ts = _arrays(probs, targets)
    return float(np.mean([0.5 * np.abs(p - t).sum() for p, t in zip(ps, ts)]))


def expected_calibration_error(probs, targets, bins: int = 15) -> float:
    ps, ts = _arrays(probs, targets)
    confidences = np.asarray([float(p.max()) for p in ps])
    # For soft targets, the expected correctness of choosing class j is target[j].
    # This collapses to ordinary 0/1 correctness for deterministic one-hot labels.
    correct = np.asarray([float(t[int(p.argmax())]) for p, t in zip(ps, ts)])
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        if i == bins - 1:
            mask = (confidences >= edges[i]) & (confidences <= edges[i + 1])
        else:
            mask = (confidences >= edges[i]) & (confidences < edges[i + 1])
        if mask.any():
            ece += float(mask.mean()) * abs(float(confidences[mask].mean()) - float(correct[mask].mean()))
    return ece


def risk_coverage(probs, targets, thresholds=(0.5, 0.7, 0.8, 0.9, 0.95)) -> list[dict]:
    ps, ts = _arrays(probs, targets)
    out = []
    for threshold in thresholds:
        selected = [(p, t) for p, t in zip(ps, ts) if float(p.max()) >= threshold]
        if not selected:
            out.append({"threshold": threshold, "coverage": 0.0, "error_rate": None, "n": 0})
            continue
        # Expected error under the target distribution; exact 0/1 error for one-hot labels.
        errors = sum(1.0 - float(t[int(p.argmax())]) for p, t in selected)
        out.append({
            "threshold": threshold,
            "coverage": len(selected) / len(ps),
            "error_rate": errors / len(selected),
            "n": len(selected),
        })
    return out


def summarize(probs, targets) -> dict:
    return {
        "n": len(probs),
        "accuracy": accuracy(probs, targets),
        "brier": brier(probs, targets),
        "nll": nll(probs, targets),
        "ece_15": expected_calibration_error(probs, targets, bins=15),
        "total_variation": total_variation(probs, targets),
        "risk_coverage": risk_coverage(probs, targets),
    }
