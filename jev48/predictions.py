from __future__ import annotations

import math
from typing import Iterable, Sequence
import numpy as np


def temperature_transform(probs: Sequence[float], temperature: float, eps: float = 1e-30) -> list[float]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and > 0")
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 1 or len(p) < 2 or (p < 0).any() or not np.isfinite(p).all() or p.sum() <= 0:
        raise ValueError("invalid probability vector")
    p = p / p.sum()
    z = np.log(np.clip(p, eps, 1.0)) / temperature
    z -= z.max()
    q = np.exp(z)
    q /= q.sum()
    return q.tolist()


def soft_nll(probs: Sequence[Sequence[float]], targets: Sequence[Sequence[float]], temperature: float) -> float:
    if len(probs) != len(targets) or not probs:
        raise ValueError("probs and targets must be nonempty and aligned")
    total = 0.0
    for p, t in zip(probs, targets):
        q = np.asarray(temperature_transform(p, temperature), dtype=np.float64)
        y = np.asarray(t, dtype=np.float64)
        total += float(-(y * np.log(np.clip(q, 1e-12, 1.0))).sum())
    return total / len(probs)


def fit_temperature_from_probs(
    probs: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    lo: float = 0.2,
    hi: float = 5.0,
) -> float:
    """Deterministic coarse-to-fine scalar temperature fit by soft-target NLL."""
    if lo <= 0 or hi <= lo:
        raise ValueError("invalid search interval")
    left, right = math.log(lo), math.log(hi)
    best_t = 1.0
    for _ in range(4):
        grid = np.exp(np.linspace(left, right, 121))
        vals = [soft_nll(probs, targets, float(t)) for t in grid]
        i = int(np.argmin(vals))
        best_t = float(grid[i])
        step = (right - left) / 120
        left = max(math.log(lo), math.log(best_t) - 3 * step)
        right = min(math.log(hi), math.log(best_t) + 3 * step)
    return best_t
