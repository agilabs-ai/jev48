from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def binary_metrics(labels: Iterable[int], probabilities: Iterable[float], bins: int = 10) -> dict[str, float | int]:
    y = np.asarray(list(labels), dtype=np.int8)
    p = np.asarray(list(probabilities), dtype=np.float64)
    if not len(y) or len(y) != len(p):
        raise ValueError("labels and probabilities must have equal non-zero length")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("probabilities must be finite and in [0,1]")
    pred = (p >= 0.5).astype(np.int8)
    pos, neg = y == 1, y == 0
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    sorted_p = p[order]
    i = 0
    while i < len(p):
        j = i + 1
        while j < len(p) and sorted_p[j] == sorted_p[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    auroc = (float(ranks[pos].sum()) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    confidence = np.where(pred == 1, p, 1 - p)
    correct = pred == y
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (confidence >= lo) & (confidence <= hi if hi == 1 else confidence < hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(confidence[mask].mean()) - float(correct[mask].mean()))
    return {
        "n": len(y),
        "accuracy": float(correct.mean()),
        "recall": float((pred[pos] == 1).mean()),
        "false_positive_rate": float((pred[neg] == 1).mean()),
        "auroc": float(auroc),
        "ece_10": float(ece),
        "brier": float(np.square(p - y).mean()),
    }


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n <= 0:
        raise ValueError("n must be positive")
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return [center - half, center + half]


def paired_bootstrap_delta(ours: Iterable[bool], reference: Iterable[bool], *, seed: int = 48, samples: int = 10_000) -> dict[str, float | list[float] | int]:
    a = np.asarray(list(ours), dtype=float)
    b = np.asarray(list(reference), dtype=float)
    if not len(a) or len(a) != len(b):
        raise ValueError("paired outcomes must have equal non-zero length")
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=float)
    for start in range(0, samples, 500):
        count = min(500, samples - start)
        idx = rng.integers(0, len(a), size=(count, len(a)))
        deltas[start:start + count] = (a[idx] - b[idx]).mean(axis=1)
    return {
        "n": len(a),
        "delta": float(a.mean() - b.mean()),
        "ci95": [float(x) for x in np.quantile(deltas, [0.025, 0.975])],
        "bootstrap_samples": samples,
        "seed": seed,
    }


def paired_cluster_bootstrap_delta(ours: Iterable[bool], reference: Iterable[bool], clusters: Iterable[str], *, seed: int = 48, samples: int = 10_000) -> dict[str, float | list[float] | int]:
    a = np.asarray(list(ours), dtype=float)
    b = np.asarray(list(reference), dtype=float)
    c = np.asarray(list(clusters), dtype=object)
    if not len(a) or len(a) != len(b) or len(a) != len(c):
        raise ValueError("paired outcomes and clusters must have equal non-zero length")
    names = np.asarray(sorted(set(map(str, c))))
    indices = [np.flatnonzero(c == name) for name in names]
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=float)
    for i in range(samples):
        chosen = rng.integers(0, len(names), size=len(names))
        idx = np.concatenate([indices[j] for j in chosen])
        deltas[i] = (a[idx] - b[idx]).mean()
    return {
        "n": len(a),
        "n_clusters": len(names),
        "cluster_field": "family",
        "delta": float(a.mean() - b.mean()),
        "ci95": [float(x) for x in np.quantile(deltas, [0.025, 0.975])],
        "bootstrap_samples": samples,
        "seed": seed,
    }
