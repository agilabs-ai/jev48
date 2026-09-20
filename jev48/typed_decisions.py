from __future__ import annotations

import json
import math
from typing import Any, Iterable

import numpy as np


DATASET = "LocalLLaMA/typed-decisions"
REVISION = "ea9306458d6e9563628369a3d1e72e362fb381d2"


def normalize_question(question: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    kind = question["type"]
    # The public benchmark permits noul questions without explicit criteria;
    # the primitive itself still has the canonical false/true support.
    if kind == "noul":
        criteria = question.get("criteria", {})
        return ["false", "true"], {
            "false": str(criteria.get("false", "No / false")),
            "true": str(criteria.get("true", "Yes / true")),
        }
    criteria = question["criteria"]
    if kind == "choice":
        ids = list(criteria)
        return ids, {str(k): str(v) for k, v in criteria.items()}
    if kind == "score":
        ids = [str(i) for i in range(len(criteria))]
        return ids, {str(i): str(v) for i, v in enumerate(criteria)}
    raise ValueError(f"unsupported question type: {kind}")


def state_text(value: str | dict[str, Any]) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def answer_from_choice(question: dict[str, Any], probabilities: dict[str, float]) -> dict[str, Any]:
    ids, _ = normalize_question(question)
    probs = np.asarray([float(probabilities[k]) for k in ids], dtype=np.float64)
    probs /= probs.sum()
    kind = question["type"]
    if kind == "noul":
        return {"noul": float(probs[1]), "probabilities": dict(zip(ids, probs.tolist()))}
    if kind == "score":
        score = float(sum(i * p for i, p in enumerate(probs)))
        return {"score": score, "probabilities": dict(zip(ids, probs.tolist()))}
    return {"choice": ids[int(probs.argmax())], "probabilities": dict(zip(ids, probs.tolist()))}


def ece(confidence: list[float], correct: list[float], bins: int = 15) -> float:
    conf = np.asarray(confidence, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    total = 0.0
    for lo in np.linspace(0.0, 1.0, bins + 1)[:-1]:
        hi = lo + 1.0 / bins
        mask = (conf >= lo) & (conf <= hi if hi >= 1.0 else conf < hi)
        if mask.any():
            total += float(mask.mean()) * abs(float(conf[mask].mean()) - float(corr[mask].mean()))
    return total


def score_predictions(rows: Iterable[dict[str, Any]]) -> dict[str, float | int | None]:
    accuracy: list[float] = []
    soft_accuracy: list[float] = []
    brier: list[float] = []
    kl: list[float] = []
    tv: list[float] = []
    score_mae: list[float] = []
    within_one: list[float] = []
    confidence: list[float] = []
    for row in rows:
        for qid, question in row["questions"].items():
            pred, gold = row["predictions"][qid], row["gold"][qid]
            kind = question["type"]
            ids, _ = normalize_question(question)
            if kind == "choice":
                p = np.asarray([pred["probabilities"].get(k, 1e-6) for k in ids], dtype=float)
                g = np.asarray([gold["probabilities"].get(k, 1e-6) for k in ids], dtype=float)
                predicted, expected = ids[int(p.argmax())], str(gold["label"])
            elif kind == "noul":
                pv = float(pred["noul"])
                gv = float(gold.get("noul", gold.get("probabilities", {}).get("true", 0.5)))
                p, g = np.asarray([1 - pv, pv]), np.asarray([1 - gv, gv])
                predicted = "true" if pv >= 0.5 else "false"
                expected = str(gold["label"]).lower()
            else:
                p = np.asarray([pred["probabilities"].get(k, 0.0) for k in ids], dtype=float)
                g = None
                predicted = str(int(p.argmax()))
                expected = str(int(gold.get("label", round(float(gold.get("score", 0.0))))))
                error = abs(float(pred["score"]) - float(gold.get("score", 0.0)))
                score_mae.append(error)
                within_one.append(float(error <= 1.0))
            p = p / p.sum()
            is_correct = float(predicted == expected)
            accuracy.append(is_correct)
            confidence.append(float(p.max()))
            if g is not None:
                g = g / g.sum()
                soft_accuracy.append(float(np.dot(p, g)))
                brier.append(float(np.square(p - g).sum()))
                tv.append(float(0.5 * np.abs(p - g).sum()))
                safe_p = np.clip(p, 1e-12, 1.0)
                kl.append(float(sum(x * math.log(x / y) for x, y in zip(g, safe_p) if x > 0)))
    return {
        "n_decisions": len(accuracy),
        "accuracy": float(np.mean(accuracy)),
        "soft_accuracy": float(np.mean(soft_accuracy)),
        "brier": float(np.mean(brier)),
        "kl": float(np.mean(kl)),
        "total_variation": float(np.mean(tv)),
        "ece_15": ece(confidence, accuracy),
        "score_mae": float(np.mean(score_mae)) if score_mae else None,
        "within_one_level": float(np.mean(within_one)) if within_one else None,
    }
