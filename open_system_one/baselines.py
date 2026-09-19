from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence
import torch


@dataclass
class ScoredOptions:
    probabilities: list[float]
    scores: list[float]


def normalized(scores: Sequence[float]) -> list[float]:
    if not scores:
        raise ValueError("scores cannot be empty")
    m = max(scores)
    xs = [math.exp(float(x) - m) for x in scores]
    s = sum(xs)
    return [x / s for x in xs]


class NaiveOptionLikelihoodScorer:
    """Correctness baseline that re-encodes the context for each candidate.

    This is intentionally *not* the optimized prefix-cache path. It gives us a
    simple generative-model likelihood baseline and makes the systems gap easy
    to measure honestly.
    """

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def score(self, context: str, options: Sequence[str], norm: str = "mean") -> ScoredOptions:
        if norm not in {"sum", "mean"}:
            raise ValueError("norm must be sum or mean")
        scores = []
        for option in options:
            prefix_ids = self.tokenizer(context, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
            opt_ids = self.tokenizer(option, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            ids = torch.cat([prefix_ids, opt_ids])[None, :].to(self.device)
            out = self.model(input_ids=ids, use_cache=False)
            logits = out.logits[0]
            start = len(prefix_ids)
            token_scores = []
            for pos in range(start, ids.shape[1]):
                lp = logits[pos - 1].float().log_softmax(-1)
                token_scores.append(float(lp[ids[0, pos]]))
            val = sum(token_scores)
            if norm == "mean" and token_scores:
                val /= len(token_scores)
            scores.append(val)
        return ScoredOptions(normalized(scores), scores)
