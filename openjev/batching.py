from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence
import torch

from .formatting import paths_for_example, paths_for_request
from .schema import DecisionExample, DecisionRequest


@dataclass
class EncodedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    group_sizes: list[int]
    targets: torch.Tensor | None
    candidate_ids: list[list[str]]
    example_ids: list[str]

    def to(self, device: str) -> "EncodedBatch":
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        if self.targets is not None:
            self.targets = self.targets.to(device)
        return self


def _tokenize(tokenizer, paths: list[str], max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(paths, padding=True, truncation=False, return_tensors="pt", add_special_tokens=True)
    attention = encoded["attention_mask"]
    lengths = attention.sum(-1)
    if int(lengths.max()) > max_length:
        raise ValueError(f"candidate path has {int(lengths.max())} tokens > max_length={max_length}; no silent truncation")
    return encoded["input_ids"], attention


def collate_examples(examples: Sequence[DecisionExample], tokenizer, max_length: int,
                     permute_candidates: bool = False, rng: random.Random | None = None) -> EncodedBatch:
    rng = rng or random
    flat_paths: list[str] = []
    group_sizes: list[int] = []
    candidate_ids: list[list[str]] = []
    target_rows: list[list[float]] = []
    ex_ids: list[str] = []
    prepared: list[DecisionExample] = []

    for ex in examples:
        if permute_candidates:
            order = list(range(len(ex.candidates)))
            rng.shuffle(order)
            ex = ex.model_copy(update={
                "candidates": [ex.candidates[i] for i in order],
                "target_probs": [ex.target_probs[i] for i in order],
            })
        prepared.append(ex)
        paths = paths_for_example(ex)
        flat_paths.extend(paths)
        group_sizes.append(len(paths))
        candidate_ids.append([c.id for c in ex.candidates])
        target_rows.append(ex.target_probs)
        ex_ids.append(ex.id)

    input_ids, attention = _tokenize(tokenizer, flat_paths, max_length)
    kmax = max(group_sizes)
    targets = torch.zeros((len(examples), kmax), dtype=torch.float32)
    for i, row in enumerate(target_rows):
        targets[i, :len(row)] = torch.tensor(row, dtype=torch.float32)
    return EncodedBatch(input_ids, attention, group_sizes, targets, candidate_ids, ex_ids)


def collate_requests(requests: Sequence[DecisionRequest], tokenizer, max_length: int) -> EncodedBatch:
    flat_paths: list[str] = []
    group_sizes: list[int] = []
    candidate_ids: list[list[str]] = []
    for req in requests:
        paths = paths_for_request(req)
        flat_paths.extend(paths)
        group_sizes.append(len(paths))
        candidate_ids.append([c.id for c in req.candidates])
    input_ids, attention = _tokenize(tokenizer, flat_paths, max_length)
    return EncodedBatch(input_ids, attention, group_sizes, None, candidate_ids, [str(i) for i in range(len(requests))])
