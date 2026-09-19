from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Protocol

from .schema import DecisionExample


@dataclass
class TeacherJudgment:
    provider: str
    model: str
    probabilities: list[float]
    candidate_order: list[int]
    raw: dict


class Teacher(Protocol):
    def judge(self, example: DecisionExample, order: list[int]) -> TeacherJudgment: ...


def _prompt(example: DecisionExample, order: list[int]) -> str:
    options = "\n".join(f"{j}: {example.candidates[i].text}" for j, i in enumerate(order))
    return (
        "You are labeling a calibrated decision dataset.\n\n"
        f"STATE:\n{example.state}\n\n"
        f"QUESTION:\n{example.question}\n\n"
        f"CANDIDATES:\n{options}\n\n"
        "Return a probability distribution over the listed candidate numbers. "
        "Use uncertainty when the evidence is genuinely ambiguous. The probabilities must sum to 1. "
        "Do not return reasoning."
    )


def _validate_probs(values: list[float], k: int) -> list[float]:
    if len(values) != k or any(not math.isfinite(float(x)) or float(x) < 0 for x in values):
        raise ValueError("invalid teacher probabilities")
    s = sum(float(x) for x in values)
    if s <= 0:
        raise ValueError("teacher probabilities sum to zero")
    return [float(x) / s for x in values]


class OpenAITeacher:
    def __init__(self, model: str, api_key: str | None = None, service_tier: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("pip install -e '.[teachers]'") from exc
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.service_tier = service_tier

    def judge(self, example: DecisionExample, order: list[int]) -> TeacherJudgment:
        k = len(order)
        schema = {
            "type": "object",
            "properties": {
                "probabilities": {
                    "type": "array",
                    "items": {"type": "number", "minimum": 0, "maximum": 1},
                    "minItems": k,
                    "maxItems": k,
                }
            },
            "required": ["probabilities"],
            "additionalProperties": False,
        }
        kwargs = {
            "model": self.model,
            "input": _prompt(example, order),
            "text": {"format": {"type": "json_schema", "name": "decision_distribution", "strict": True, "schema": schema}},
        }
        if self.service_tier:
            kwargs["service_tier"] = self.service_tier
        response = self.client.responses.create(**kwargs)
        obj = json.loads(response.output_text)
        probs = _validate_probs(obj["probabilities"], k)
        return TeacherJudgment(
            "openai", self.model, probs, order,
            {"response_id": getattr(response, "id", None), "usage": str(getattr(response, "usage", ""))},
        )


class AnthropicTeacher:
    def __init__(self, model: str, api_key: str | None = None):
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("pip install -e '.[teachers]'") from exc
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def judge(self, example: DecisionExample, order: list[int]) -> TeacherJudgment:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            temperature=0,
            messages=[{
                "role": "user",
                "content": _prompt(example, order) + '\nReturn JSON only: {"probabilities":[...]}',
            }],
        )
        text = "".join(block.text for block in message.content if getattr(block, "type", None) == "text")
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("teacher did not return JSON")
        obj = json.loads(text[start:end + 1])
        probs = _validate_probs(obj["probabilities"], len(order))
        return TeacherJudgment("anthropic", self.model, probs, order, {"message_id": getattr(message, "id", None)})


def ensemble_judgments(
    example: DecisionExample,
    teachers: list[Teacher],
    samples_each: int = 2,
    seed: int = 17,
) -> tuple[list[float], list[dict]]:
    rng = random.Random(seed)
    mapped: list[list[float]] = []
    audit: list[dict] = []
    for teacher in teachers:
        for _ in range(samples_each):
            order = list(range(len(example.candidates)))
            rng.shuffle(order)
            judgment = teacher.judge(example, order)
            original = [0.0] * len(order)
            for returned_pos, original_pos in enumerate(order):
                original[original_pos] = judgment.probabilities[returned_pos]
            mapped.append(original)
            audit.append({
                "provider": judgment.provider,
                "model": judgment.model,
                "order": order,
                "returned": judgment.probabilities,
                "mapped": original,
                "raw": judgment.raw,
            })
    if not mapped:
        raise ValueError("at least one teacher judgment is required")
    mean = [sum(row[i] for row in mapped) / len(mapped) for i in range(len(mapped[0]))]
    s = sum(mean)
    return [x / s for x in mean], audit
