from __future__ import annotations

from typing import Any, Literal
import hashlib
import json
import math

from pydantic import BaseModel, Field, field_validator, model_validator


Split = Literal["train", "dev", "calibration", "test", "ood"]
TargetKind = Literal[
    "deterministic_truth",
    "known_distribution",
    "observed_outcome",
    "teacher_distribution",
    "empirical_distribution",
]


class Candidate(BaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class DecisionExample(BaseModel):
    id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    split: Split
    domain: str = Field(min_length=1)
    state: str = Field(min_length=1)
    question: str = Field(min_length=1)
    candidates: list[Candidate] = Field(min_length=2, max_length=255)
    target_probs: list[float]
    target_kind: TargetKind
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("candidates")
    @classmethod
    def unique_candidates(cls, value: list[Candidate]) -> list[Candidate]:
        ids = [c.id for c in value]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate IDs must be unique")
        return value

    @field_validator("target_probs")
    @classmethod
    def valid_probs(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("target_probs cannot be empty")
        if any((not isinstance(x, (int, float))) or isinstance(x, bool) or not math.isfinite(x) or x < 0 or x > 1 for x in value):
            raise ValueError("target probabilities must be finite in [0, 1]")
        if abs(math.fsum(value) - 1.0) > 1e-6:
            raise ValueError(f"target probabilities must sum to 1, got {math.fsum(value)}")
        return [float(x) for x in value]

    @model_validator(mode="after")
    def aligned(self) -> "DecisionExample":
        if len(self.candidates) != len(self.target_probs):
            raise ValueError("candidates and target_probs must have the same length")
        if self.target_kind == "deterministic_truth":
            ones = sum(abs(x - 1.0) < 1e-9 for x in self.target_probs)
            zeros = sum(abs(x) < 1e-9 for x in self.target_probs)
            if ones != 1 or zeros != len(self.target_probs) - 1:
                raise ValueError("deterministic_truth must be one-hot")
        return self

    @property
    def gold_index(self) -> int:
        return max(range(len(self.target_probs)), key=self.target_probs.__getitem__)

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class DecisionRequest(BaseModel):
    state: str = Field(min_length=1)
    question: str = Field(min_length=1)
    candidates: list[Candidate] = Field(min_length=2, max_length=255)


class DecisionResponse(BaseModel):
    best: str
    probabilities: dict[str, float]
    temperature: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)

class ChoiceQuestion(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: str = Field(min_length=1)
    criteria: dict[str, str]

    @field_validator("criteria")
    @classmethod
    def valid_criteria(cls, value: dict[str, str]) -> dict[str, str]:
        if not 2 <= len(value) <= 255:
            raise ValueError("choice requires 2..255 criteria")
        if any(not k or not v for k, v in value.items()):
            raise ValueError("criteria keys and descriptions must be nonempty")
        return value


class SystemOneRequest(BaseModel):
    state: str = Field(min_length=1)
    questions: dict[str, ChoiceQuestion]
    model: str | None = None

    @field_validator("questions")
    @classmethod
    def nonempty_questions(cls, value: dict[str, ChoiceQuestion]) -> dict[str, ChoiceQuestion]:
        if not value:
            raise ValueError("at least one question is required")
        return value


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class SystemOneResponse(BaseModel):
    model: str = "jev48"
    answers: dict[str, ChoiceAnswer]
    usage: dict[str, int | float]
    metadata: dict[str, Any] = Field(default_factory=dict)
