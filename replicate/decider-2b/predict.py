"""Replicate predictor for Mapika/decider-2b."""

from __future__ import annotations

import json
from typing import Any

import torch
from cog import BasePredictor, Input
from decider.infer import Decider


MODEL_PATH = "/weights/decider-2b"
DEFAULT_QUESTIONS = json.dumps(
    {
        "intent": {
            "type": "choice",
            "instructions": "What should happen next?",
            "criteria": {
                "approve": "Safe to proceed automatically",
                "review": "Needs a human decision",
                "reject": "Should not proceed",
            },
        },
        "urgent": {
            "type": "noul",
            "instructions": "Does this need attention today?",
            "criteria": {
                "true": "Delay would materially worsen the outcome",
                "false": "It can safely wait",
            },
        },
    },
    indent=2,
)


def parse_questions(raw: str) -> dict[str, dict[str, Any]]:
    """Parse and validate the Jev-shaped question map used by Decider."""
    try:
        questions = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"questions_json must be valid JSON: {exc.msg}") from exc

    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions_json must be a non-empty JSON object")
    if len(questions) > 32:
        raise ValueError("A request can contain at most 32 questions")

    for key, question in questions.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Every question must have a non-empty string ID")
        if not isinstance(question, dict):
            raise ValueError(f"Question {key!r} must be an object")
        question_type = question.get("type")
        if question_type not in {"choice", "noul", "score"}:
            raise ValueError(f"Question {key!r} must have type choice, noul, or score")
        instructions = question.get("instructions", question.get("question"))
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError(f"Question {key!r} needs non-empty instructions")
        criteria = question.get("criteria", question.get("options"))
        if question_type == "choice":
            if isinstance(criteria, list):
                valid = 2 <= len(criteria) <= 255 and all(isinstance(item, str) and item.strip() for item in criteria)
            else:
                valid = isinstance(criteria, dict) and 2 <= len(criteria) <= 255 and all(
                    isinstance(name, str) and name.strip() for name in criteria
                )
            if not valid:
                raise ValueError(f"Choice question {key!r} needs 2 to 255 criteria")
        elif question_type == "score":
            if isinstance(criteria, dict):
                values = list(criteria.values())
            else:
                values = criteria
            if not isinstance(values, list) or not 2 <= len(values) <= 10:
                raise ValueError(f"Score question {key!r} needs 2 to 10 ordered criteria")
        elif criteria is not None and not isinstance(criteria, dict):
            raise ValueError(f"Noul question {key!r} criteria must be an object")
    return questions


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the pinned checkpoint once per warm worker."""
        self.model = Decider(
            MODEL_PATH,
            device="cuda",
            dtype=torch.float16,
            use_graphs=False,
        )

    def predict(
        self,
        state: str = Input(
            description="Text or JSON-like context to evaluate.",
            default="A customer says their card was charged twice and asks for an immediate refund.",
        ),
        questions_json: str = Input(
            description=(
                "A JSON object of Jev-shaped questions keyed by ID. Supported types: "
                "choice, noul, and score."
            ),
            default=DEFAULT_QUESTIONS,
        ),
        independent: bool = Input(
            description="Score each question independently so questions cannot influence one another.",
            default=True,
        ),
    ) -> dict[str, Any]:
        if not state.strip():
            raise ValueError("state must not be empty")
        if len(state) > 100_000:
            raise ValueError("state must be at most 100,000 characters")

        questions = parse_questions(questions_json)
        return self.model.system_one(state, questions, independent=independent)
