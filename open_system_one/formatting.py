from __future__ import annotations

from .schema import Candidate, DecisionExample, DecisionRequest


FORMAT_VERSION = "oso-path-v1"


def candidate_path(state: str, question: str, candidate: Candidate) -> str:
    """Canonical semantic path seen by the backbone.

    Candidate IDs are transport metadata and are intentionally not included in the
    model text. The candidate description carries the semantics.
    """
    return (
        "<STATE>\n"
        f"{state.strip()}\n"
        "</STATE>\n"
        "<QUESTION>\n"
        f"{question.strip()}\n"
        "</QUESTION>\n"
        "<CANDIDATE>\n"
        f"{candidate.text.strip()}\n"
        "</CANDIDATE>"
    )


def paths_for_example(example: DecisionExample) -> list[str]:
    return [candidate_path(example.state, example.question, c) for c in example.candidates]


def paths_for_request(request: DecisionRequest) -> list[str]:
    return [candidate_path(request.state, request.question, c) for c in request.candidates]
