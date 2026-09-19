from __future__ import annotations

import pytest

from jev48.jev import JevClient
from jev48.schema import Candidate, DecisionExample


def _example() -> DecisionExample:
    return DecisionExample(
        id="x",
        state="Customer was charged twice.",
        question="Which team should handle this?",
        candidates=[Candidate(id="billing", text="Billing"), Candidate(id="other", text="Other")],
        target_probs=[1.0, 0.0],
        split="test",
        domain="routing",
        family_id="f",
        target_kind="deterministic_truth",
    )


def test_auto_prefers_typesafe(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    c = JevClient(provider="auto")
    try:
        assert c.provider == "typesafe"
        assert c.endpoint == "https://api.typesafe.ai/v1/systemone"
        assert c.model == "jev-latest"
    finally:
        c.close()


def test_auto_falls_back_to_openrouter(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    c = JevClient(provider="auto")
    try:
        assert c.provider == "openrouter"
        assert c.endpoint == "https://openrouter.ai/api/alpha/decisions"
        assert c.model == "typesafe/jev-1.13"
    finally:
        c.close()


def test_missing_key_fails(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError):
        JevClient(provider="auto")


def test_score_preserves_candidate_order_and_normalizes():
    c = JevClient(api_key="test", provider="typesafe")
    try:
        c._post = lambda payload: ({
            "model": "jev-test",
            "answers": {"decision": {"choice": "billing", "probabilities": {"billing": 8, "other": 2}}},
            "usage": {"input_tokens": 12, "output_tokens": 0},
        }, 7.5)
        row = c.score(_example())
        assert row["probabilities"] == [0.8, 0.2]
        assert row["gateway"] == "typesafe"
        assert row["latency_ms"] == 7.5
    finally:
        c.close()
