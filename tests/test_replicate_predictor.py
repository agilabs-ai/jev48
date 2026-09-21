import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


PREDICTOR_PATH = Path(__file__).parents[1] / "replicate" / "decider-2b" / "predict.py"


@pytest.fixture()
def predictor_module(monkeypatch):
    cog = types.ModuleType("cog")
    cog.BasePredictor = object
    cog.Input = lambda **kwargs: kwargs.get("default")
    torch = types.ModuleType("torch")
    torch.float16 = "float16"
    decider = types.ModuleType("decider")
    infer = types.ModuleType("decider.infer")
    infer.Decider = object
    monkeypatch.setitem(sys.modules, "cog", cog)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "decider", decider)
    monkeypatch.setitem(sys.modules, "decider.infer", infer)

    spec = importlib.util.spec_from_file_location("replicate_decider_predict", PREDICTOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_questions_accepts_jev_types(predictor_module):
    raw = json.dumps(
        {
            "route": {"type": "choice", "instructions": "Where?", "criteria": ["a", "b"]},
            "urgent": {"type": "noul", "instructions": "Urgent?", "criteria": {"true": "today"}},
            "risk": {"type": "score", "instructions": "Risk?", "criteria": ["low", "high"]},
        }
    )
    assert list(predictor_module.parse_questions(raw)) == ["route", "urgent", "risk"]


@pytest.mark.parametrize(
    "raw, message",
    [
        ("not json", "valid JSON"),
        ("{}", "non-empty"),
        (json.dumps({"q": {"type": "choice", "instructions": "Q", "criteria": ["only"]}}), "2 to 255"),
        (json.dumps({"q": {"type": "other", "instructions": "Q"}}), "choice, noul, or score"),
    ],
)
def test_parse_questions_rejects_invalid_payloads(predictor_module, raw, message):
    with pytest.raises(ValueError, match=message):
        predictor_module.parse_questions(raw)


def test_predict_calls_system_one(predictor_module):
    class FakeModel:
        def system_one(self, state, questions, independent):
            return {"state": state, "questions": questions, "independent": independent}

    predictor = predictor_module.Predictor()
    predictor.model = FakeModel()
    questions = json.dumps({"q": {"type": "noul", "instructions": "Safe?", "criteria": {}}})
    result = predictor.predict("hello", questions, False)
    assert result["state"] == "hello"
    assert result["independent"] is False


def test_predict_uses_default_questions_when_input_is_blank(predictor_module):
    class FakeModel:
        def system_one(self, state, questions, independent):
            return {"questions": questions}

    predictor = predictor_module.Predictor()
    predictor.model = FakeModel()
    result = predictor.predict("hello", "", True)
    assert "action" in result["questions"]
