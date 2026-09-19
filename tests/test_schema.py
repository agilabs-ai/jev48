import pytest

from open_system_one.schema import Candidate, DecisionExample
from open_system_one.formatting import paths_for_example


def make():
    return DecisionExample(
        id="x",family_id="f",split="train",domain="demo",state="state",question="question?",
        candidates=[Candidate(id="a",text="Alpha"),Candidate(id="b",text="Beta")],
        target_probs=[0.25,0.75],target_kind="known_distribution",
    )


def test_valid_example_roundtrip():
    ex=make()
    assert DecisionExample.model_validate_json(ex.model_dump_json()) == ex
    assert len(ex.sha256()) == 64


def test_rejects_bad_distribution():
    with pytest.raises(ValueError):
        DecisionExample.model_validate(make().model_dump() | {"target_probs":[0.4,0.4]})


def test_candidate_ids_not_in_model_text():
    ex=make()
    paths=paths_for_example(ex)
    assert "Alpha" in paths[0]
    assert "<CANDIDATE>\na\n" not in paths[0]
