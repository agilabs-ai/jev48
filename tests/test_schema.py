import pytest

from jev48.schema import Candidate, DecisionExample


def make(kind="known_distribution"):
    return DecisionExample(
        id="x", family_id="f", split="train", domain="demo", state="state", question="question?",
        candidates=[Candidate(id="a", text="Alpha"), Candidate(id="b", text="Beta")],
        target_probs=[0.25, 0.75], target_kind=kind,
    )


def test_valid_example_roundtrip():
    ex = make()
    assert DecisionExample.model_validate_json(ex.model_dump_json()) == ex
    assert len(ex.sha256()) == 64


def test_empirical_distribution_supported():
    assert make("empirical_distribution").target_kind == "empirical_distribution"


def test_rejects_bad_distribution():
    with pytest.raises(ValueError):
        DecisionExample.model_validate(make().model_dump() | {"target_probs": [0.4, 0.4]})


def test_deterministic_truth_must_be_one_hot():
    with pytest.raises(ValueError):
        DecisionExample.model_validate(make().model_dump() | {"target_kind": "deterministic_truth"})
