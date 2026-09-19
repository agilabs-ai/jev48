import importlib.util
from pathlib import Path

from openjev.schema import Candidate, DecisionExample


spec=importlib.util.spec_from_file_location("bridge", Path(__file__).parents[1]/"scripts"/"export_nanojev_training_data.py")
bridge=importlib.util.module_from_spec(spec); spec.loader.exec_module(bridge)


def test_known_distribution_maps_to_programmatic_gold():
    ex=DecisionExample(
        id="x", family_id="f", split="train", domain="sim", state="s", question="q",
        candidates=[Candidate(id="a",text="A"),Candidate(id="b",text="B")],
        target_probs=[0.3,0.7],target_kind="known_distribution",
    )
    row=bridge.convert(ex)
    assert row["gold_probs_kind"]["decision"] == "programmatic_conditional_distribution"
    assert row["gold_label_kind"]["decision"] == "unobserved"
    assert "gold" not in row


def test_deterministic_maps_to_nanojev_truth():
    ex=DecisionExample(
        id="x", family_id="f", split="train", domain="real", state="s", question="q",
        candidates=[Candidate(id="a",text="A"),Candidate(id="b",text="B")],
        target_probs=[1.0,0.0],target_kind="deterministic_truth",
    )
    row=bridge.convert(ex)
    assert row["gold"]["decision"] == "a"
    assert row["gold_probs_kind"]["decision"] == "deterministic_truth"
