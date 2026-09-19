from openjev.nanojev import build_nanojev_request, parse_nanojev_response
from openjev.schema import Candidate, DecisionExample


def example():
    return DecisionExample(
        id="x",family_id="f",split="test",domain="d",state="state",question="question",
        candidates=[Candidate(id="a",text="alpha"),Candidate(id="b",text="beta")],
        target_probs=[1.0,0.0],target_kind="deterministic_truth",
    )


def test_nanojev_roundtrip_mapping_preserves_candidate_ids():
    ex=example(); req=build_nanojev_request([ex])
    assert req["states"][0]["questions"]["decision"]["criteria"] == {"a":"alpha","b":"beta"}
    response={"states":[{"id":"x","answers":{"decision":{"probabilities":{"b":0.2,"a":0.8}}}}]}
    assert parse_nanojev_response(response,[ex]) == [[0.8,0.2]]
