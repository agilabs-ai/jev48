from dataclasses import dataclass

from jev48.decider_bridge import _flatten_upstream_examples


@dataclass
class Q:
    text: str
    options: list[str]
    gold: int


@dataclass
class E:
    context: str
    qs: list[Q]


def test_decider_bridge_permutes_without_changing_gold_semantics():
    ex = E("ctx", [Q("q", ["zero", "one", "two"], 1)])
    rows = _flatten_upstream_examples("fake", [ex], "ood", 1, "seed")
    row = rows[0]
    assert row.candidates[row.gold_index].text == "one"
    assert row.metadata["candidate_order_randomized"] is True
