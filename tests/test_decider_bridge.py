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


def test_decider_bridge_collapses_exact_source_duplicates_before_limit():
    duplicate = E("same context", [Q("same question", ["no", "yes"], 1)])
    distinct = E("different context", [Q("same question", ["no", "yes"], 0)])

    rows = _flatten_upstream_examples("fake", [duplicate, duplicate, distinct], "ood", 2, "seed")

    assert len(rows) == 2
    assert sorted(row.metadata["upstream_duplicate_count"] for row in rows) == [1, 2]


def test_decider_bridge_preserves_conflicting_duplicates_only_when_allowed():
    negative = E("same context", [Q("same question", ["no", "yes"], 0)])
    positive = E("same context", [Q("same question", ["no", "yes"], 1)])

    rows = _flatten_upstream_examples(
        "fake", [negative, positive], "train", 2, "seed", allow_conflicting_duplicates=True
    )

    assert len({row.id for row in rows}) == 2
    assert {row.gold_index for row in rows} == {0, 1}
    assert all(row.metadata["upstream_conflicting_duplicate"] for row in rows)
