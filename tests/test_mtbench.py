from jev48.mtbench import aggregate_human_votes, assign_question_splits


def fake_row(qid, winner, a="gpt-4", b="gpt-3.5-turbo", ca="A answer", cb="B answer", turn=1):
    return {
        "question_id": qid,
        "model_a": a,
        "model_b": b,
        "winner": winner,
        "judge": "j",
        "conversation_a": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": ca}],
        "conversation_b": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": cb}],
        "turn": turn,
    }


def test_split_is_outcome_blind_and_complete():
    splits = assign_question_splits(range(81, 161))
    assert list(splits.values()).count("train") == 50
    assert list(splits.values()).count("dev") == 10
    assert list(splits.values()).count("calibration") == 10
    assert list(splits.values()).count("test") == 10


def test_aggregate_swapped_model_order_and_soft_votes():
    rows = [
        fake_row(81, "model_a"),
        fake_row(81, "model_b"),
        fake_row(81, "model_a", a="gpt-3.5-turbo", b="gpt-4", ca="B answer", cb="A answer"),
        fake_row(81, "tie"),
    ]
    # Need 80 IDs for the frozen split function; add one vote for every other ID.
    for qid in range(82, 161):
        rows.append(fake_row(qid, "model_a"))
    exs = aggregate_human_votes(rows)
    group = [e for e in exs if e.metadata["question_id"] == 81][0]
    assert group.target_kind == "empirical_distribution"
    assert group.metadata["vote_count"] == 4
    # 2 canonical votes for gpt-4, 1 for gpt-3.5, 1 tie; display may swap A/B.
    assert sorted(group.target_probs[:2]) == [0.25, 0.5]
    assert group.target_probs[2] == 0.25
    assert "gpt-4" not in group.state
    assert "gpt-3.5" not in group.state


def test_multiline_responses_do_not_break_turn_alignment():
    rows = []
    for qid in range(81, 161):
        rows.append({
            "question_id": qid,
            "model_a": "gpt-4",
            "model_b": "gpt-3.5-turbo",
            "winner": "model_a",
            "judge": "j",
            "conversation_a": [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "line one\nline two\nline three"},
                {"role": "user", "content": "Follow up"},
                {"role": "assistant", "content": "short"},
            ],
            "conversation_b": [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "one line"},
                {"role": "user", "content": "Follow up"},
                {"role": "assistant", "content": "different\nanswer"},
            ],
            "turn": 2,
        })
    ex = aggregate_human_votes(rows)[0]
    # Shared user turns stay shared rather than being accidentally paired with\n    # extra internal response lines.
    assert ex.state.count("USER: Follow up") == 1
    assert "⏎" in ex.state


def test_empty_response_is_preserved_explicitly():
    rows = [fake_row(qid, "model_a", cb="") for qid in range(81, 161)]

    ex = next(e for e in aggregate_human_votes(rows) if e.metadata["question_id"] == 81)

    assert "ASSISTANT: [EMPTY RESPONSE]" in ex.state
