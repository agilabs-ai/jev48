from jev48.typed_decisions import answer_from_choice, normalize_question, score_predictions


def test_question_adapters_cover_all_public_primitives():
    assert normalize_question({"type": "noul", "criteria": {}})[0] == ["false", "true"]
    assert normalize_question({"type": "noul", "instructions": "It is duplicate."})[1]["true"] == "Yes / true"
    score = {"type": "score", "criteria": ["low", "high"]}
    assert answer_from_choice(score, {"0": 0.25, "1": 0.75})["score"] == 0.75


def test_public_scorer_perfect_predictions():
    question = {"type": "choice", "instructions": "pick", "criteria": {"a": "A", "b": "B"}}
    rows = [{"questions": {"q": question}, "predictions": {"q": {"choice": "a", "probabilities": {"a": 0.8, "b": 0.2}}}, "gold": {"q": {"label": "a", "probabilities": {"a": 0.8, "b": 0.2}}}}]
    metrics = score_predictions(rows)
    assert metrics["accuracy"] == 1.0
    assert metrics["brier"] == 0.0
