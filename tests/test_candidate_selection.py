# Selection behavior is integration-tested through the CLI in Modal; keep a tiny metric sanity check here.
from jev48.metrics import summarize


def test_soft_preference_brier_distinguishes_probability_quality():
    y = [[0.7, 0.2, 0.1]]
    good = summarize([[0.7, 0.2, 0.1]], y)
    hardish = summarize([[0.98, 0.01, 0.01]], y)
    assert good["brier"] < hardish["brier"]
