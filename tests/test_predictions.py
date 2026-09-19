from jev48.predictions import fit_temperature_from_probs, temperature_transform


def test_temperature_transform_normalizes():
    q = temperature_transform([0.8, 0.2], 2.0)
    assert abs(sum(q) - 1) < 1e-12
    assert q[0] < 0.8
    assert q[0] > 0.5


def test_temperature_fit_soft_targets_moves_overconfident_probs_up():
    probs = [[0.99, 0.01]] * 20
    targets = [[0.7, 0.3]] * 20
    t = fit_temperature_from_probs(probs, targets)
    assert t > 1.0
