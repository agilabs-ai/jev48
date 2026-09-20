import pytest

from jev48.benchmark_metrics import binary_metrics, paired_bootstrap_delta, paired_cluster_bootstrap_delta, wilson_interval


def test_binary_metrics_perfect():
    got = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert got["accuracy"] == 1.0
    assert got["recall"] == 1.0
    assert got["false_positive_rate"] == 0.0
    assert got["auroc"] == 1.0
    assert got["brier"] == pytest.approx(0.025)


def test_wilson_and_paired_delta():
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi
    got = paired_bootstrap_delta([True, True, False, True], [True, False, False, False], samples=1000)
    assert got["delta"] == 0.5
    assert got["ci95"][0] <= got["delta"] <= got["ci95"][1]


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        binary_metrics([], [])
    with pytest.raises(ValueError):
        paired_bootstrap_delta([True], [])


def test_cluster_bootstrap_tracks_clusters():
    got = paired_cluster_bootstrap_delta([True, True, False, False], [False, False, False, False], ["a", "a", "b", "b"], samples=1000)
    assert got["delta"] == 0.5
    assert got["n_clusters"] == 2
