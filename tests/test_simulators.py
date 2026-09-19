from collections import Counter

from openjev.simulators import generate_dataset


def test_simulator_splits_and_probs():
    rows=generate_dataset(seed=1,seen_per_domain=20,ood_count=10)
    counts=Counter(x.split for x in rows)
    assert counts["train"] > 0 and counts["test"] > 0 and counts["ood"]==10
    assert all(abs(sum(x.target_probs)-1)<1e-8 for x in rows)
    assert {x.domain for x in rows if x.split=="ood"}=={"equipment"}
