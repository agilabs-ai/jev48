from openjev.reporting import sliced_summary
from openjev.schema import Candidate, DecisionExample


def ex(i, split, domain, target, k=2):
    return DecisionExample(
        id=f"e{i}", family_id=f"f{i}", split=split, domain=domain,
        state="state", question="question",
        candidates=[Candidate(id=f"c{j}", text=f"candidate {j}") for j in range(k)],
        target_probs=target, target_kind="deterministic_truth",
    )


def test_sliced_summary_seen_ood_domain():
    examples=[
        ex(1,"test","seen",[1.0,0.0]),
        ex(2,"ood","unseen",[0.0,1.0]),
    ]
    probs=[[0.9,0.1],[0.4,0.6]]
    targets=[e.target_probs for e in examples]
    out=sliced_summary(probs,targets,examples)
    assert out["overall"]["accuracy"] == 1.0
    assert out["by_split"]["test"]["n"] == 1
    assert out["by_split"]["ood"]["n"] == 1
    assert set(out["by_domain"]) == {"seen","unseen"}
