from open_system_one.schema import Candidate,DecisionExample
from open_system_one.teachers import TeacherJudgment,ensemble_judgments


class FakeTeacher:
    def judge(self,example,order):
        # Semantic probability is fixed by original candidate identity, then expressed in returned order.
        semantic=[.1,.7,.2]
        returned=[semantic[i] for i in order]
        return TeacherJudgment("fake","fake",returned,order,{})


def test_ensemble_remaps_permuted_candidates():
    ex=DecisionExample(
        id="x",family_id="x",split="train",domain="x",state="x",question="x?",
        candidates=[Candidate(id="a",text="a"),Candidate(id="b",text="b"),Candidate(id="c",text="c")],
        target_probs=[1,0,0],target_kind="deterministic_truth",
    )
    probs,_=ensemble_judgments(ex,[FakeTeacher()],samples_each=5,seed=1)
    assert all(abs(a-b)<1e-8 for a,b in zip(probs,[.1,.7,.2]))
