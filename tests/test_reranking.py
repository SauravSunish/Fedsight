import numpy as np
from fedsight.demo import create_demo
from fedsight.service import FedSight


def test_classifier_can_promote_a_candidate_outside_initial_top_one(tmp_path, monkeypatch):
    service = FedSight(tmp_path)
    cid = create_demo(service)
    baseline = service.rank(cid, "tester", top_k=1)
    monkeypatch.setattr("fedsight.evaluation.predict_artifact", lambda artifact, rows: np.array([(1 - row["similarity"]) / 2 for row in rows]))
    reranked = service.rank(cid, "tester", top_k=1, classifier={"synthetic": True, "threshold": .01, "id": "test-only"})
    assert len(baseline) == len(reranked)
    assert {p["right_id"] for p in baseline if p["left_id"] == baseline[0]["left_id"]} != {p["right_id"] for p in reranked if p["left_id"] == baseline[0]["left_id"]}
    assert all(p["ranking"]["final_rank_in_source_pair"] == 1 for p in reranked)
    assert len(service.repo.list(cid, "run")[-1]["track_outcomes"]) == 10
