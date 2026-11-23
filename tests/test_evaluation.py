import copy
import numpy as np
import pytest
from fedsight.evaluation import synthetic_dataset, validate_dataset, train_compare, predict_artifact, binary_metrics, retrieval_metrics
from fedsight.research import federated_simulation


def test_split_leakage_checks_both_identities_and_sequences():
    rows = synthetic_dataset(pairs_per_split=10)
    assert validate_dataset(rows)["train"] == [5, 5]
    for field in ("identity_a", "sequence_b", "case_id"):
        bad = copy.deepcopy(rows)
        bad[10][field] = bad[0][field]
        with pytest.raises(ValueError):
            validate_dataset(bad)


def test_model_comparison_numeric_roundtrip():
    import json
    rows = synthetic_dataset(pairs_per_split=20)
    artifact, report = train_compare(rows)
    assert artifact["synthetic"] and report["held_out_test"]["n"] == 20
    values = [r["features"] for r in rows[:5]]
    assert np.allclose(predict_artifact(artifact, values), predict_artifact(json.loads(json.dumps(artifact)), values))
    assert len(report["comparisons"]) == 3


def test_metrics_and_federated_simulation():
    result = binary_metrics([0, 1, 0, 1], [.1, .9, .7, .3])
    assert result["tp"] == result["fp"] == result["tn"] == result["fn"] == 1
    metrics = retrieval_metrics([{"ranked": ["b", "a"], "relevant": ["a"]}, {"ranked": [], "relevant": []}], k=2)
    assert metrics["rank_1"] == 0 and metrics["mAP"] == .5 and metrics["unknown_false_accept_rate"] == 0
    clients = [(np.array([[0], [1], [2], [3]]), np.array([0, 0, 1, 1]))] * 3
    simulation = federated_simulation(clients)
    assert simulation["rounds"][-1]["weighted_training_loss"] < simulation["rounds"][0]["weighted_training_loss"]
