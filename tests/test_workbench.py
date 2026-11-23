import io
import json
import zipfile
import pytest
from fedsight.service import FedSight
from fedsight.demo import create_demo
from fedsight.graph import evidence_graph, constrained_query
from fedsight.reports import export_bundle, csv_bytes


@pytest.fixture
def demo(tmp_path):
    service = FedSight(tmp_path)
    return service, create_demo(service)


def test_demo_persistence_review_history_and_graph(demo):
    service, cid = demo
    assert len(service.active_tracks(cid)) == 10
    proposals = service.repo.active_proposals(cid)
    assert len(proposals) > 0
    graph, _, _ = evidence_graph(service, cid)
    assert graph.number_of_edges() == 0
    proposal = proposals[0]
    service.repo.review(cid, proposal["id"], "alice", "accept", "source context compared")
    assert evidence_graph(service, cid)[0].number_of_edges() == 1
    service.repo.review(cid, proposal["id"], "bob", "reject", "different appearance after inspection")
    assert evidence_graph(service, cid)[0].number_of_edges() == 0
    assert len(service.repo.list(cid, "review")) == 2
    assert FedSight(service.root).repo.verify()["ok"]
    with pytest.raises(ValueError):
        service.repo.put("proposal", proposal, "alice", replace=True)
    with pytest.raises(ValueError):
        service.repo.transition(cid, "Reviewed", "alice")


def test_reranking_preserves_old_proposals_and_reviews(demo):
    service, cid = demo
    old = service.repo.active_proposals(cid)[0]
    service.repo.review(cid, old["id"], "alice", "defer", "need clearer frame")
    service.rank(cid, "alice")
    assert old["id"] not in {p["id"] for p in service.repo.active_proposals(cid)}
    assert service.repo.get(cid, old["id"])["features"] == old["features"]
    assert len(service.repo.list(cid, "review")) == 1


def test_queries_deny_arbitrary_commands(demo):
    service, cid = demo
    assert len(constrained_query(service, cid, "show sources")) == 3
    with pytest.raises(ValueError):
        constrained_query(service, cid, "MATCH (n) DETACH DELETE n")


def test_bundle_contains_verifiable_provenance(demo):
    service, cid = demo
    blob = export_bundle(service, cid, "alice")
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        assert archive.read("case-report.pdf").startswith(b"%PDF")
        case = json.loads(archive.read("case.json"))
        assert case["audit_verification"]["ok"]
        assert all(x["ok"] for x in case["asset_verification"])
        from fedsight.core import digest
        for name, sha in json.loads(archive.read("manifest.json")).items():
            assert digest(archive.read(name)) == sha
    assert b"'=HYPERLINK" in csv_bytes([{"source": '=HYPERLINK("bad")'}])


def test_rule_events_idempotent(demo):
    service, cid = demo
    first = service.repo.list(cid, "event")
    service.run_rules(cid, "alice")
    assert len(first) == len(service.repo.list(cid, "event")) == 3


def test_disabled_rule_is_skipped_without_removing_existing_events(demo):
    service, cid = demo
    repo = service.repo
    tracks = service.active_tracks(cid)
    rule = repo.put("rule", {"case_id": cid, "type": "count", "label": "new count", "threshold": 1}, "alice")
    repo.put("rule_state", {"case_id": cid, "rule_id": rule["id"], "enabled": False}, "alice")
    original = len(repo.list(cid, "event"))
    service.run_rules(cid, "alice")
    assert len(repo.list(cid, "event")) == original
    repo.put("rule_state", {"case_id": cid, "rule_id": rule["id"], "enabled": True}, "alice")
    service.run_rules(cid, "alice")
    assert any(event["rule_id"] == rule["id"] for event in repo.list(cid, "event"))
