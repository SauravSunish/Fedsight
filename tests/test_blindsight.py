import dataclasses
import io
import json
import zipfile

import pytest

from fedsight.blindsight import infer_hypothesis, validate_region, validate_portal
from fedsight.core import TimeAnchor
from fedsight.demo import create_demo
from fedsight.reports import export_bundle
from fedsight.service import FedSight


def scenario():
    anchor = dataclasses.asdict(TimeAnchor("2026-01-01T10:00:00Z", "UTC", 0, 2, 0, "clock check"))
    sources = {"sa": {"camera_id": "A", "time_anchor": anchor}, "sb": {"camera_id": "B", "time_anchor": anchor}}
    tracks = {"a": {"id": "a", "source_id": "sa", "camera_id": "A", "start_time": 40., "end_time": 60.,
                    "motion": {"exit_xy_norm": [.95, .5]}},
              "b": {"id": "b", "source_id": "sb", "camera_id": "B", "start_time": 180., "end_time": 200.,
                    "motion": {"entry_xy_norm": [.05, .5]}}}
    proposal = {"id": "p", "left_id": "a", "right_id": "b", "features": {"values": {"similarity": .72}}}
    transitions = [{"from_camera": "A", "to_camera": "B", "min_s": 90., "max_s": 150.,
                    "provenance": "walk test", "confidence": .8}]
    regions = [{"id": "z", "kind": "blind", "label": "Corridor", "box": [.2, .2, .8, .8],
                "distance_m": 120., "provenance": "floorplan measurement"}]
    portals = [{"id": "pa", "blind_zone_id": "z", "camera_id": "A", "role": "exit", "box": [.8, .2, 1, .8]},
               {"id": "pb", "blind_zone_id": "z", "camera_id": "B", "role": "entry", "box": [0, .2, .2, .8]}]
    return proposal, tracks, sources, transitions, regions, portals


def test_hypothesis_requires_ordered_time_and_mapped_portals():
    p, tracks, sources, transitions, regions, portals = scenario()
    h = infer_hypothesis(p, tracks, sources, transitions, regions, portals, {})
    assert h["assessment"] == "possible_unobserved_movement"
    assert h["gap_duration_s"] == [116, 124]
    assert h["zones"][0]["walking_time_compatible"] is True
    assert h["meaning"].startswith("Hypothesis only")
    assert h["status"] == "requires_review"
    sources["sb"]["time_anchor"] = {}
    unknown = infer_hypothesis(p, tracks, sources, transitions, regions, portals, {})
    assert unknown["assessment"] == "time_unknown"
    assert unknown["gap_utc"] is None


def test_hypothesis_records_rejected_link_and_timing_conflict():
    p, tracks, sources, transitions, regions, portals = scenario()
    transitions[0]["min_s"], transitions[0]["max_s"] = 200., 300.
    regions[0]["distance_m"] = 600.
    h = infer_hypothesis(p, tracks, sources, transitions, regions, portals,
                         {"p": {"decision": "reject"}})
    assert h["assessment"] == "unverified_route"
    assert any("rejected" in text for text in h["contradictory_evidence"])
    assert any("travel times conflict" in text for text in h["contradictory_evidence"])
    assert h["zones"][0]["walking_time_compatible"] is False


def test_region_and_portal_validation():
    with pytest.raises(ValueError):
        validate_region({"kind": "blind", "label": "x", "box": [.8, 0, .2, 1], "provenance": "map"})
    with pytest.raises(ValueError):
        validate_portal({"camera_id": "A", "blind_zone_id": "z", "role": "teleport",
                         "box": [0, 0, 1, 1], "provenance": "map"})


def test_blindsight_records_are_audited_and_exported(tmp_path):
    service = FedSight(tmp_path)
    case_id = create_demo(service)
    camera = service.repo.list(case_id, "camera")[0]
    region = service.add_coverage_region(case_id, "tester", kind="blind", label="Hallway",
                                         box=[.2, .2, .8, .8], distance_m=10., provenance="marked plan")
    service.add_blind_portal(case_id, "tester", camera_id=camera["id"], blind_zone_id=region["id"],
                             role="both", box=[0, .2, .2, .8], provenance="frame inspection")
    source = service.repo.list(case_id, "source")[0]
    alignment = service.set_blind_alignment(case_id, "tester", source_id=source["id"], camera_id=camera["id"],
                                             time_anchor=source["time_anchor"], provenance="checked demo clock")
    service.rank(case_id, "tester")
    assert alignment["id"] in service.repo.list(case_id, "run")[-1]["alignment_ids"]
    generated = service.generate_blind_hypotheses(case_id, "tester")
    assert generated
    assert alignment["id"] in generated[0]["alignment_ids"]
    assert len(service.active_blind_hypotheses(case_id)) == len(generated)
    service.add_blind_clue(case_id, "tester", hypothesis_id=generated[0]["id"], source_id=source["id"],
                           clip_time_s=12., category="door", stance="context_only",
                           description="Door appears open", provenance="manual review")
    service.review_blind_hypothesis(case_id, "tester", generated[0]["id"], "defer", "Need clock alignment")
    assert service.repo.verify()["ok"]
    with zipfile.ZipFile(io.BytesIO(export_bundle(service, case_id, "tester"))) as archive:
        snapshot = json.loads(archive.read("case.json"))
        assert snapshot["blind_hypotheses"] and snapshot["blind_clues"] and snapshot["blind_reviews"] and snapshot["source_alignments"]
        assert "blind-hypotheses.csv" in archive.namelist()


def test_blindsight_page_renders_configured_hypotheses(tmp_path, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("FEDSIGHT_DATA_DIR", str(tmp_path))
    service = FedSight(tmp_path)
    case_id = create_demo(service)
    camera = service.repo.list(case_id, "camera")[0]
    region = service.add_coverage_region(case_id, "tester", kind="blind", label="Hallway",
                                         box=[.2, .2, .8, .8], distance_m=10., provenance="marked plan")
    service.add_blind_portal(case_id, "tester", camera_id=camera["id"], blind_zone_id=region["id"],
                             role="both", box=[0, .2, .2, .8], provenance="frame inspection")
    service.generate_blind_hypotheses(case_id, "tester")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=60).run()
    app.selectbox(key="global_cmd_palette").set_value("BlindSight").run()
    assert app.title[0].value == "BlindSight"
    assert not app.exception
