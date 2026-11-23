"""Application service: durable multi-source processing and provenance."""
from __future__ import annotations

import dataclasses
import importlib.metadata
import io
import json
import os
import tempfile
from contextlib import ExitStack
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

from .core import VERSION, TimeAnchor, RejectionPolicy, canonical, digest, uid, now
from .repository import Repository
from .security import Keyring, EvidenceVault
from .quality import assess_track, appearance_attributes
from .pair_features import pair_features, rejection
from .matching import RankConfig, retrieve_and_rank
from .rules import evaluate_rules, evaluate_scene_rules


class FedSight:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.keys = Keyring(Path(os.environ.get("FEDSIGHT_KEY_DIR", self.root / "keys")))
        self.repo = Repository(self.root / "cases.sqlite3", self.keys)
        self.vault = EvidenceVault(self.root / "vault", self.keys)
        self.work = self.root / "working"
        self.work.mkdir(parents=True, exist_ok=True)

    def read_asset(self, asset, actor="local-reviewer"):
        try:
            return self.vault.read(asset)
        except (ValueError, OSError):
            self.repo.audit(asset["case_id"], actor, "asset.authentication_failed", asset["asset_id"], asset["sha256"])
            raise

    def save_asset(self, data, case_id, actor, **metadata):
        asset = self.vault.put(data, case_id, uid("asset"))
        try:
            self.repo.put("asset", {**metadata, **asset, "id": asset["asset_id"]}, actor)
        except Exception:
            self.vault.remove(case_id, asset["asset_id"], asset["sha256"])
            raise
        return asset

    def delete_case(self, case_id, actor):
        from .processing_queue import QUEUE as processing
        from .render_queue import QUEUE as rendering
        from .semantic_search import QUEUE as indexing
        from .local_vision import QUEUE as vision
        scope = (str(self.root.resolve()), case_id)
        with ExitStack() as locks:
            for queue in (processing, rendering, indexing, vision):
                locks.enter_context(queue.removing_case(scope))
            self.repo.delete_case(case_id, actor)
        return self.clean_deleted_assets()

    def clean_deleted_assets(self):
        """Resume committed file removals after a crash or a temporary file lock."""
        pending = 0
        with self.repo.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for row in conn.execute("SELECT * FROM pending_asset_removal").fetchall():
                try:
                    self.vault.remove(row["case_id"], row["asset_id"], row["sha256"])
                except OSError:
                    pending += 1
                    continue
                conn.execute("DELETE FROM pending_asset_removal WHERE case_id=? AND asset_id=?",
                             (row["case_id"], row["asset_id"]))
        return pending

    def ingest(self, case_id, filename, blob, actor, *, camera_id=None, time_anchor=None):
        case = self.repo.get(case_id, case_id, "case")
        if case["demo"]:
            raise ValueError("Use a separate case for real footage; this case is synthetic.")
        if case["status"] not in {"Created", "Under Review", "Failed"}:
            raise ValueError("Reopen this case before adding evidence.")
        suffix = Path(filename).suffix.lower()
        if suffix not in {".mp4", ".avi", ".mov", ".mkv"} or not 0 < len(blob) <= 200 * 1024 * 1024:
            raise ValueError("Use a supported video up to 200 MB.")
        if camera_id:
            self.repo.get(case_id, camera_id, "camera")
        anchor = dataclasses.asdict(TimeAnchor(**(time_anchor or {})))
        sha = digest(blob)
        for source in self.repo.list(case_id, "source"):
            if source["sha256"] == sha:
                self.repo.audit(case_id, actor, "source.duplicate_ignored", source["id"], sha)
                return source
        if len(self.repo.list(case_id, "source")) >= 32:
            raise ValueError("This local workbench limits a case to 32 recordings.")
        source_id = uid("source")
        asset = self.save_asset(blob, case_id, actor, role="original", parent_id=None)
        return self.repo.put("source", {"id": source_id, "case_id": case_id, "name": Path(filename).name,
                              "sha256": sha, "asset": asset, "camera_id": camera_id, "time_anchor": anchor,
                              "suffix": suffix, "demo": False}, actor)

    def register_camera(self, case_id, label, actor, **metadata):
        if not label.strip():
            raise ValueError("Camera label is required.")
        return self.repo.put("camera", {**metadata, "case_id": case_id, "label": label.strip()}, actor)

    def add_transition(self, case_id, actor, **edge):
        from .cameras import validate_transition
        validate_transition(edge)
        self.repo.get(case_id, edge["from_camera"], "camera")
        self.repo.get(case_id, edge["to_camera"], "camera")
        return self.repo.put("transition", {**edge, "case_id": case_id}, actor)

    def persist_tracks(self, case_id, source, tracks, actor, job_id, meta):
        saved = []
        for track in tracks:
            value = dict(track)
            value["quality_report"] = assess_track(value) if "crops_bgr" in value else value.get("quality_report", {})
            preview = value.pop("preview_rgb", None)
            value.pop("crops_bgr", None)
            if preview is not None:
                output = io.BytesIO()
                Image.fromarray(np.asarray(preview, dtype=np.uint8)).save(output, format="PNG")
                asset = self.save_asset(output.getvalue(), case_id, actor, role="preview",
                                        parent_id=source.get("asset", {}).get("asset_id"),
                                        transformation="representative track crop", source_id=source["id"],
                                        start_time=value.get("start_time"), end_time=value.get("end_time"))
                value["preview_asset"] = asset
                value["attributes"] = appearance_attributes(preview)
                if value.get("object_class", "person") == "person" and value.get("bag_observation_samples", 0):
                    value["attributes"]["bag"] = "possible nearby bag"
            value.update(id=uid("track"), case_id=case_id, source_id=source["id"], source_key=source["id"],
                         camera_id=source.get("camera_id"), job_id=job_id, frame_size=meta.get("frame_size", [1, 1]))
            value["tracker_id"] = value["track_id"]
            value["track_id"] = value["id"]
            saved.append(self.repo.put("track", value, actor))
        return saved

    def active_tracks(self, case_id):
        jobs = self.repo.list(case_id, "job")
        latest = {}
        for job in jobs:
            if job["status"] == "completed":
                latest[job["source_id"]] = job["id"]
        return [t for t in self.repo.list(case_id, "track") if latest.get(t["source_id"]) == t["job_id"]]

    def process(self, case_id, actor, video_config, embedding_config, *, progress=None, force=False,
                should_stop=None, stage=None):
        from .video import analyze_video
        from .embeddings import build_osnet_embeddings
        video_config.validate()
        self.repo.transition(case_id, "Processing", actor)
        sources = self.repo.list(case_id, "source")
        if not sources:
            self.repo.transition(case_id, "Failed", actor)
            raise ValueError("Ingest at least one video first.")
        latest_jobs = {j["source_id"]: j for j in self.repo.list(case_id, "job")}
        completed = set() if force else {source_id for source_id, job in latest_jobs.items() if job["status"] == "completed"}
        failed = []
        cancelled = False
        for index, source in enumerate(sources):
            if source["id"] in completed:
                continue
            if should_stop and should_stop():
                cancelled = True
                break
            job = self.repo.put("job", {"case_id": case_id, "source_id": source["id"], "status": "running",
                                       "video_config": dataclasses.asdict(video_config), "embedding_config": dataclasses.asdict(embedding_config)}, actor)
            path = None
            try:
                blob = self.read_asset(source["asset"], actor)
                with tempfile.NamedTemporaryFile(dir=self.work, suffix=source["suffix"], delete=False) as tmp:
                    tmp.write(blob)
                    path = Path(tmp.name)
                os.chmod(path, 0o600)
                callback = (lambda done, total: progress(index, len(sources), done, total, source["name"])) if progress else None
                if stage:
                    stage("Detect and track", source["name"])
                tracks, meta = analyze_video(path, source_name=source["name"], source_key=source["id"], config=video_config, progress_callback=callback, should_stop=should_stop)
                if meta.get("stop_reason") == "cancelled" or (should_stop and should_stop()):
                    cancelled = True
                    job.update(status="cancelled", meta=meta, finished_at=now())
                    break
                if stage:
                    stage("Assess attributes", source["name"])
                for track in tracks:
                    track["quality_report"] = assess_track(track)
                note = "No usable tracks."
                if tracks:
                    people = [t for t in tracks if t.get("object_class", "person") == "person"]
                    vehicles = [t for t in tracks if t.get("object_class") == "vehicle"]
                    if stage:
                        stage("Build appearance features", source["name"])
                    if people:
                        _, note = build_osnet_embeddings(people, config=embedding_config)
                    if vehicles:
                        from .vehicle import describe_vehicles
                        describe_vehicles(vehicles)
                        note += f" {len(vehicles)} vehicles use colour/shape descriptors, not vehicle Re-ID."
                    bags = [t for t in tracks if t.get("object_class") == "bag"]
                    if bags:
                        note += f" {len(bags)} bag tracklets indexed without appearance embeddings."
                if should_stop and should_stop():
                    cancelled = True
                    job.update(status="cancelled", finished_at=now())
                    break
                if stage:
                    stage("Preserve indexed tracks", source["name"])
                self.persist_tracks(case_id, source, tracks, actor, job["id"], meta)
                weights = Path(video_config.weights)
                model_manifest = {"software": VERSION, "detector": video_config.weights,
                                  "detector_sha256": digest(weights.read_bytes()) if weights.is_file() else None,
                                  "reid": embedding_config.model_name, "reid_sha256": digest(Path(embedding_config.model_path).read_bytes()) if embedding_config.model_path else "imagenet-default",
                                  "packages": environment_versions()}
                job.update(status="completed", meta=meta, embedding_note=note, model_manifest=model_manifest, finished_at=now())
            except Exception as exc:
                job.update(status="failed", error=f"{type(exc).__name__}: {str(exc)[:400]}", finished_at=now())
                failed.append(source["id"])
            finally:
                if path is not None:
                    path.unlink(missing_ok=True)
                self.repo.put("job", job, actor, replace=True)
        self.repo.transition(case_id, "Under Review" if not failed and not cancelled else "Failed", actor)
        clip_note = None
        if not failed and not cancelled:
            clip_weights = os.environ.get("FEDSIGHT_CLIP_WEIGHTS")
            if clip_weights:
                try:
                    from .semantic_search import get_model, index_case
                    if stage:
                        stage("Index text-image previews", "")
                    clip_note = index_case(self, case_id, actor, get_model(clip_weights))
                except Exception as exc:
                    clip_note = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            if stage:
                stage("Rank cross-camera candidates", "")
            self.rank(case_id, actor)
        return {"failed_sources": failed, "cancelled": cancelled, "tracks": len(self.active_tracks(case_id)), "clip_index": clip_note}

    def rank(self, case_id, actor, *, top_k=5, policy=None, rank_config=None, classifier=None):
        policy = policy or RejectionPolicy()
        rank_config = rank_config or RankConfig()
        case = self.repo.get(case_id, case_id, "case")
        if case["status"] not in {"Under Review", "Reviewed"}:
            raise ValueError("Finish processing before generating proposals.")
        if classifier is not None and bool(classifier.get("synthetic")) != bool(case["demo"]):
            raise ValueError("Synthetic classifiers and real cases must remain separate.")
        if case["status"] == "Reviewed":
            self.repo.transition(case_id, "Under Review", actor)
        tracks = self.active_tracks(case_id)
        sources = {s["id"]: s for s in self.repo.list(case_id, "source")}
        alignments = self.blind_alignments(case_id)
        for source_id, alignment in alignments.items():
            sources[source_id] = {**sources[source_id], "camera_id": alignment["camera_id"],
                                  "time_anchor": alignment["time_anchor"]}
        transitions = self.repo.list(case_id, "transition")
        groups = {}
        for track in tracks:
            if track["source_id"] in alignments:
                track["camera_id"] = alignments[track["source_id"]]["camera_id"]
            anchor = TimeAnchor(**sources[track["source_id"]].get("time_anchor", {}))
            track["utc_start_bounds"] = anchor.interval(track["start_time"])
            track["utc_end_bounds"] = anchor.interval(track["end_time"])
            groups.setdefault((track["source_id"], track.get("object_class", "person")), []).append(track)
        run_id, proposals, backends = uid("run"), [], []
        for (source_a, class_a), (source_b, class_b) in combinations(groups, 2):
            if source_a == source_b or class_a != class_b:
                continue
            a, b = groups[(source_a, class_a)], groups[(source_b, class_b)]
            # Rerank the retrieval pool before truncating to the requested shortlist.
            pool_size = max(top_k, rank_config.pool) if classifier is not None else top_k
            pairs, backend = retrieve_and_rank(a, b, top_k=pool_size, config=rank_config)
            backends.append(backend)
            scored = []
            for pair in pairs:
                features = pair_features(pair, sources, transitions)
                switch = any(t.get("embedding_meta", {}).get("possible_id_switch", False) for t in (pair["track_a"], pair["track_b"]))
                verdict = rejection(features, policy, possible_id_switch=switch)
                if class_a == "vehicle":
                    verdict = {"state": "uncertain", "reasons": ["Vehicle colour/shape similarity is an untrained visual search aid; review original footage."]}
                probability = None
                if classifier is not None and class_a == "person":
                    from .evaluation import predict_artifact
                    probability = float(predict_artifact(classifier, [features["values"]])[0])
                    if probability < classifier["threshold"]:
                        verdict = {"state": "no_reliable_candidate", "reasons": ["Below frozen calibrated-classifier threshold."]}
                scored.append({"case_id": case_id, "run_id": run_id,
                    "left_id": pair["track_a"]["id"], "right_id": pair["track_b"]["id"],
                    "features": features, "verdict": verdict, "probability": probability,
                    "classifier_id": classifier.get("id") if classifier else None,
                    "ranking": {k: v for k, v in pair.items() if k not in {"track_a", "track_b"}},
                    "model_version": VERSION, "demo": case["demo"]})
            scored.sort(key=lambda p: (-(p["probability"] if p["probability"] is not None else p["ranking"]["score"]), p["left_id"], p["right_id"]))
            kept = {}
            for candidate in scored:
                left = candidate["left_id"]
                if kept.get(left, 0) >= top_k:
                    continue
                kept[left] = kept.get(left, 0) + 1
                candidate["ranking"]["final_rank_in_source_pair"] = kept[left]
                proposals.append(self.repo.put("proposal", candidate, actor))
        states = {t["id"]: [] for t in tracks}
        for p in proposals:
            for key in ("left_id", "right_id"):
                states[p[key]].append(p["verdict"]["state"])
        outcomes = {tid: ("review_priority" if "review_priority" in values else "uncertain" if "uncertain" in values else "no_reliable_candidate") for tid, values in states.items()}
        run = {"id": run_id, "case_id": case_id, "top_k": top_k, "policy": dataclasses.asdict(policy),
               "rank_config": dataclasses.asdict(rank_config), "proposal_count": len(proposals), "track_ids": [t["id"] for t in tracks],
               "transition_ids": [t["id"] for t in transitions], "backends": sorted(set(backends)), "software": VERSION,
               "alignment_ids": [a["id"] for a in alignments.values()],
               "classifier_id": classifier.get("id") if classifier else None, "track_outcomes": outcomes}
        self.repo.put("run", run, actor)  # Commit visibility only after all proposals exist.
        return proposals

    def run_rules(self, case_id, actor):
        rules = self.repo.list(case_id, "rule")
        states = {state["rule_id"]: state for state in self.repo.list(case_id, "rule_state")}
        rules = [rule for rule in rules if states.get(rule["id"], {}).get("enabled", True)]
        existing = {(e["rule_id"], e["track_id"]) for e in self.repo.list(case_id, "event")}
        tracks = self.active_tracks(case_id)
        for track in tracks:
            for event in evaluate_rules(track, rules, track["frame_size"]):
                if (event["rule_id"], event["track_id"]) not in existing:
                    self.repo.put("event", {**event, "case_id": case_id}, actor)
        scene_existing = {(e["rule_id"], e["source_id"]) for e in self.repo.list(case_id, "event") if e["method"] in {"crowd", "count"}}
        for event in evaluate_scene_rules(tracks, rules):
            if event["method"] == "unattended_object" and (event["rule_id"], event["track_id"]) in existing:
                continue
            if event["method"] == "unattended_object" or (event["rule_id"], event["source_id"]) not in scene_existing:
                self.repo.put("event", {**event, "case_id": case_id}, actor)

    def add_coverage_region(self, case_id, actor, **region):
        from .blindsight import validate_region
        validate_region(region)
        if region.get("camera_id"):
            self.repo.get(case_id, region["camera_id"], "camera")
        return self.repo.put("coverage_region", {**region, "case_id": case_id}, actor)

    def add_blind_portal(self, case_id, actor, **portal):
        from .blindsight import validate_portal
        validate_portal(portal)
        self.repo.get(case_id, portal["camera_id"], "camera")
        zone = self.repo.get(case_id, portal["blind_zone_id"], "coverage_region")
        if zone["kind"] != "blind":
            raise ValueError("Portal destination must be a blind region.")
        return self.repo.put("blind_portal", {**portal, "case_id": case_id}, actor)

    def set_blind_alignment(self, case_id, actor, *, source_id, camera_id, time_anchor, provenance):
        self.repo.get(case_id, source_id, "source")
        self.repo.get(case_id, camera_id, "camera")
        if not provenance.strip():
            raise ValueError("Source alignment provenance is required.")
        anchor = dataclasses.asdict(TimeAnchor(**time_anchor))
        return self.repo.put("source_alignment", {"case_id": case_id, "source_id": source_id,
                                                   "camera_id": camera_id, "time_anchor": anchor,
                                                   "provenance": provenance.strip(), "actor": actor}, actor)

    def blind_alignments(self, case_id):
        return {a["source_id"]: a for a in self.repo.list(case_id, "source_alignment")}

    def generate_blind_hypotheses(self, case_id, actor, *, min_speed_mps=.5, max_speed_mps=2.0):
        from .blindsight import infer_hypothesis
        proposals = self.repo.active_proposals(case_id)[:200]
        tracks = {t["id"]: t for t in self.active_tracks(case_id)}
        sources = {s["id"]: s for s in self.repo.list(case_id, "source")}
        alignments = self.blind_alignments(case_id)
        for source_id, alignment in alignments.items():
            sources[source_id] = {**sources[source_id], "camera_id": alignment["camera_id"],
                                  "time_anchor": alignment["time_anchor"]}
        for track in tracks.values():
            if track["source_id"] in alignments:
                track["camera_id"] = alignments[track["source_id"]]["camera_id"]
        transitions = self.repo.list(case_id, "transition")
        regions = self.repo.list(case_id, "coverage_region")
        portals = self.repo.list(case_id, "blind_portal")
        reviews = self.repo.latest_reviews(case_id)
        run_id = uid("blind_run")
        saved = []
        for proposal in proposals:
            if proposal["left_id"] not in tracks or proposal["right_id"] not in tracks:
                continue
            if tracks[proposal["left_id"]].get("object_class", "person") != "person":
                continue
            competing = [p["id"] for p in proposals if p["id"] != proposal["id"]
                         and ({p["left_id"], p["right_id"]} & {proposal["left_id"], proposal["right_id"]})
                         and p.get("ranking", {}).get("score", 0) >= proposal.get("ranking", {}).get("score", 0) - .08]
            hypothesis = infer_hypothesis(proposal, tracks, sources, transitions, regions, portals, reviews,
                                          min_speed_mps=min_speed_mps, max_speed_mps=max_speed_mps,
                                          competing_proposals=competing)
            saved.append(self.repo.put("blind_hypothesis", {**hypothesis, "case_id": case_id, "run_id": run_id,
                                                              "competing_proposal_ids": competing,
                                                              "source_proposal_run_id": proposal["run_id"],
                                                              "alignment_ids": [a["id"] for a in alignments.values()],
                                                              "region_ids": [r["id"] for r in regions],
                                                              "portal_ids": [p["id"] for p in portals],
                                                              "transition_ids": [t["id"] for t in transitions],
                                                              "software": VERSION}, actor))
        self.repo.put("blind_run", {"id": run_id, "case_id": case_id, "hypothesis_ids": [h["id"] for h in saved],
                                    "min_speed_mps": min_speed_mps, "max_speed_mps": max_speed_mps,
                                    "meaning": "Video-only, unobserved route hypotheses; not verified occupancy."}, actor)
        return saved

    def active_blind_hypotheses(self, case_id):
        runs = self.repo.list(case_id, "blind_run")
        if not runs:
            return []
        latest = runs[-1]
        return [self.repo.get(case_id, hid, "blind_hypothesis") for hid in latest["hypothesis_ids"]]

    def add_blind_clue(self, case_id, actor, *, hypothesis_id, source_id, clip_time_s,
                       category, stance, description, provenance):
        self.repo.get(case_id, hypothesis_id, "blind_hypothesis")
        self.repo.get(case_id, source_id, "source")
        if not np.isfinite(clip_time_s) or clip_time_s < 0:
            raise ValueError("Clue time must be a nonnegative clip-relative second.")
        jobs = [j for j in self.repo.list(case_id, "job") if j["source_id"] == source_id and j["status"] == "completed"]
        if jobs:
            meta = jobs[-1].get("meta", {})
            if meta.get("fps") and meta.get("total_frames") and clip_time_s > meta["total_frames"] / meta["fps"] + 1:
                raise ValueError("Clue time exceeds the recording duration.")
        if category not in {"door", "shadow", "reflection", "object_change", "audio", "other"}:
            raise ValueError("Choose a supported visual or audio clue category.")
        if stance not in {"supporting", "contradictory", "context_only"}:
            raise ValueError("Choose how the clue relates to this hypothesis.")
        if not description.strip() or not provenance.strip():
            raise ValueError("Clue description and observation provenance are required.")
        return self.repo.put("blind_clue", {"case_id": case_id, "hypothesis_id": hypothesis_id,
                                             "source_id": source_id, "clip_time_s": clip_time_s,
                                             "category": category, "stance": stance,
                                             "description": description.strip(), "provenance": provenance.strip(),
                                             "actor": actor, "meaning": "Investigator observation; not automatically verified."}, actor)

    def review_blind_hypothesis(self, case_id, actor, hypothesis_id, decision, reason):
        self.repo.get(case_id, hypothesis_id, "blind_hypothesis")
        if decision not in {"defer", "supports_possible_route", "discount_hypothesis"} or not reason.strip():
            raise ValueError("Choose a BlindSight assessment and provide a reason.")
        return self.repo.put("blind_review", {"case_id": case_id, "hypothesis_id": hypothesis_id,
                                              "decision": decision, "reason": reason.strip(), "actor": actor,
                                              "meaning": "Assessment of a route hypothesis, not a confirmed identity or occupancy."}, actor)

    def verify_assets(self, case_id, actor):
        results = []
        for asset in self.repo.list(case_id, "asset"):
            try:
                self.read_asset(asset, actor)
                results.append({"asset_id": asset["asset_id"], "ok": True, "role": asset["role"]})
            except (ValueError, OSError) as exc:
                results.append({"asset_id": asset["asset_id"], "ok": False, "error": str(exc)})
        return results

    def add_floor_calibration(self, case_id, actor, *, source_id, camera_id=None,
                              video_points, floor_points, floor_plan_asset=None,
                              pixels_per_metre=1.0, provenance=""):
        from .spatial import make_calibration_record
        self.repo.get(case_id, source_id, "source")
        if camera_id:
            self.repo.get(case_id, camera_id, "camera")
        record = make_calibration_record(case_id, source_id, camera_id, video_points,
                                         floor_points, floor_plan_asset,
                                         pixels_per_metre=pixels_per_metre,
                                         actor=actor, provenance=provenance)
        return self.repo.put("floor_calibration", record, actor)

    def floor_calibrations(self, case_id):
        return self.repo.list(case_id, "floor_calibration")


def environment_versions():
    result = {}
    for name in ("numpy", "opencv-python", "opencv-python-headless", "torch", "torchvision", "torchreid", "ultralytics", "scikit-learn", "streamlit"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result
