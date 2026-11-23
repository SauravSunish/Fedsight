"""Synthetic workflow fixture. Generated vectors are not a Re-ID benchmark."""
from __future__ import annotations
import dataclasses
import numpy as np
from .core import TimeAnchor, uid


def create_demo(service, actor="demo-reviewer"):
    repo = service.repo
    case = repo.create_case("Atrium / corridor · synthetic walkthrough", "Three virtual cameras, synthetic track vectors and illustrated thumbnails. No real people or measured model accuracy.", actor, demo=True, consent="Synthetic images and embeddings; no personal footage.")
    case_id = case["id"]
    cameras = [service.register_camera(case_id, label, actor, zone=label, x=x, y=y, site="Demo site", calibration_version="synthetic-v1")
               for label, x, y in (("Atrium", 0, 0), ("Corridor", 1, 0), ("Exit", 2, 1))]
    for a, b in zip(cameras, cameras[1:]):
        service.add_transition(case_id, actor, from_camera=a["id"], to_camera=b["id"], min_s=10, max_s=90,
                               confidence=.8, provenance="Synthetic layout for workflow demonstration")
    repo.transition(case_id, "Processing", actor)
    rng = np.random.default_rng(17)
    bases = np.eye(16)[:4]
    palette = [(55, 111, 211), (201, 85, 71), (72, 150, 116), (163, 141, 65)]
    for camera_index, camera in enumerate(cameras):
        source_id = uid("source")
        source = repo.put("source", {"id": source_id, "case_id": case_id, "name": f"Synthetic {camera['label']}",
            "sha256": None, "camera_id": camera["id"], "demo": True,
            "time_anchor": dataclasses.asdict(TimeAnchor("2026-01-01T10:00:00+00:00", "UTC", camera_index * 55, 2, 0, "synthetic anchor"))}, actor)
        job = repo.put("job", {"case_id": case_id, "source_id": source_id, "status": "running", "demo": True}, actor)
        tracks = []
        for person in range(3 if camera_index < 2 else 4):
            from PIL import Image, ImageDraw
            image = Image.new("RGB", (160, 240), (230, 235, 241))
            draw = ImageDraw.Draw(image)
            draw.ellipse((62, 30, 98, 66), fill=(155, 163, 177))
            draw.rounded_rectangle((45, 70, 115, 165), 14, fill=palette[person])
            draw.rectangle((51, 160, 75, 218), fill=(52, 64, 83))
            draw.rectangle((85, 160, 109, 218), fill=(52, 64, 83))
            draw.text((12, 8), "SYNTHETIC", fill=(60, 70, 90))
            emb = bases[person] + rng.normal(0, .035, 16)
            emb /= np.linalg.norm(emb)
            frames = np.arange(20) * 10
            x = np.linspace(80, 540, 20) if person != 2 else np.full(20, 300)
            trajectory = np.column_stack([frames, x, np.full(20, 160 + person * 25), np.full(20, 50), np.full(20, 100)])
            tracks.append({"track_id": f"demo_{person}", "source_name": source["name"], "embedding": emb,
                "embeddings": [emb, emb], "quality": [.8, .85], "embedding_meta": {"mean_quality": .82, "consistency": .96, "possible_id_switch": False, "model_name": "synthetic", "n_crops_used": 2},
                "preview_rgb": np.asarray(image), "frame_count": 20, "first_frame": 0, "last_frame": 190,
                "start_time": person * 2, "end_time": 19 + person * 2, "duration_s": 19, "avg_conf": .91,
                "trajectory": trajectory, "motion": {"stationary": person == 2}, "demo_subject": f"synthetic-{person}",
                "quality_report": {"mean_quality": .82, "explanation": "Synthetic quality values."}})
        service.persist_tracks(case_id, source, tracks, actor, job["id"], {"frame_size": [640, 360]})
        job.update(status="completed", meta={"frame_size": [640, 360], "processed_frames": 200, "incomplete": False},
                   embedding_note="Synthetic vectors; no detector or embedding inference performed.", model_manifest={"reid": "synthetic"})
        repo.put("job", job, actor, replace=True)
    repo.put("rule", {"case_id": case_id, "type": "dwell", "seconds": 15, "label": "Stationary track over 15s", "camera_id": None}, actor)
    repo.transition(case_id, "Under Review", actor)
    service.rank(case_id, actor)
    service.run_rules(case_id, actor)
    try:
        from .spatial_demo import setup_demo_spatial
        sources = repo.list(case_id, "source")
        setup_demo_spatial(service, case_id, cameras, sources, actor)
    except Exception:
        pass
    return case_id
