"""Exercise actual CPU detector/tracker and OSNet code on generated non-person data.

Run `python -m scripts.smoke_vision` from the project root. Never use the random
checkpoint below for matching. This is an offline runtime diagnostic, not a benchmark.
"""
import json
import os
import tempfile
from pathlib import Path


def main():
    import cv2
    import numpy as np
    import torch
    from fedsight.embeddings import OSNetConfig, build_osnet_embeddings
    from fedsight.service import FedSight
    from fedsight.video import AnalyzeConfig
    from torchreid import models

    root = Path(__file__).resolve().parents[1]
    if not (root / "yolo11n.pt").is_file():
        raise RuntimeError("Pre-stage yolo11n.pt in the project before running the offline check.")
    base = root / "tmp" / "vision-checks"
    base.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(17)
    torch.set_num_threads(1)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    with tempfile.TemporaryDirectory(dir=base) as temp:
        where = Path(temp)
        model = models.build_model("osnet_x0_25", num_classes=1, pretrained=False, use_gpu=False)
        weights = where / "synthetic-random-smoke-only.pth"
        torch.save(model.state_dict(), weights)
        cfg = OSNetConfig(model_name="osnet_x0_25", model_path=str(weights), device="cpu")
        rng = np.random.default_rng(17)
        tracks, _ = build_osnet_embeddings([{"track_id": "runtime", "crops_bgr": [rng.integers(0, 255, (96, 48, 3), dtype=np.uint8) for _ in range(3)]}], cfg)
        embedding = tracks[0]["embedding"]
        assert np.isfinite(embedding).all() and np.isclose(np.linalg.norm(embedding), 1)

        clip = where / "geometry.avi"
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
        if not writer.isOpened():
            raise RuntimeError("MJPEG encoder unavailable.")
        for i in range(6):
            frame = np.zeros((240, 320, 3), np.uint8)
            cv2.rectangle(frame, (20 + i * 10, 50), (60 + i * 10, 80), (200, 100, 50), -1)
            writer.write(frame)
        writer.release()
        service = FedSight(where / "workbench")
        case = service.repo.create_case("Generated geometry runtime check", "No people; no accuracy claim.", "smoke")
        cid = case["id"]
        source = service.ingest(cid, clip.name, clip.read_bytes(), "smoke")
        result = service.process(cid, "smoke", AnalyzeConfig(weights="yolo11n.pt", tracker="auto", device="cpu", frame_step=1, imgsz=320, max_processed_frames=6), cfg)
        jobs = service.repo.list(cid, "job")
        if result["failed_sources"]:
            raise RuntimeError(jobs[-1]["error"])
        assert jobs[-1]["meta"]["processed_frames"] == 6
        assert service.read_asset(source["asset"]) == clip.read_bytes()
        assert service.repo.verify()["ok"]
        print(json.dumps({"detector_and_tracker_frames": 6, "effective_tracker": jobs[-1]["meta"]["actual_tracker"], "osnet_embedding_dimensions": int(embedding.size),
                          "encryption_and_audit": "verified", "accuracy": "not measured; generated non-person data and random OSNet weights"}, indent=2))


if __name__ == "__main__":
    main()
