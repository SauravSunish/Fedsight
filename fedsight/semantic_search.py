"""Case-scoped CLIP retrieval with encrypted persisted vectors.

The derived matrix is rebuilt in memory from encrypted, audited case records.
Similarity scores are rankings, not identity probabilities or verification.
"""
from __future__ import annotations

import hashlib
import io
import os
from .processing_queue import CaseTaskRegistry
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image


ARCHITECTURE = "MobileCLIP2-S0"
_MODELS = {}
_MODEL_LOCK = threading.Lock()


def model_identifier(weights_path):
    path = Path(weights_path)
    if not path.is_file():
        raise ValueError("Select a trusted local MobileCLIP2-S0 checkpoint before indexing.")
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return f"{ARCHITECTURE}:{h.hexdigest()}"


class MobileClip:
    def __init__(self, weights_path):
        self.model_id = model_identifier(weights_path)
        try:
            import torch
            import open_clip
        except Exception as exc:
            raise RuntimeError("MobileCLIP is unavailable; install its optional dependencies and check local application control.") from exc
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(ARCHITECTURE, pretrained=str(weights_path), device=self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(ARCHITECTURE)

    def image_vectors(self, images):
        batch = self.torch.stack([self.preprocess(image) for image in images]).to(self.device)
        with self.torch.inference_mode():
            vectors = self.model.encode_image(batch, normalize=True)
        return vectors.float().cpu().numpy()

    def text_vector(self, text):
        with self.torch.inference_mode():
            vector = self.model.encode_text(self.tokenizer([text]).to(self.device), normalize=True)
        return vector.float().cpu().numpy()[0]


def get_model(weights_path):
    model_id = model_identifier(weights_path)
    with _MODEL_LOCK:
        if model_id not in _MODELS:
            _MODELS[model_id] = MobileClip(weights_path)
        return _MODELS[model_id]


def _unit(vector):
    a = np.asarray(vector, dtype=np.float32)
    if a.ndim != 1 or not a.size or not np.isfinite(a).all() or np.linalg.norm(a) < 1e-10:
        raise ValueError("Invalid CLIP embedding.")
    return a / np.linalg.norm(a)


def index_case(service, case_id, actor, model, *, batch_size=16, progress=None):
    """Append only missing embeddings; vectors inherit case encryption and audit."""
    tracks = [t for t in service.active_tracks(case_id) if t.get("preview_asset")]
    existing = {(r["track_id"], r["model_id"], r["preview_hash"]) for r in service.repo.list(case_id, "clip_vector")}
    pending = [t for t in tracks if (t["id"], model.model_id, t["preview_asset"]["sha256"]) not in existing]
    indexed = 0
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        images = []
        for t in batch:
            with Image.open(io.BytesIO(service.read_asset(t["preview_asset"], actor))) as im:
                images.append(im.convert("RGB"))
        vectors = np.asarray(model.image_vectors(images))
        if vectors.ndim != 2 or len(vectors) != len(batch):
            raise ValueError("Image encoder returned an invalid batch.")
        for track, vector in zip(batch, vectors):
            service.repo.put("clip_vector", {"case_id": case_id, "track_id": track["id"],
                "model_id": model.model_id, "preview_hash": track["preview_asset"]["sha256"],
                "vector": _unit(vector).tolist()}, actor)
            indexed += 1
        if progress:
            progress(indexed, len(pending))
    return {"indexed": indexed, "available": len(tracks), "model_id": model.model_id}


def search_case(service, case_id, model, query, *, limit=10, camera_id=None,
                start_s=None, end_s=None, min_width=None, min_height=None):
    """Exact cosine ranking after metadata filtering of active, same-model tracks."""
    if not query.strip() or len(query) > 600 or not 1 <= limit <= 100:
        raise ValueError("Enter a description and request 1–100 results.")
    tracks = {t["id"]: t for t in service.active_tracks(case_id)}
    alignments = service.blind_alignments(case_id) if camera_id else {}
    vectors = {}
    for record in service.repo.list(case_id, "clip_vector"):
        track = tracks.get(record.get("track_id"))
        if not track or record.get("model_id") != model.model_id or record.get("preview_hash") != track.get("preview_asset", {}).get("sha256"):
            continue
        resolved_camera = alignments.get(track.get("source_id"), {}).get("camera_id", track.get("camera_id"))
        if camera_id and resolved_camera != camera_id:
            continue
        if start_s is not None and track.get("end_time", 0) < start_s:
            continue
        if end_s is not None and track.get("start_time", 0) > end_s:
            continue
        # Bounding-box dimensions are sourced from the saved track trajectory.
        trajectory = track.get("trajectory", [])
        if min_width is not None and not any(len(p) >= 5 and p[3] >= min_width for p in trajectory):
            continue
        if min_height is not None and not any(len(p) >= 5 and p[4] >= min_height for p in trajectory):
            continue
        vectors[track["id"]] = _unit(record["vector"])
    if not vectors:
        return {"results": [], "indexed": 0, "search_ms": 0.0}
    query_vector = _unit(model.text_vector(query))
    ids = list(vectors)
    matrix = np.stack([vectors[i] for i in ids])
    if matrix.shape[1] != query_vector.size:
        raise ValueError("Text and image embeddings have incompatible dimensions.")
    started = time.perf_counter()
    scores = matrix @ query_vector
    order = np.argsort(-scores, kind="stable")[:limit]
    elapsed = (time.perf_counter() - started) * 1000
    return {"results": [{"track": tracks[ids[i]], "score": float(scores[i])} for i in order],
            "indexed": len(ids), "search_ms": round(elapsed, 2)}


class IndexQueue(CaseTaskRegistry):

    def get(self, key):
        with self.lock:
            return self.tasks.get(key)

    def start(self, service, case_id, actor, weights):
        from .processing_queue import ProcessingTask
        key = (str(service.root.resolve()), case_id)
        with self.lock:
            previous = self.tasks.get(key)
            if previous and previous.snapshot()["status"] in {"queued", "running"}:
                raise ValueError("An index build is already running for this case.")
            task = ProcessingTask(case_id)
            self.tasks[key] = task
        def run():
            task.update(status="running", stage="Loading MobileCLIP")
            try:
                model = get_model(weights)
                def progress(done, total):
                    task.update(stage=f"Indexing previews ({done}/{total})", progress=done / max(total, 1))
                result = index_case(service, case_id, actor, model, progress=progress)
                task.update(status="completed", stage="Index ready", result=result, progress=1.)
            except Exception as exc:
                task.update(status="failed", stage="Index failed", error=f"{type(exc).__name__}: {str(exc)[:300]}")
        threading.Thread(target=run, daemon=True, name="fedsight-clip-index").start()
        return task


QUEUE = IndexQueue()
