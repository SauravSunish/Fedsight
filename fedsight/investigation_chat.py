"""Case-scoped, source-linked investigation search and optional vision reranking."""
from __future__ import annotations

import base64
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .core import TimeAnchor


COLOURS = {"black", "white", "gray", "grey", "red", "blue", "green", "yellow", "brown"}
UPPER = {"shirt", "jacket", "top", "hoodie", "coat", "tshirt", "sweater"}
LOWER = {"trousers", "pants", "jeans", "shorts", "skirt"}
VEHICLES = {"bicycle": 1, "bike": 1, "car": 2, "motorcycle": 3, "motorbike": 3, "bus": 5, "truck": 7}
PERSON_WORDS = {"person", "people", "man", "woman", "pedestrian", "individual", "someone"}
VISUAL_TERMS = {"backpack", "bag", "hat", "cap", "umbrella", "phone", "luggage", "suitcase", "glasses", "logo", "stripe"}
MODELS = ("gpt-6.1-terra", "gpt-6.1-sol")


def _seconds(token: str) -> float:
    parts = [int(x) for x in token.split(":")]
    if len(parts) == 2:
        return float(parts[0] * 60 + parts[1])
    if len(parts) == 3:
        return float(parts[0] * 3600 + parts[1] * 60 + parts[2])
    raise ValueError("Use mm:ss or hh:mm:ss for clip times.")


def query_filters(query: str) -> dict:
    sequence = re.findall(r"[a-z]+", query.lower())
    words = set(sequence)
    category = "person" if words & (PERSON_WORDS | UPPER | LOWER) else "vehicle" if words & (set(VEHICLES) | {"vehicle"}) else None
    subtype = next((VEHICLES[w] for w in VEHICLES if w in words), None)
    colour = next(("gray" if w == "grey" else w for w in sequence if w in COLOURS), None)
    def garment_colour(garments):
        for i, token in enumerate(sequence):
            if token in garments:
                nearby = sequence[max(0, i-2):i+3]
                found = next(("gray" if w == "grey" else w for w in nearby if w in COLOURS), None)
                if found:
                    return found
        return None
    upper, lower = garment_colour(UPPER), garment_colour(LOWER)
    if upper or lower:
        colour = None
    visual_terms = sorted(words & VISUAL_TERMS)
    if visual_terms and not (upper or lower):
        colour = None  # Colour may describe an unindexed object, not the person crop.
    motion = "stationary" if words & {"stationary", "standing", "waiting"} else "moving" if words & {"walking", "running", "moving"} else None
    times = re.findall(r"\b(?:\d{1,2}:)?\d{1,2}:\d{2}\b", query)
    interval = (_seconds(times[0]), _seconds(times[1])) if len(times) >= 2 else ((_seconds(times[0])-15, _seconds(times[0])+15) if times else None)
    if interval is not None:
        interval = (max(0., interval[0]), interval[1])
    return {"category": category, "subtype": subtype, "colour": colour, "upper": upper, "lower": lower,
            "motion": motion, "interval": interval, "needs_vision": bool(visual_terms),
            "visual_terms": visual_terms}


def local_candidates(query: str, tracks: list[dict], sources: dict, *, limit=512) -> tuple[list[dict], dict]:
    if not query.strip() or len(query) > 600:
        raise ValueError("Describe the person, vehicle or event in 1–600 characters.")
    filters = query_filters(query)
    requested_sources = {sid for sid, source in sources.items()
                         if len(Path(source["name"]).stem) >= 4 and Path(source["name"]).stem.lower() in query.lower()}
    result = []
    for track in tracks:
        source = sources[track["source_id"]]
        if requested_sources and track["source_id"] not in requested_sources:
            continue
        category = track.get("object_class", "person")
        if filters["category"] and category != filters["category"]:
            continue
        if filters["subtype"] and track.get("class_id") != filters["subtype"]:
            continue
        if filters["interval"]:
            a, b = filters["interval"]
            if track["end_time"] < a or track["start_time"] > b:
                continue
        attrs = track.get("attributes", {})
        score, reasons = 0., []
        if filters["colour"]:
            fields = [attrs.get("color"), attrs.get("upper_color"), attrs.get("lower_color")]
            if filters["colour"] not in fields:
                continue
            score += .25; reasons.append("Approximate crop colour")
        if filters["upper"]:
            if attrs.get("upper_color") != filters["upper"]:
                continue
            score += .25; reasons.append("Approximate upper-region colour")
        if filters["lower"]:
            if attrs.get("lower_color") != filters["lower"]:
                continue
            score += .25; reasons.append("Approximate lower-region colour")
        if filters["motion"]:
            stationary = bool(track.get("motion", {}).get("stationary", False))
            if stationary != (filters["motion"] == "stationary"):
                continue
            score += .15; reasons.append("Track motion")
        q = query.lower()
        if source["name"].lower() in q or any(w.lower() in q for w in source["name"].split() if len(w) >= 4):
            score += .2; reasons.append("Source name")
        if filters["category"]:
            score += .1; reasons.append("Detected category")
        if filters["interval"]:
            score += .1; reasons.append("Clip time overlap")
        quality = float(track.get("embedding_meta", {}).get("mean_quality", track.get("quality_report", {}).get("mean_quality", 0)))
        score += .1 * quality
        result.append({"track": track, "score": round(score, 4), "reasons": reasons,
                       "verification": "metadata_candidate"})
    result.sort(key=lambda item: (-item["score"], -item["track"].get("duration_s", 0), item["track"]["id"]))
    filters["source_ids"] = sorted(requested_sources)
    return result[:limit], filters


def evidence_time(track: dict, source: dict, job: dict | None = None) -> dict:
    frames = track.get("crop_frames") or []
    scores = track.get("crop_scores") or []
    frame = track.get("preview_frame")
    if frame is None and frames:
        frame = frames[max(range(len(frames)), key=lambda i: scores[i] if i < len(scores) else 0)]
    fps = (job or {}).get("meta", {}).get("fps")
    clip_s = frame / fps if frame is not None and fps else track["start_time"]
    bounds = TimeAnchor(**source.get("time_anchor", {})).interval(clip_s)
    utc = None if bounds is None else datetime.fromtimestamp(sum(bounds)/2, timezone.utc).isoformat()
    return {"preview_frame": frame, "preview_clip_s": round(float(clip_s), 2),
            "track_start_s": track["start_time"], "track_end_s": track["end_time"],
            "utc_midpoint": utc, "utc_uncertainty_bounds": bounds}


SCHEMA = {"type": "object", "properties": {"matches": {"type": "array", "items": {
    "type": "object", "properties": {"image_id": {"type": "integer"},
    "verdict": {"type": "string", "enum": ["match", "uncertain", "no_match"]},
    "visible_reason": {"type": "string"}},
    "required": ["image_id", "verdict", "visible_reason"], "additionalProperties": False}}},
    "required": ["matches"], "additionalProperties": False}


def vision_rerank(query: str, candidates: list[dict], service, actor: str, api_key: str, *,
                  model="gpt-6.1-terra", max_images=16, client=None, stream=False) -> list[dict]:
    if model not in MODELS or not 1 <= max_images <= 48:
        raise ValueError("Choose a supported model and a preview limit between 1 and 48.")
    if client is None:
        if not api_key:
            raise ValueError("An OpenAI API key is required for visual matching.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install requirements-ai.txt to enable OpenAI vision search.") from exc
        client = OpenAI(api_key=api_key, timeout=60., max_retries=1)
    selected = [item for item in candidates if item["track"].get("preview_asset")][:max_images]
    if not selected:
        return []
    matches = []
    for start in range(0, len(selected), 8):
        batch = selected[start:start+8]
        content = [{"type": "input_text", "text": (
            "Assess the investigator's description against these indexed preview crops. "
            "Return one verdict per image ID. Match only directly visible features; choose uncertain if obscured. "
            "Do not identify a real person, infer intent, or invent objects, timestamps, cameras or events. "
            f"Description: {query[:600]}")}]
        for index, item in enumerate(batch, 1):
            raw = service.read_asset(item["track"]["preview_asset"], actor)
            with Image.open(io.BytesIO(raw)) as im:
                im = im.convert("RGB")
                im.thumbnail((384, 384))
                output = io.BytesIO()
                im.save(output, format="JPEG", quality=80)
            content.append({"type": "input_text", "text": f"Image ID {index}:"})
            content.append({"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode(), "detail": "low"})
        response = client.responses.create(model=model, input=[{"role": "user", "content": content}],
                                           text={"format": {"type": "json_schema", "name": "visual_search",
                                                            "schema": SCHEMA, "strict": True}}, store=False, stream=stream)
        if stream:
            completed = None
            for event in response:
                if event.type == "response.failed":
                    raise RuntimeError("ChatGPT plan request failed. Check the plan limit or app access in ChatGPT settings.")
                if event.type == "response.completed":
                    completed = event.response
            if completed is None:
                raise RuntimeError("ChatGPT plan request ended without a completed response.")
            response = completed
        parsed = json.loads(response.output_text)
        seen = set()
        for assessment in parsed.get("matches", []):
            index = assessment.get("image_id")
            if not isinstance(index, int) or not 1 <= index <= len(batch) or index in seen:
                continue
            seen.add(index)
            if assessment.get("verdict") not in {"match", "uncertain"}:
                continue
            item = dict(batch[index-1])
            item["verification"] = assessment["verdict"]
            item["visual_reason"] = str(assessment.get("visible_reason", ""))[:300]
            item["model"] = model
            matches.append(item)
    matches.sort(key=lambda item: (item["verification"] != "match", -item["score"], item["track"]["id"]))
    return matches
