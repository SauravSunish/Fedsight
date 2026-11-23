"""Bounded, local-only visual search over existing tracklet preview crops."""
from __future__ import annotations

import base64
import io
import json
from .processing_queue import CaseTaskRegistry
import threading
import time
from collections import defaultdict, deque

from PIL import Image, ImageDraw


def search_previews(question, tools, actor, model="gemma4:12b", limit=36, *, post=None, candidate_ids=None):
    """Return model-labelled leads, never asserting identity or exhaustive search."""
    import requests
    from .investigator import normalize_description
    started = time.monotonic()
    tracks = [t for t in tools.tracks.values()
              if t.get("object_class", "person") == "person" and t.get("preview_asset")]
    tracks.sort(key=lambda t: (tools.sources[t["source_id"]]["name"], t["start_time"], t["id"]))
    available_count = len(tracks)
    if candidate_ids is not None:
        order = {track_id: i for i, track_id in enumerate(candidate_ids)}
        tracks = sorted((t for t in tracks if t["id"] in order), key=lambda t: order[t["id"]])
    by_source = defaultdict(deque)
    for track in tracks:
        by_source[track["source_id"]].append(track)
    max_items = len(tracks) if (limit is None or limit <= 0) else limit
    selected = []
    if candidate_ids is not None:
        selected = tracks[:max_items]
    else:
        while len(selected) < max_items and any(by_source.values()):
            for items in by_source.values():
                if items and len(selected) < max_items:
                    selected.append(items.popleft())
    if not selected:
        raise ValueError("No person preview images are available in this case.")

    batch_size = 12
    batches = [selected[i:i + batch_size] for i in range(0, len(selected), batch_size)]

    schema = {"type": "object", "properties": {"results": {"type": "array", "items": {
        "type": "object", "properties": {"image": {"type": "integer"},
        "verdict": {"type": "string", "enum": ["match", "uncertain", "no_match"]},
        "reason": {"type": "string"}}, "required": ["image", "verdict", "reason"]}}}, "required": ["results"]}

    seen, evidence, rejected = set(), [], []
    batch_offset = 0

    for batch in batches:
        sheet = Image.new("RGB", (768, ((len(batch) + 2) // 3) * 280), "#202020")
        draw = ImageDraw.Draw(sheet)
        for i, track in enumerate(batch):
            raw = tools.service.read_asset(track["preview_asset"], actor)
            with Image.open(io.BytesIO(raw)) as original:
                crop = original.convert("RGB")
                crop.thumbnail((246, 244))
                x, y = (i % 3) * 256, (i // 3) * 280
                sheet.paste(crop, (x + (256 - crop.width) // 2, y + 30))
                draw.text((x + 8, y + 5), f"Image {i+1}", fill="white")
        out = io.BytesIO()
        sheet.save(out, format="JPEG", quality=85)
        clean_query = normalize_description(question)[:600]
        prompt = (f"Examine all {len(batch)} numbered crops, image numbers 1 through {len(batch)}. "
                  "Return one result for every image. Requested visible appearance: " + clean_query +
                  ". Do not infer gender or identity. Evaluate visible appearance (hair length, baldness, clothing, glasses, headwear, bags) regardless of gender terms in the search. "
                  "Only say match when visible features clearly match the description; "
                  "use uncertain if partially visible or ambiguous; otherwise use no_match. Never guess hidden features. Return JSON only: "
                  '{"results":[{"image":1,"verdict":"match|uncertain|no_match","reason":"brief visible evidence"}]}')
        try:
            response = (post or requests.post)("http://127.0.0.1:11434/api/chat", json={
                "model": model, "stream": False, "think": False, "format": schema, "keep_alive": "15m",
                "options": {"temperature": 0, "num_predict": 1400, "num_ctx": 4096},
                "messages": [{"role": "user", "content": prompt,
                              "images": [base64.b64encode(out.getvalue()).decode("ascii")]}]}, timeout=240)
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError("Ollama is not running locally at http://127.0.0.1:11434. Start Ollama ('ollama serve') and verify your model is installed.") from exc
        except requests.exceptions.Timeout as exc:
            raise RuntimeError("The local vision model timed out. Try again or reduce the preview limit.") from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Could not connect to Ollama: {exc}") from exc

        response.raise_for_status()
        try:
            parsed = json.loads(response.json()["message"]["content"])
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError("The vision model returned an incomplete or invalid response. Try again or narrow the search.") from exc

        batch_seen = set()
        for item in parsed.get("results", []):
            n = item.get("image")
            if type(n) is not int or not 1 <= n <= len(batch) or n in batch_seen or item.get("verdict") not in {"match", "uncertain", "no_match"}:
                continue
            batch_seen.add(n)
            global_idx = batch_offset + (n - 1)
            seen.add(global_idx)
            record = tools.track_evidence(batch[n - 1])
            record["visual_reason"] = str(item.get("reason", ""))[:200]
            record["visual_verdict"] = item["verdict"]
            if item.get("verdict") in {"match", "uncertain"}:
                evidence.append(record)
            else:
                rejected.append(record)

        if not batch_seen:
            raise ValueError("The local vision model did not assess any selected image.")
        batch_offset += len(batch)

    if not seen:
        raise ValueError("The local vision model did not assess any selected image.")
    evidence.sort(key=lambda item: item["visual_verdict"] != "match")
    total = available_count
    matches = sum(e["visual_verdict"] == "match" for e in evidence)
    uncertain = len(evidence) - matches
    answer = (f"Local vision reviewed {len(seen)} of {total} available person previews and found "
              f"{matches} possible match(es) and {uncertain} uncertain image(s) to review. "
              "Uncertain images are not confirmed matches. These are visual leads, not confirmed identities; "
              "review the original footage. Gender is not assessed. Unreviewed previews may contain additional matches.")
    return {"answer": answer, "evidence": evidence, "reviewed_no_match": rejected,
            "working_set": [e["track_id"] for e in evidence],
            "tool_calls": [{"name": "local_vision_search", "arguments": {"model": model, "reviewed": len(seen), "available": total}}],
            "engine": "Local vision · " + model, "latency_ms": round((time.monotonic() - started) * 1000)}


class VisionQueue(CaseTaskRegistry):
    """One visual query per conversation; work and UI lifetimes are separate."""

    def get(self, key):
        with self.lock:
            return self.tasks.get(key)

    def start(self, key, question, tools, actor, model="gemma4:12b", limit=36, candidate_ids=None):
        from .processing_queue import ProcessingTask
        with self.lock:
            previous = self.tasks.get(key)
            if previous and previous.snapshot()["status"] in {"queued", "running"}:
                raise ValueError("A visual search is already running in this conversation.")
            task = ProcessingTask(tools.case_id)
            self.tasks[key] = task
        def run():
            task.update(status="running", stage="Reviewing previews")
            try:
                kwargs = {"candidate_ids": candidate_ids} if candidate_ids is not None else {}
                result = search_previews(question, tools, actor, model, limit, **kwargs)
                task.update(status="completed", result=result, progress=1.)
            except Exception as exc:
                task.update(status="failed", error=str(exc)[:300])
        threading.Thread(target=run, daemon=True, name="fedsight-vision").start()
        return task


QUEUE = VisionQueue()
