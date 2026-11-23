"""Render annotated clips off the Streamlit UI thread."""
from __future__ import annotations

import os
import tempfile
from .processing_queue import CaseTaskRegistry
import threading
from pathlib import Path
from .processing_queue import ProcessingTask
from .review_video import render_annotated_clip


class RenderQueue(CaseTaskRegistry):

    def get(self, root, case_id, source_id):
        with self.lock:
            return self.tasks.get((str(root), case_id, source_id))

    def start(self, service, case_id, source, actor, tracks, *, start_s, duration_s,
              options, time_anchor, validated_ids, rules):
        key = (str(service.root.resolve()), case_id, source["id"])
        with self.lock:
            previous = self.tasks.get(key)
            if previous and previous.snapshot()["status"] in {"queued", "running"}:
                raise ValueError("An annotated clip is already rendering for this recording.")
            task = ProcessingTask(case_id)
            self.tasks[key] = task
        def work():
            input_path = output_path = None
            task.update(status="running", stage="Rendering annotated clip")
            try:
                blob = service.read_asset(source["asset"], actor)
                with tempfile.NamedTemporaryFile(dir=service.work, suffix=source.get("suffix", ".mp4"), delete=False) as tmp:
                    tmp.write(blob)
                    input_path = Path(tmp.name)
                os.chmod(input_path, 0o600)
                with tempfile.NamedTemporaryFile(dir=service.work, suffix=".mp4", delete=False) as tmp:
                    output_path = Path(tmp.name)
                os.chmod(output_path, 0o600)
                info = render_annotated_clip(str(input_path), str(output_path), tracks, start_s=start_s,
                                             duration_s=duration_s, options=options, time_anchor=time_anchor,
                                             validated_ids=validated_ids, rules=rules)
                rendered = output_path.read_bytes()
                asset = service.save_asset(rendered, case_id, actor, role="annotated_clip",
                                           parent_id=source["asset"]["asset_id"], source_id=source["id"],
                                           transformation="sampled bounding boxes, HUD and source timestamp",
                                           start_time=start_s, render_meta=info, overlay_options=options,
                                           validated_track_ids=sorted(validated_ids), rule_ids=[r["id"] for r in rules],
                                           track_ids=[t["id"] for t in tracks])
                task.update(status="completed", stage="Rendered", progress=1., result={"asset": asset, "render_meta": info})
            except Exception as exc:
                task.update(status="failed", stage="Render failed", error=f"{type(exc).__name__}: {str(exc)[:300]}")
            finally:
                if input_path is not None:
                    input_path.unlink(missing_ok=True)
                if output_path is not None:
                    output_path.unlink(missing_ok=True)
        threading.Thread(target=work, daemon=True, name="fedsight-render-" + source["id"][-8:]).start()
        return task


QUEUE = RenderQueue()
