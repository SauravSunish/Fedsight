"""In-process processing workers; Streamlit calls remain on the UI thread."""
from __future__ import annotations

from contextlib import contextmanager
import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProcessingTask:
    case_id: str
    cancel_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    status: str = "queued"
    stage: str = "Queued"
    source: str = ""
    progress: float = 0.0
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: float = field(default_factory=time.monotonic)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            elapsed = max(0., time.monotonic() - self.started_at)
            eta = elapsed * (1 - self.progress) / self.progress if .01 <= self.progress < 1 else None
            return {"status": self.status, "stage": self.stage, "source": self.source,
                    "progress": self.progress, "result": self.result, "error": self.error,
                    "elapsed_s": elapsed, "eta_s": eta}

    def update(self, **values: Any) -> None:
        with self.lock:
            for name, value in values.items():
                setattr(self, name, value)


class CaseTaskRegistry:
    def __init__(self):
        self.lock = threading.Lock()
        self.tasks = {}

    @contextmanager
    def removing_case(self, scope):
        with self.lock:
            keys = [key for key in self.tasks if key[:2] == scope]
            if any(self.tasks[key].snapshot()["status"] in {"queued", "running", "cancelling"} for key in keys):
                raise ValueError("A background task is using this case. Finish or stop it before deleting.")
            yield
            for key in keys:
                del self.tasks[key]


class ProcessingQueue(CaseTaskRegistry):
    """One background task per case in this application process."""

    def get(self, root: str, case_id: str) -> ProcessingTask | None:
        with self.lock:
            return self.tasks.get((root, case_id))

    def start(self, service, case_id: str, actor: str, video_config, embedding_config, *, force: bool = False) -> ProcessingTask:
        key = (str(service.root.resolve()), case_id)
        with self.lock:
            prior = self.tasks.get(key)
            if prior and prior.snapshot()["status"] in {"queued", "running", "cancelling"}:
                raise ValueError("A processing task is already running for this case.")
            task = ProcessingTask(case_id)
            self.tasks[key] = task

        def run():
            task.update(status="running")
            def progress(index, count, done, total, name):
                task.update(source=name, progress=min(1., (index + done / max(total, 1)) / max(count, 1)))
            def stage(name, source):
                task.update(stage=name, source=source)
            try:
                result = service.process(case_id, actor, video_config, embedding_config, progress=progress,
                                         force=force, should_stop=task.cancel_event.is_set, stage=stage)
                status = "cancelled" if result.get("cancelled") else "failed" if result.get("failed_sources") else "completed"
                task.update(status=status, stage=status.title(), result=result,
                            progress=1.0 if status == "completed" else task.snapshot()["progress"])
            except Exception as exc:
                # A worker failure must not leave the case indefinitely locked in Processing.
                try:
                    if service.repo.get(case_id, case_id, "case")["status"] == "Processing":
                        service.repo.transition(case_id, "Failed", actor)
                except Exception:
                    pass
                task.update(status="failed", stage="Failed", error=f"{type(exc).__name__}: {str(exc)[:300]}")

        threading.Thread(target=run, daemon=True, name="fedsight-processing-" + case_id[-8:]).start()
        return task

    def cancel(self, root: str, case_id: str) -> bool:
        task = self.get(root, case_id)
        if not task or task.snapshot()["status"] not in {"queued", "running", "cancelling"}:
            return False
        task.cancel_event.set()
        task.update(status="cancelling", stage="Stopping after current operation")
        return True


QUEUE = ProcessingQueue()
