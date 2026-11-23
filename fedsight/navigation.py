"""Workbench routes, public page links and per-mode history."""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Page:
    title: str
    handler: str
    modes: tuple[str, ...]
    icon: str
    shortcut: str = ""
    live_handler: str = ""

    @property
    def slug(self):
        return re.sub(r"[^a-z0-9]+", "-", self.title.lower()).strip("-")


CASE = ("case",)
LIVE = ("live",)
BOTH = ("case", "live")
REGISTRY = {p.title: p for p in (
    Page("Overview", "overview", CASE, "◈"),
    Page("Evidence & processing", "evidence", CASE, "▤"),
    Page("Video review", "video_review", CASE, "▷"),
    Page("ThreadSight review", "review", CASE, "⇄", "Review proposed links"),
    Page("Search & digest", "search_digest", CASE, "⌕"),
    Page("FedSight Investigator", "investigation_chat", CASE, "⌕", "Search recordings"),
    Page("Cases", "case_board", CASE, "▤"),
    Page("Timeline & graph", "timeline", CASE, "⇥"),
    Page("Analytics", "analytics", CASE, "▥"),
    Page("Evaluation lab", "evaluation", CASE, "◇"),
    Page("Integrity & audit", "integrity", CASE, "✓"),
    Page("Live wall", "wall", LIVE, "▦", "View camera wall"),
    Page("Live respond", "respond", LIVE, "⚑", "Review live alerts"),
    Page("Rules & events", "rules", BOTH, "⚑"),
    Page("Camera intelligence", "cameras", BOTH, "◎", live_handler="camera_page"),
    Page("Spatial intelligence", "spatial_intelligence", BOTH, "⌖", "Explore camera coverage", live_handler="spatial_page"),
    Page("BlindSight", "blindsight", BOTH, "◇", live_handler="blind_page"),
    Page("Live analytics", "live_analytics", LIVE, "▥"),
    Page("Live Investigator", "live_investigator", LIVE, "⌕"),
    Page("System health", "health", LIVE, "◉"),
)}
DEFAULT = {"case": "Overview", "live": "Live wall"}


def pages(mode):
    return [p.title for p in REGISTRY.values() if mode in p.modes]


def workspace_menu(mode, current):
    """Keep site tools in Case shortcuts and show the current tool when opened."""
    return [name for name in pages(mode)
            if mode == "live" or REGISTRY[name].modes == CASE or name == current]


def resolve(mode, page, history):
    mode = mode if mode in DEFAULT else "case"
    target = next((p for p in REGISTRY.values() if page in (p.slug, p.title)), None)
    if target:
        return (mode if mode in target.modes else target.modes[0], target.title)
    remembered = history.get(mode)
    return mode, remembered if remembered in pages(mode) else DEFAULT[mode]
