"""Tactical forensic intelligence dossiers: SVG gauges, telemetry sparklines, and chain-of-custody stamps."""
from __future__ import annotations

from html import escape
import math
from typing import Any
import numpy as np


# ---------------------------------------------------------------------------
# Biometric appearance color swatches
# ---------------------------------------------------------------------------
COLOR_PALETTE = {
    "black": ("#0f172a", "#334155"),
    "white": ("#f8fafc", "#94a3b8"),
    "gray": ("#64748b", "#cbd5e1"),
    "grey": ("#64748b", "#cbd5e1"),
    "charcoal": ("#1e293b", "#475569"),
    "slate": ("#334155", "#64748b"),
    "dark slate": ("#1a202c", "#4a5568"),
    "red": ("#dc2626", "#fca5a5"),
    "crimson": ("#991b1b", "#f87171"),
    "blue": ("#2563eb", "#93c5fd"),
    "navy": ("#1a365d", "#60a5fa"),
    "navy blue": ("#1a365d", "#60a5fa"),
    "green": ("#16a34a", "#86efac"),
    "forest green": ("#14532d", "#4ade80"),
    "yellow": ("#ca8a04", "#fde047"),
    "amber": ("#d97706", "#fcd34d"),
    "brown": ("#78350f", "#d97706"),
    "purple": ("#7e22ce", "#c084fc"),
    "cyan": ("#0891b2", "#67e8f9"),
    "unknown": ("#475569", "#94a3b8"),
}


def render_color_swatch(label: str, color_name: str | None, confidence_pct: int | None = None) -> str:
    """Render a tactical inline color swatch with confidence percentage."""
    c_name = (color_name or "unknown").lower()
    bg, border = COLOR_PALETTE.get(c_name, ("#475569", "#94a3b8"))
    confidence = f'<span style="margin-left:5px">{int(confidence_pct)}%</span>' if confidence_pct is not None else ""
    return (
        f'<span style="display:inline-flex;align-items:center;background:var(--card);border:1px solid var(--border);'
        f'padding:2px 8px;border-radius:4px;font-family:\'JetBrains Mono\',monospace;font-size:11px;margin-right:6px;">'
        f'<span style="display:inline-block;width:10px;height:10px;background:{bg};border:1px solid {border};'
        f'border-radius:2px;margin-right:6px;"></span>'
        f'<span style="color:var(--muted);margin-right:4px;">{escape(label)}:</span>'
        f'<span style="color:var(--text);font-weight:600;">{escape(c_name.upper())}</span>'
        f'{confidence}'
        f'</span>'
    )


def render_biometric_badges(track: dict) -> str:
    """Render upper/lower clothing swatches and accessory badges."""
    attrs = track.get("attributes") or {}
    if track.get("object_class", "person") == "person":
        chips = [render_color_swatch("UPPER", attrs.get("upper_color")),
                 render_color_swatch("LOWER", attrs.get("lower_color"))]
    else:
        chips = [render_color_swatch("COLOUR", attrs.get("color"))]
    if attrs.get("bag") in (True, "possible nearby bag"):
        chips.append(
            f'<span style="display:inline-flex;align-items:center;background:var(--card);border:1px solid var(--warn);'
            f'padding:2px 8px;border-radius:4px;font-family:\'JetBrains Mono\',monospace;font-size:11px;color:var(--warn);">'
            f'🎒 POSSIBLE NEARBY BAG'
            f'</span>'
        )
    return "".join(chips)


# ---------------------------------------------------------------------------
# Motion velocity sparkline (Inline SVG)
# ---------------------------------------------------------------------------
def render_velocity_sparkline(trajectory: np.ndarray | list | None, fps: float | None = None,
                              *, width: int = 110, height: int = 24) -> str:
    """Generate an inline SVG sparkline showing velocity over time from trajectory samples."""
    unavailable = '<span>Image motion · unavailable</span>'
    if trajectory is None or len(trajectory) < 2:
        return unavailable
    arr = np.asarray(trajectory, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return unavailable
    steps = np.diff(arr[:, :3], axis=0)
    steps = steps[np.isfinite(steps).all(axis=1) & (steps[:, 0] > 0)]
    if not len(steps):
        return unavailable
    units = "px/s" if fps and fps > 0 else "px/frame"
    speeds = np.hypot(steps[:, 1], steps[:, 2]) / steps[:, 0]
    if units == "px/s":
        speeds *= fps
    max_s = max(float(np.max(speeds)), 1.0)
    avg_s = float(np.average(speeds, weights=steps[:, 0]))
    n = len(speeds)

    pts = []
    for i, s in enumerate(speeds):
        x = (i / max(n - 1, 1)) * (width - 4) + 2
        y = height - 2 - (s / max_s) * (height - 6)
        pts.append(f"{x:.1f},{y:.1f}")

    path_data = " ".join(pts)
    first_pt = pts[0]
    last_pt = pts[-1]
    last_x = (width - 4) + 2
    fill_path = f"M {first_pt} L {path_data} L {last_x},{height} L 2,{height} Z"

    svg = (
        f'<span style="display:inline-flex;align-items:center;gap:6px;font-family:\'JetBrains Mono\',monospace;font-size:11px;">'
        f'<span style="color:var(--muted);">Image motion</span>'
        f'<span style="color:var(--accent);font-weight:600;">{avg_s:.1f} {units}</span>'
        f'<svg width="{width}" height="{height}" style="background:var(--bg-from);border:1px solid var(--border);border-radius:3px;vertical-align:middle;">'
        f'<path d="{fill_path}" fill="var(--surface-active)"/>'
        f'<polyline points="{path_data}" fill="none" stroke="var(--accent)" stroke-width="1.5" stroke-linecap="round"/>'
        f'</svg>'
        f'</span>'
    )
    return svg


# ---------------------------------------------------------------------------
# Circular confidence radar gauge / dial (Inline SVG)
# ---------------------------------------------------------------------------
def render_confidence_dial(score: float, label: str = "CONFIDENCE", *, size: int = 76) -> str:
    """Render a high-tech tactical circular radar dial in SVG."""
    clamped = max(0.0, min(1.0, float(score)))
    pct = int(clamped * 100)

    if clamped >= 0.75:
        stroke_color = "var(--accent)"
        glow_color = "var(--surface-active)"
        status_text = "HIGH"
    elif clamped >= 0.50:
        stroke_color = "var(--warn)"
        glow_color = "var(--warn)"
        status_text = "MODERATE"
    else:
        stroke_color = "var(--critical)"
        glow_color = "var(--critical-glow)"
        status_text = "WEAK"

    r = size * 0.38
    cx, cy = size / 2, size / 2
    circ = 2 * math.pi * r
    dash_offset = circ * (1.0 - clamped)

    svg = (
        f'<div style="display:inline-flex;flex-direction:column;align-items:center;text-align:center;margin:4px 8px;">'
        f'<svg width="{size}" height="{size}" style="transform:rotate(-90deg);overflow:visible;">'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="var(--border)" stroke-width="5"/>'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{stroke_color}" stroke-width="5"'
        f' stroke-dasharray="{circ:.1f}" stroke-dashoffset="{dash_offset:.1f}" stroke-linecap="round"'
        f' style="filter:drop-shadow(0 0 4px {glow_color});transition:stroke-dashoffset 0.5s ease;"/>'
        f'</svg>'
        f'<div style="margin-top:-{int(size*0.65)}px;font-family:\'JetBrains Mono\',monospace;font-size:14px;font-weight:700;color:var(--text);">{pct}%</div>'
        f'<div style="font-family:\'Chakra Petch\',sans-serif;font-size:9px;color:{stroke_color};letter-spacing:0.1em;margin-top:10px;">{status_text}</div>'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:8.5px;color:var(--muted);letter-spacing:0.08em;margin-top:2px;">{label}</div>'
        f'</div>'
    )
    return svg


# ---------------------------------------------------------------------------
# Digital Chain-of-Custody rubber stamp (Inline SVG / CSS)
# ---------------------------------------------------------------------------
def render_custody_stamp(actor: str = "OFFICER", action: str = "VERIFIED EVIDENCE",
                         timestamp: str = "", hash_prefix: str = "",
                         stamp_color: str = "teal") -> str:
    """Render an authentic military/forensic digital rubber stamp graphic."""
    ts = timestamp[:19] if timestamp else "AUDITED"
    if stamp_color == "red":
        border_c = "var(--critical)"
        bg_c = "var(--card)"
        shadow_c = "var(--critical-glow)"
        text_c = "var(--critical)"
    else:
        border_c = "var(--accent)"
        bg_c = "var(--card)"
        shadow_c = "var(--surface-active)"
        text_c = "var(--accent)"

    return (
        f'<div style="display:inline-block;border:2px dashed {border_c};background:{bg_c};'
        f'padding:5px 12px;border-radius:4px;transform:rotate(-3deg);margin:4px 0;'
        f'box-shadow:0 0 10px {shadow_c};font-family:\'Chakra Petch\',sans-serif;">'
        f'<div style="font-size:11px;font-weight:800;letter-spacing:0.18em;color:{text_c};text-align:center;">'
        f'◈ {escape(action)} ◈'
        f'</div>'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:8.5px;color:var(--muted);letter-spacing:0.08em;text-align:center;margin-top:2px;">'
        f'ACTOR: {escape(actor.upper())} · {escape(ts)} {"· SHA:" + escape(hash_prefix[:6]) if hash_prefix else ""}'
        f'</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Complete Tactical Suspect Dossier Header
# ---------------------------------------------------------------------------
def render_tactical_dossier_header(track: dict, source: dict) -> str:
    """Build the full tactical suspect dossier HTML header with biometric badges and sparkline."""
    tid = track["id"][-8:].upper()
    obj_class = track.get("object_class", "person").upper()
    confidence = track.get("avg_conf")
    detection = f" · {confidence:.0%} detection" if confidence is not None else ""

    biometrics = render_biometric_badges(track)
    sparkline = render_velocity_sparkline(track.get("trajectory"))

    html = (
        f'<div class="targeting-card" style="margin-bottom:10px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--border);padding-bottom:6px;margin-bottom:8px;">'
        f'  <div style="font-family:\'Chakra Petch\',sans-serif;font-size:13px;font-weight:700;color:var(--accent);letter-spacing:0.12em;">'
        f'    Sighting · {escape(tid)}'
        f'  </div>'
        f'  <div style="display:flex;gap:6px;align-items:center;">'
        f'    <span style="font-family:\'JetBrains Mono\',monospace;font-size:10px;background:var(--card);border:1px solid var(--border);color:var(--muted);padding:2px 6px;border-radius:3px;">'
        f'      {escape(obj_class)}{detection}'
        f'    </span>'
        f'  </div>'
        f'</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:6px;">'
        f'  {biometrics}'
        f'</div>'
        f'<div style="margin-top:6px;">'
        f'  {sparkline}'
        f'</div>'
        f'</div>'
    )
    return html
