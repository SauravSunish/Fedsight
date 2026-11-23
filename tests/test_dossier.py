"""Unit tests for tactical dossier cards, biometric badges, gauges, and custody stamps."""
import pytest
import numpy as np

from fedsight.dossier import (
    render_color_swatch,
    render_biometric_badges,
    render_velocity_sparkline,
    render_confidence_dial,
    render_custody_stamp,
    render_tactical_dossier_header,
)


def test_render_color_swatch():
    html = render_color_swatch("UPPER", "navy blue", 92)
    assert "NAVY BLUE" in html
    assert "92%" in html
    assert "#1a365d" in html


def test_render_biometric_badges():
    track = {
        "id": "trk_12345",
        "avg_conf": 0.88,
        "attributes": {
            "upper_color": "Navy Blue",
            "lower_color": "Dark Slate",
            "bag": True,
        },
    }
    html = render_biometric_badges(track)
    assert "UPPER" in html
    assert "NAVY BLUE" in html
    assert "LOWER" in html
    assert "DARK SLATE" in html
    assert "POSSIBLE NEARBY BAG" in html


def test_render_velocity_sparkline():
    traj = np.array([
        [0, 10, 10, 20, 40],
        [10, 30, 25, 20, 40],
        [20, 60, 40, 20, 40],
    ])
    html = render_velocity_sparkline(traj, fps=25.0)
    assert "<svg" in html
    assert "polyline" in html
    assert "px/s" in html

    # Empty trajectory
    empty_html = render_velocity_sparkline(None)
    assert "unavailable" in empty_html


def test_unknown_appearance_has_no_bag_or_invented_measurements():
    html = render_tactical_dossier_header({"id": "track_42", "attributes": {"bag": "unknown"}}, {})
    assert "UNKNOWN" in html
    assert "BAG" not in html
    assert "90%" not in html
    assert "unavailable" in html


def test_render_confidence_dial():
    html_high = render_confidence_dial(0.892, label="MATCH CONFIDENCE")
    assert "89%" in html_high
    assert "HIGH" in html_high
    assert "MATCH CONFIDENCE" in html_high

    html_med = render_confidence_dial(0.62)
    assert "62%" in html_med
    assert "MODERATE" in html_med

    html_low = render_confidence_dial(0.35)
    assert "35%" in html_low
    assert "WEAK" in html_low


def test_render_custody_stamp():
    html_teal = render_custody_stamp("AGENT-07", "VERIFIED EVIDENCE", "2026-11-16T13:40:00", "a8b9c0", stamp_color="teal")
    assert "VERIFIED EVIDENCE" in html_teal
    assert "AGENT-07" in html_teal
    assert "var(--accent)" in html_teal

    html_red = render_custody_stamp("OFFICER", "CHAIN OF CUSTODY", stamp_color="red")
    assert "CHAIN OF CUSTODY" in html_red
    assert "var(--critical)" in html_red


def test_render_tactical_dossier_header():
    track = {
        "id": "track_alpha_8d9d",
        "object_class": "person",
        "avg_conf": 0.94,
        "attributes": {"color": "blue"},
    }
    source = {"id": "src_1", "name": "North Gate"}
    html = render_tactical_dossier_header(track, source)
    assert "TARGET DOSSIER // ALPHA_8D9D" in html or "8D9D" in html
    assert "PERSON" in html
    assert "94% detection" in html
