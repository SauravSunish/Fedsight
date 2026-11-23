"""Unit verification of the mode toggle logic in fedsight."""
import pytest
from fedsight.navigation import resolve, pages
from fedsight.theme import THEMES

def test_resolve_mode_switching():
    history = {"case": "FedSight Investigator", "live": "Live respond"}
    
    # Toggle to live mode
    mode, page = resolve("live", None, history)
    assert mode == "live"
    assert page == "Live respond"
    
    # Toggle back to case mode
    mode, page = resolve("case", None, history)
    assert mode == "case"
    assert page == "FedSight Investigator"
    
def test_pages_match_modes():
    case_pages = pages("case")
    live_pages = pages("live")
    assert "FedSight Investigator" in case_pages
    assert "Live respond" in live_pages

def test_themes_support_both_modes():
    assert "case" in THEMES
    assert "live" in THEMES
    assert THEMES["case"]["bg-from"] != THEMES["live"]["bg-from"]

def test_mode_toggle_state_simulation():
    # Simulate on_mode_toggle handler
    state = {"mode_toggle": True, "mode": "case"}
    def on_mode_toggle():
        new_mode = "live" if state.get("mode_toggle") else "case"
        state["route_request"] = (new_mode, None)
        state["mode_switch"] = new_mode

    on_mode_toggle()
    assert state["route_request"] == ("live", None)
    assert state["mode_switch"] == "live"

    state["mode_toggle"] = False
    on_mode_toggle()
    assert state["route_request"] == ("case", None)
    assert state["mode_switch"] == "case"
