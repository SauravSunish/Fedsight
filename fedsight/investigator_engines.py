"""Optional provider adapters for selecting an allowlisted case tool.

Provider text is never used as the factual answer: only validated tool results
are used for the deterministic response. Credentials stay in process memory.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from .investigator import answer_from_intent, answer_offline, parse_intent


PROVIDERS = {
    "Local (Ollama)": {"url": "http://127.0.0.1:11434/v1", "model": "llama3.1:8b", "env": None},
    "Cloud fast (Groq)": {"url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile", "env": "GROQ_API_KEY"},
    "Cloud planner (Gemini)": {"url": "https://generativelanguage.googleapis.com/v1beta/openai/", "model": "gemini-2.0-flash", "env": "GEMINI_API_KEY"},
    "OpenRouter": {"url": "https://openrouter.ai/api/v1", "model": "openrouter/free", "env": "OPENROUTER_API_KEY"},
    "OpenAI": {"url": "https://api.openai.com/v1", "model": "gpt-4o-mini", "env": "OPENAI_API_KEY"},
}

TOOL_SPECS = {
    "search_tracklets": {"description": "Search indexed tracklets by appearance, category or camera.", "properties": {
        "class": {"type": "string", "enum": ["person", "vehicle", "bag"]}, "camera": {"type": "string"},
        "upper_color": {"type": "string"}, "lower_color": {"type": "string"}, "bag": {"type": "boolean"}}},
    "get_events": {"description": "Find recorded rule events and alerts.", "properties": {"rule_type": {"type": "string", "enum": ["zone", "line", "dwell", "crowd", "count", "speed", "wrong_direction", "unattended_object"]}, "camera": {"type": "string"}, "severity": {"type": "string", "enum": ["low", "medium", "high"]}}},
    "count_objects": {"description": "Count recorded line crossing events; not unique people.", "properties": {"method": {"type": "string", "enum": ["line"]}, "camera": {"type": "string"}}},
    "get_timeline": {"description": "List tracklet sightings in time order.", "properties": {"camera": {"type": "string"}}},
    "get_blindsight_gaps": {"description": "List unobserved interval hypotheses.", "properties": {}},
    "summarize_case": {"description": "Count case sources, tracklets, events and review backlog.", "properties": {}},
    "visual_search": {"description": "Find appearances similar to an existing tracklet ID.", "properties": {"track_id": {"type": "string"}}},
    "get_tracklet": {"description": "Inspect a specific existing tracklet ID.", "properties": {"track_id": {"type": "string"}}},
}


def definitions() -> list[dict[str, Any]]:
    return [{"type": "function", "function": {"name": name, "description": spec["description"],
             "parameters": {"type": "object", "properties": spec["properties"], "additionalProperties": False,
                            "required": ["track_id"] if name in {"visual_search", "get_tracklet"} else []}}}
            for name, spec in TOOL_SPECS.items()]


def test_connection(provider: str, api_key: str | None = None, *, model: str | None = None, client=None) -> bool:
    """Check a real tool call with a fixed test prompt, without case evidence."""
    from openai import OpenAI
    config = PROVIDERS[provider]
    credential = api_key or (os.environ.get(config["env"]) if config["env"] else "ollama")
    if not credential:
        raise ValueError("Set the API key for this provider first.")
    if client is None:
        client = OpenAI(base_url=config["url"], api_key=credential, timeout=30., max_retries=0)
    test_tools = type("TestTools", (), {"cameras": {}, "tracks": {}})()
    name, _ = plan_with_provider("Summarize this case", test_tools, provider=provider,
                                 model=model or config["model"], api_key=credential, client=client)
    if name != "summarize_case":
        raise ValueError("The model did not select the expected test tool.")
    return True


def plan_with_provider(question: str, tools, *, provider: str, model: str, api_key: str | None = None, client=None) -> tuple[str, dict]:
    """Request one tool call and validate it before accessing the case."""
    if provider not in PROVIDERS:
        raise ValueError("Unknown reasoning engine.")
    config = PROVIDERS[provider]
    credential = api_key or (os.environ.get(config["env"]) if config["env"] else "ollama")
    if not credential:
        raise ValueError(f"No key configured for {provider}.")
    if client is None:
        from openai import OpenAI
        client = OpenAI(base_url=config["url"], api_key=credential, timeout=8., max_retries=0)
    cameras = list(tools.cameras.values())
    response = client.chat.completions.create(model=model, messages=[
        {"role": "system", "content": "Choose one case-search tool for the user's question. Do not invent IDs, times, cameras or facts. Only use track IDs explicitly provided by the user. Return a tool call, never a prose answer. Known camera labels: " + json.dumps(cameras)},
        {"role": "user", "content": question[:600]}], tools=definitions(), tool_choice="auto")
    calls = response.choices[0].message.tool_calls or []
    if len(calls) != 1:
        raise ValueError("The reasoning engine did not select exactly one tool.")
    call = calls[0].function
    name = call.name
    if name not in TOOL_SPECS:
        raise ValueError("The reasoning engine selected an unsupported tool.")
    args = json.loads(call.arguments)
    allowed = TOOL_SPECS[name]["properties"]
    if not isinstance(args, dict) or set(args) - set(allowed):
        raise ValueError("Invalid tool arguments.")
    for key, value in args.items():
        field = allowed[key]
        if field["type"] == "string" and (not isinstance(value, str) or len(value) > 100):
            raise ValueError("Invalid tool parameter.")
        if field["type"] == "boolean" and not isinstance(value, bool):
            raise ValueError("Invalid tool parameter.")
        if "enum" in field and value not in field["enum"]:
            raise ValueError("Unsupported tool parameter.")
    if args.get("camera") and (args["camera"] not in cameras or args["camera"].lower() not in question.lower()):
        raise ValueError("Camera was not specified by the user or is not in the active case.")
    if name in {"get_tracklet", "visual_search"} and (args.get("track_id") not in tools.tracks or args["track_id"].lower() not in question.lower()):
        raise ValueError("Tracklet is not in the active case.")
    if name == "search_tracklets":
        args = {"filters": args}
    return name, args


def answer_with_engine(question: str, tools, working_set: list[str], *, provider: str, model: str,
                       api_key: str | None = None, client=None) -> dict:
    """Prefer the chosen planner; fall back to the offline parser on any failure."""
    safe_intent, _ = parse_intent(question, tools, working_set)
    if safe_intent in {"refuse", "clarify_time", "clarify_track", "clarify_second", "clarify_unsupported_appearance", "clarify_person_description", "clarify_colour"} or re.search(r"\d{1,2}:\d{2}", question) or any(w in question.lower().split() for w in ("them", "those", "these", "second")):
        return answer_offline(question, tools, working_set)
    try:
        intent, args = plan_with_provider(question, tools, provider=provider, model=model, api_key=api_key, client=client)
        result = answer_from_intent(intent, args, tools, working_set=working_set)
        result["engine"] = provider + " · " + model
        return result
    except Exception as exc:
        result = answer_offline(question, tools, working_set)
        result["fallback"] = f"{provider} unavailable or returned an invalid tool call ({type(exc).__name__}); used offline search."
        return result
