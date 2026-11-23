"""Grounded, case-scoped investigation tools and deterministic offline answers."""
from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Any

from .core import TimeAnchor
from .investigation_chat import query_filters
from .review_video import similar_tracks


class InvestigatorTools:
    """Read only evidence from one case; return IDs that can be verified locally."""
    def __init__(self, service, case_id: str):
        self.service, self.case_id = service, case_id
        repo = service.repo
        self.case = repo.get(case_id, case_id, "case")
        self.sources = {s["id"]: s for s in repo.list(case_id, "source")}
        for sid, alignment in service.blind_alignments(case_id).items():
            if sid in self.sources:
                self.sources[sid] = {**self.sources[sid], "camera_id": alignment["camera_id"], "time_anchor": alignment["time_anchor"]}
        self.cameras = {c["id"]: c["label"] for c in repo.list(case_id, "camera")}
        self.tracks = {t["id"]: t for t in service.active_tracks(case_id)}
        self.rules = {r["id"]: r for r in repo.list(case_id, "rule")}
        self.events = repo.list(case_id, "event")
        self.event_reviews = {r["event_id"]: r for r in repo.list(case_id, "event_review")}

    def _camera(self, source_id: str) -> str:
        source = self.sources.get(source_id)
        return self.cameras.get(source.get("camera_id") if source else None, "Unknown camera")

    def _utc(self, source_id: str, seconds: float) -> str | None:
        source = self.sources.get(source_id)
        if not source:
            return None
        bounds = TimeAnchor(**source.get("time_anchor", {})).interval(seconds)
        return datetime.fromtimestamp(sum(bounds) / 2, timezone.utc).isoformat() if bounds else None

    def _within_utc(self, source_id: str, seconds: float, after: str | None, before: str | None) -> bool:
        if not after and not before:
            return True
        source = self.sources.get(source_id)
        if not source:
            return False
        bounds = TimeAnchor(**source.get("time_anchor", {})).interval(seconds)
        if bounds is None:
            return False
        low = datetime.fromisoformat(after).timestamp() if after else float("-inf")
        high = datetime.fromisoformat(before).timestamp() if before else float("inf")
        return bounds[1] >= low and bounds[0] <= high

    def track_evidence(self, track: dict) -> dict:
        source = self.sources.get(track.get("source_id"), {"id": track.get("source_id"), "name": "Unknown source"})
        return {"id": track["id"], "type": "tracklet", "track_id": track["id"], "source_id": source.get("id"),
                "source": source.get("name", "Unknown source"), "camera": self._camera(track.get("source_id")), "start_s": track["start_time"],
                "end_s": track["end_time"], "start_utc": self._utc(track.get("source_id"), track["start_time"]),
                "end_utc": self._utc(track.get("source_id"), track["end_time"]), "class": track.get("object_class", "person"),
                "clock_uncertainty_s": source.get("time_anchor", {}).get("uncertainty_s", 0),
                "attributes": track.get("attributes", {}), "duration_s": track.get("duration_s", 0),
                "source_hash": source.get("sha256"), "preview_asset": track.get("preview_asset")}

    def search_tracklets(self, filters: dict | None = None, *, working_set: list[str] | None = None) -> dict:
        filters = filters or {}
        result = []
        for track in self.tracks.values():
            if working_set is not None and track["id"] not in working_set:
                continue
            attr = track.get("attributes", {})
            if filters.get("class") and track.get("object_class", "person") != filters["class"]:
                continue
            if filters.get("camera") and filters["camera"].lower() not in self._camera(track["source_id"]).lower():
                continue
            if filters.get("upper_color") and attr.get("upper_color") != filters["upper_color"]:
                continue
            if filters.get("lower_color") and attr.get("lower_color") != filters["lower_color"]:
                continue
            if filters.get("bag") and attr.get("bag") != "possible nearby bag":
                continue
            if filters.get("min_quality") and track.get("embedding_meta", {}).get("mean_quality", 0) < filters["min_quality"]:
                continue
            if filters.get("min_dwell") and track.get("duration_s", 0) < filters["min_dwell"]:
                continue
            if filters.get("after_utc") or filters.get("before_utc"):
                if not self._within_utc(track["source_id"], track["start_time"], filters.get("after_utc"), filters.get("before_utc")):
                    continue
            result.append(self.track_evidence(track))
        result.sort(key=lambda x: (x["source"], x["start_s"], x["id"]))
        return {"tool": "search_tracklets", "filters": filters, "total": len(result), "evidence": result[:50],
                "note": "Appearance attributes and bag proximity are approximate; missing values are not negatives. Time filters exclude unaligned sources and include overlapping uncertainty bounds."}

    def get_events(self, *, rule_type: str | None = None, camera: str | None = None, severity: str | None = None,
                   after_utc: str | None = None, before_utc: str | None = None) -> dict:
        evidence = []
        for event in self.events:
            source = self.sources.get(event["source_id"])
            if not source or (rule_type and event.get("method") != rule_type):
                continue
            if camera and camera.lower() not in self._camera(source["id"]).lower():
                continue
            if severity and event.get("severity", "medium") != severity:
                continue
            if not self._within_utc(source["id"], event["start_time"], after_utc, before_utc):
                continue
            track = self.tracks.get(event.get("track_id"))
            evidence.append({"id": event["id"], "type": "event", "track_id": event.get("track_id"),
                             "source_id": source["id"], "source": source["name"], "camera": self._camera(source["id"]),
                             "start_s": event["start_time"], "end_s": event.get("end_time", event["start_time"]),
                             "start_utc": self._utc(source["id"], event["start_time"]),
                             "clock_uncertainty_s": source.get("time_anchor", {}).get("uncertainty_s", 0),
                             "rule": self.rules.get(event["rule_id"], {}).get("label", event["method"]),
                             "method": event["method"], "severity": event.get("severity", "medium"),
                             "review": self.event_reviews.get(event["id"], {}).get("decision", "pending"),
                             "preview_asset": track.get("preview_asset") if track else None,
                             "source_hash": source.get("sha256")})
        evidence.sort(key=lambda x: (x["start_utc"] is None, x["start_utc"] or "", x["source"], x["start_s"]))
        return {"tool": "get_events", "total": len(evidence), "evidence": evidence[:50],
                "note": "Flags are derived from sampled detections and require review. Time filters exclude unaligned sources and include overlapping uncertainty bounds."}

    def get_tracklet(self, track_id: str) -> dict:
        track = self.tracks.get(track_id)
        if track is None:
            raise ValueError("Tracklet is not active in this case.")
        return {"tool": "get_tracklet", "total": 1, "evidence": [self.track_evidence(track)]}

    def visual_search(self, track_id: str) -> dict:
        if track_id not in self.tracks:
            raise ValueError("Select an indexed tracklet in this case.")
        found = similar_tracks(self.tracks[track_id], list(self.tracks.values()), limit=20)
        return {"tool": "visual_search", "total": len(found), "evidence": [dict(self.track_evidence(t), similarity=round(score, 4)) for t, score in found],
                "note": "Existing descriptor cosine; uncalibrated and not identity probability. Text/image CLIP is not configured."}

    def get_timeline(self, camera: str | None = None) -> dict:
        evidence = [self.track_evidence(t) for t in self.tracks.values()
                    if not camera or camera.lower() in self._camera(t["source_id"]).lower()]
        evidence.sort(key=lambda x: (x["start_utc"] is None, x["start_utc"] or "", x["source"], x["start_s"]))
        return {"tool": "get_timeline", "total": len(evidence), "evidence": evidence[:50],
                "note": "Unaligned source times are clip-relative and cannot establish cross-camera order."}

    def get_blindsight_gaps(self) -> dict:
        evidence = [{"id": h["id"], "type": "hypothesis", "track_id": h.get("from_track_id"),
                     "assessment": h["assessment"], "zones": [z["label"] for z in h["zones"]],
                     "gap_utc": h["gap_utc"], "meaning": h["meaning"]} for h in self.service.active_blind_hypotheses(self.case_id)]
        return {"tool": "get_blindsight_gaps", "total": len(evidence), "evidence": evidence[:50],
                "note": "Unobserved intervals are hypotheses, not observed presence."}

    def count_objects(self, *, method: str | None = None, camera: str | None = None,
                      after_utc: str | None = None, before_utc: str | None = None) -> dict:
        result = self.get_events(rule_type=method, camera=camera, after_utc=after_utc, before_utc=before_utc)
        result["tool"] = "count_objects"
        result["note"] = "Count of recorded rule events, not deduplicated people or all physical crossings."
        return result

    def summarize_case(self) -> dict:
        proposals = self.service.repo.active_proposals(self.case_id)
        reviews = self.service.repo.latest_reviews(self.case_id)
        return {"tool": "summarize_case", "counts": {"sources": len(self.sources), "tracklets": len(self.tracks),
                "events": len(self.events), "pending_proposals": sum(reviews.get(p["id"], {}).get("decision") not in {"accept", "reject"} for p in proposals)},
                "evidence": self.get_events()["evidence"][:8], "note": "Counts are recorded tracklets and flags, not unique people."}


def normalize_description(question: str) -> str:
    """Correct a small set of unambiguous visual-search typos before routing."""
    text = re.sub(r"\b(?:haie|haeir|haiir|hairs)\b", "hair", question, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:glases|glassses|specticles)\b", "glasses", text, flags=re.IGNORECASE)
    return text


def parse_intent(question: str, tools: InvestigatorTools, working_set: list[str] | None = None) -> tuple[str, dict]:
    text = normalize_description(question).lower().strip()
    if not text or len(text) > 600:
        raise ValueError("Enter a question of 1–600 characters.")
    if re.search(r"\b(criminal|guilty|terrorist|ethnicity|race|religion|identify by face|who is this|name of (?:the |this )?person)\b", text):
        return "refuse", {}
    # These visual details are not indexed. Never silently drop them and present
    # unrelated tracklets as if they matched the complete description.
    if re.search(r"\b(hair|haired|bald|beard|bearded|moustache|mustache|hat|cap|glasses|spectacles|sunglasses|mask|shoes|boots|umbrella|phone|logo|tattoo|striped|stripes|dress)\b", text):
        return "clarify_unsupported_appearance", {}
    # Clock times require an explicit date and offset; never infer an alignment.
    stamps = re.findall(r"\b\d{4}-\d{2}-\d{2}[t ]\d{2}:\d{2}(?::\d{2})?(?:z|[+-]\d{2}:\d{2})", text)
    time_filter = {}
    if stamps or re.search(r"\b\d{1,2}:\d{2}\b", text):
        if not stamps or ("between" in text and len(stamps) < 2):
            return "clarify_time", {}
        try:
            converted = [datetime.fromisoformat(s.upper().replace("Z", "+00:00")).astimezone(timezone.utc).isoformat() for s in stamps[:2]]
        except ValueError:
            return "clarify_time", {}
        if len(converted) > 1:
            if converted[0] > converted[1]:
                return "clarify_time", {}
            time_filter = {"after_utc": converted[0], "before_utc": converted[1]}
        elif "before" in text:
            time_filter = {"before_utc": converted[0]}
        else:
            time_filter = {"after_utc": converted[0]}
    camera = next((label for label in tools.cameras.values() if re.search(r"\b" + re.escape(label.lower()) + r"\b", text)), None)
    if "second" in text:
        return ("get_tracklet", {"track_id": working_set[1]}) if working_set and len(working_set) > 1 else ("clarify_second", {})
    if any(w in text for w in ("blind spot", "blindspot", "blind sight", "blindsight", "between the", "unobserved")):
        return "get_blindsight_gaps", {}
    if any(w in text for w in ("similar", "looks like", "look like")):
        match = re.search(r"track_[a-z0-9]+", text)
        return ("visual_search", {"track_id": match.group(0)}) if match else ("visual_search", {"track_id": working_set[0]}) if working_set else ("clarify_track", {})
    if any(w in text for w in ("intrusion", "intruder", "restricted zone", "entered the zone", "entered the restricted")):
        return "get_events", {"rule_type": "zone", "camera": camera, **time_filter}
    if any(w in text for w in ("alert", "event", "incident")):
        return "get_events", {"camera": camera, **time_filter}
    if "cross" in text and any(w in text for w in ("count", "how many", "number")):
        return "count_objects", {"method": "line", "camera": camera, **time_filter}
    if "timeline" in text or "sequence" in text:
        return "get_timeline", {"camera": camera}
    if "summar" in text or "overview" in text:
        return "summarize_case", {}
    if any(w in text for w in ("person", "people", "man", "woman", "men", "women", "male", "female", "boy", "girl",
                               "bag", "backpack", "vehicle", "car", "tracklet", "them", "shirt", "top", "jacket")):
        parsed = query_filters(question)
        if parsed["colour"] and not (parsed["upper"] or parsed["lower"]):
            return "clarify_colour", {}
        filters = {"class": parsed["category"], "upper_color": parsed["upper"], "lower_color": parsed["lower"],
                   "bag": any(w in text for w in ("bag", "backpack", "suitcase")), "camera": camera, **time_filter}
        has_clue = bool(parsed["upper"] or parsed["lower"] or camera or time_filter or filters["bag"] or parsed["category"] == "vehicle")
        if re.search(r"\b(man|woman|men|women|male|female|boy|girl)\b", text) and not has_clue:
            return "clarify_person_description", {}
        return "search_tracklets", {"filters": {k: v for k, v in filters.items() if v},
                                     "working_set": working_set if working_set and re.search(r"\b(them|those|these)\b", text) else None}
    return "clarify", {}


def answer_offline(question: str, tools: InvestigatorTools, working_set: list[str] | None = None) -> dict:
    start = time.perf_counter()
    intent, args = parse_intent(question, tools, working_set)
    return answer_from_intent(intent, args, tools, working_set=working_set, started=start)


def answer_from_intent(intent: str, args: dict, tools: InvestigatorTools, *, working_set=None, started=None) -> dict:
    """Create answer text only from the case tool's structured output."""
    start = started if started is not None else time.perf_counter()
    clarifications = {"refuse": "I can help search visible appearance and recorded events, but cannot infer identity, criminality or sensitive traits.",
                      "clarify_unsupported_appearance": "I can't reliably search for hair length or other unindexed details in these videos yet, so I can't identify a correct match from that description. Try an indexed clue such as a blue shirt or black trousers, or review the original footage. Clothing colours are approximate.",
                      "clarify_person_description": "I can't reliably determine a person's gender from the indexed footage. Try searching by an approximate clothing colour and camera instead, then review the original video.",
                      "clarify_colour": "Which part of the clothing has that colour—for example, a blue shirt or blue trousers? I can't reliably search a colour without knowing the clothing region.",
                      "clarify_time": "Is that a clip-relative time or a date and time with a UTC offset? Please specify the recording and date for a clock-time search.",
                      "clarify_track": "Which tracklet ID should I compare?",
                      "clarify_second": "There is no second tracklet in the current working set.",
                      "clarify": "Would you like to search tracklets, inspect alerts, review a timeline, or summarize this case?"}
    if intent in clarifications:
        return {"answer": clarifications[intent], "tool_calls": [], "evidence": [], "working_set": working_set or [], "engine": "Built-in offline", "latency_ms": round((time.perf_counter()-start)*1000)}
    result = getattr(tools, intent)(**args)
    evidence = result.get("evidence", [])
    if intent == "summarize_case":
        c = result["counts"]
        answer = f"This case contains {c['sources']} sources, {c['tracklets']} indexed tracklets, {c['events']} recorded events, and {c['pending_proposals']} proposals awaiting resolution."
    elif intent == "count_objects":
        answer = f"The case contains {result['total']} recorded line-crossing event(s) for the selected filters. This is an event count, not a count of unique people."
    elif evidence:
        if intent == "search_tracklets":
            total = result.get("total", len(evidence))
            evidence = evidence[:5]
            gender_note = " Note that gender is not assessed; all matching person tracklets are included." if args.get("filters", {}).get("class") == "person" else ""
            answer = (f"Found {total} matching tracklet(s); showing {len(evidence)} in recording order. "
                      "These are not ranked by likelihood and can include multiple tracklets of the same person. "
                      "Add a camera or clothing colour to narrow the search. Clothing colour and bag proximity are approximate."
                      f"{gender_note}")
        else:
            answer = f"Found {result.get('total', len(evidence))} case record(s). {result.get('note', '')}"
    else:
        answer = f"No matching records were found. {result.get('note', '')} Try widening the filters or processing additional footage."
    answer += " Visual match is a lead, not a confirmed identity."
    return {"answer": answer, "tool_calls": [{"name": intent, "arguments": args, "result": result}], "evidence": evidence,
            "working_set": [e["track_id"] for e in evidence if e.get("track_id") in tools.tracks],
            "engine": "Built-in offline", "latency_ms": round((time.perf_counter()-start)*1000)}
