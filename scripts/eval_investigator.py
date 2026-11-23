"""Run a deterministic intent and evidence-grounding check on synthetic data."""
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedsight.demo import create_demo
from fedsight.investigator import InvestigatorTools, answer_offline, parse_intent
from fedsight.service import FedSight

CASES = [
    ("Summarize this case", "summarize_case"), ("Give an overview", "summarize_case"),
    ("Show the timeline", "get_timeline"), ("Show sequence in Corridor", "get_timeline"),
    ("Show restricted zone events", "get_events"), ("Any intrusions?", "get_events"),
    ("Show alerts from Corridor", "get_events"), ("List event alerts", "get_events"),
    ("How many crossed the line?", "count_objects"), ("Count line crossings", "count_objects"),
    ("Find a person in blue shirt", "search_tracklets"), ("Show people with a red top", "search_tracklets"),
    ("Find a vehicle", "search_tracklets"), ("Show anyone with a bag", "search_tracklets"),
    ("Show a person near Atrium", "search_tracklets"), ("Find blue jacket", "search_tracklets"),
    ("Show blind spot hypotheses", "get_blindsight_gaps"), ("What happened between the Atrium and the Exit?", "get_blindsight_gaps"),
    ("Show unobserved intervals", "get_blindsight_gaps"), ("Who is this person?", "refuse"),
    ("Who is guilty?", "refuse"), ("Who entered after 10:01?", "clarify_time"),
    ("Find anyone who looks like this", "clarify_track"), ("Tell me a joke", "clarify"),
    ("Show the second one", "clarify_second"),
]


def main():
    with tempfile.TemporaryDirectory() as directory:
        service = FedSight(Path(directory))
        case_id = create_demo(service)
        tools = InvestigatorTools(service, case_id)
        correct = grounded = 0
        event_ids = {e["id"] for e in tools.events}
        hypothesis_ids = {h["id"] for h in service.active_blind_hypotheses(case_id)}
        for question, expected in CASES:
            intent, _ = parse_intent(question, tools)
            result = answer_offline(question, tools)
            correct += intent == expected
            ok = True
            for item in result["evidence"]:
                if item["type"] == "tracklet":
                    original = tools.tracks.get(item["id"])
                    ok &= bool(original and item["start_s"] == original["start_time"] and item["start_utc"] == tools._utc(original["source_id"], original["start_time"]))
                elif item["type"] == "event":
                    event = next((e for e in tools.events if e["id"] == item["id"]), None)
                    ok &= bool(event and item["id"] in event_ids and item["start_s"] == event["start_time"] and item["start_utc"] == tools._utc(event["source_id"], event["start_time"]))
                else:
                    ok &= item["id"] in hypothesis_ids
            grounded += bool(ok)
            if intent != expected or not ok:
                print(f"FAIL: {question!r}: expected {expected}, got {intent}, grounded={ok}")
        print(f"Tool selection: {correct}/{len(CASES)}; evidence grounding: {grounded}/{len(CASES)}")
        return 0 if correct == len(CASES) and grounded == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
