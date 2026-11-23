import pytest

from fedsight.demo import create_demo
from fedsight.investigator import InvestigatorTools, answer_offline, parse_intent
from fedsight.service import FedSight


@pytest.fixture
def investigator(tmp_path):
    service = FedSight(tmp_path)
    cid = create_demo(service)
    return service, cid, InvestigatorTools(service, cid)


@pytest.mark.parametrize("question,expected", [
    ("Summarize this case", "summarize_case"),
    ("Show restricted zone intrusions", "get_events"),
    ("Show alerts from Corridor", "get_events"),
    ("How many crossed the line?", "count_objects"),
    ("Show the timeline", "get_timeline"),
    ("What happened between the Atrium and the Exit?", "get_blindsight_gaps"),
    ("Find a person wearing a blue shirt", "search_tracklets"),
    ("Who is this person?", "refuse"),
    ("Who entered after 10:01?", "clarify_time"),
])
def test_intent_selection(investigator, question, expected):
    _, _, tools = investigator
    assert parse_intent(question, tools)[0] == expected


def test_tools_are_scoped_and_evidence_is_grounded(investigator):
    service, cid, tools = investigator
    answer = answer_offline("show the timeline", tools)
    assert answer["evidence"]
    assert all(item["id"] in tools.tracks for item in answer["evidence"])
    assert "Visual match is a lead" in answer["answer"]
    other = service.repo.create_case("other", "", "tester")["id"]
    with pytest.raises(ValueError):
        InvestigatorTools(service, other).get_tracklet(answer["evidence"][0]["id"])
    followup = answer_offline("show the second one", tools, answer["working_set"])
    assert followup["evidence"][0]["id"] == answer["working_set"][1]


def test_events_and_counts_distinguish_flags_from_people(investigator):
    _, _, tools = investigator
    result = tools.get_events(camera="Corridor")
    assert all(e["camera"] == "Corridor" for e in result["evidence"])
    count = answer_offline("How many crossed the line?", tools)
    assert "not a count of unique people" in count["answer"]
    assert tools.summarize_case()["counts"]["tracklets"] == 10


def test_explicit_utc_filter_and_clock_ambiguity(investigator):
    _, _, tools = investigator
    intent, args = parse_intent("show alerts after 2026-01-01T10:01:00Z", tools)
    assert intent == "get_events" and args["after_utc"].endswith("+00:00")
    assert parse_intent("show alerts after 10:01", tools)[0] == "clarify_time"
    selected = tools.get_events(after_utc="2026-01-01T10:00:50+00:00")
    assert all(e["start_utc"] is not None for e in selected["evidence"])


def test_provider_tool_call_is_validated_and_answer_uses_case_data(investigator):
    from fedsight.investigator_engines import answer_with_engine
    _, _, tools = investigator
    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    func = type("Function", (), {"name": "summarize_case", "arguments": "{}"})()
                    call = type("Call", (), {"function": func})()
                    message = type("Message", (), {"tool_calls": [call]})()
                    return type("Response", (), {"choices": [type("Choice", (), {"message": message})()]})()
    result = answer_with_engine("Summarize this case", tools, [], provider="Local (Ollama)", model="test", client=Client())
    assert result["engine"] == "Local (Ollama) · test"
    assert "10 indexed tracklets" in result["answer"]
    assert result["tool_calls"][0]["name"] == "summarize_case"
    class Bad:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("unavailable")
    fallback = answer_with_engine("Summarize this case", tools, [], provider="Local (Ollama)", model="test", client=Bad())
    assert fallback["engine"] == "Built-in offline" and "unavailable" in fallback["fallback"]


def test_unindexed_appearance_does_not_return_unrelated_tracks(investigator):
    from fedsight.investigator_engines import answer_with_engine
    _, _, tools = investigator
    for question in ("give man with long hair", "man with long haie", "person with a blue jacket and glasses"):
        result = answer_with_engine(question, tools, [], provider="Local (Ollama)", model="test")
        assert not result["evidence"] and not result["tool_calls"]
        assert "can't reliably" in result["answer"]


def test_broad_person_search_is_limited_without_claiming_rank(investigator):
    _, _, tools = investigator
    result = answer_offline("show people", tools)
    assert len(result["evidence"]) <= 5
    assert "not ranked" in result["answer"]
    assert "track_" not in result["answer"]


def test_connection_checks_actual_model_tool_call():
    from fedsight.investigator_engines import test_connection as check_connection
    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    assert kwargs["model"] == "selected-model"
                    func = type("Function", (), {"name": "summarize_case", "arguments": "{}"})()
                    call = type("Call", (), {"function": func})()
                    msg = type("Message", (), {"tool_calls": [call]})()
                    return type("Response", (), {"choices": [type("Choice", (), {"message": msg})()]})()
    assert check_connection("Cloud planner (Gemini)", "test-key", model="selected-model", client=Client())


def test_person_with_clothing_search_not_intercepted_by_gender(investigator):
    _, _, tools = investigator
    intent, args = parse_intent("find a man wearing a blue shirt", tools)
    assert intent == "search_tracklets"
    assert args["filters"]["upper_color"] == "blue"
    assert args["filters"]["class"] == "person"
    answer = answer_offline("find a man wearing a blue shirt", tools)
    assert answer["evidence"]
    assert "not ranked" in answer["answer"]


def test_bare_gender_search_clarifies(investigator):
    _, _, tools = investigator
    for question in ("find a man", "show the woman", "who is the girl"):
        intent, _ = parse_intent(question, tools)
        assert intent == "clarify_person_description"


def test_visual_search_typo_normalization(investigator):
    _, _, tools = investigator
    intent, _ = parse_intent("find a person with glases", tools)
    assert intent == "clarify_unsupported_appearance"

