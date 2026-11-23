"""Local evidence tools for time-bounded live questions and blind-zone review."""
import re
import time
from datetime import datetime
from types import SimpleNamespace

from ...investigator import parse_intent
from .archive import utc
from .live_pipeline import inside
from .stream_manager import active_rules


def overdue(repo,archive,site,current):
    cameras = {c["id"]:c["label"] for c in repo.list(site,"camera")}
    portals = repo.list(site,"blind_portal")
    edges = repo.list(site,"transition")
    if not edges:
        return []
    start = current - min(3600,max(e["max_s"] for e in edges)+120)
    samples = {k:archive.samples(site,k,start,current) for k in cameras}
    rows = []
    for edge in edges:
        a,b = edge["from_camera"],edge["to_camera"]
        paths = [p for p in portals if p["camera_id"]==a and p["role"] in ("both","exit")]
        if not paths:
            continue
        exits = {}
        for ts,tracks,_ in samples[a]:
            for track in tracks:
                exits[track["id"]] = ts,track["center"]
        for tid,(ts,pt) in exits.items():
            if current-ts<max(3,edge["max_s"]) or not any(inside(pt,p) for p in paths):
                continue
            if not any(ts<s[0]<=ts+3 for s in samples[a]):
                continue
            low,high = ts+edge["min_s"],ts+edge["max_s"]
            window = [r for r in samples[b] if low-1<=r[0]<=high+1]
            if not window or max(y-x for x,y in zip([low]+[r[0] for r in window],[r[0] for r in window]+[high]))>2:
                continue
            # Any person at the destination could explain the interval.
            if any(r[1] for r in window):
                continue
            rows.append({"Departure UTC":utc(ts),"From":cameras[a],"Expected at":cameras[b],
                         "Expected by UTC":utc(high),"Track":tid,"Status":"Hypothesis · requires human review"})
    return sorted(rows,key=lambda r:r["Departure UTC"],reverse=True)[:50]


def answer(repo,manager,site,question,minutes,planner=None):
    cameras = {c["id"]:c["label"] for c in repo.list(site,"camera")}
    tools = SimpleNamespace(cameras=cameras, tracks={})
    intent,args = parse_intent(question,tools,[])
    text=question.lower()
    current=time.time()
    duration=re.search(r"last\s+(\d+)\s*(?:min|minute)",text)
    minutes=min(minutes,int(duration[1])) if duration else minutes
    start=current-minutes*60
    end=current
    if args.get("after_utc"):
        start=max(start,datetime.fromisoformat(args["after_utc"]).timestamp())
    if args.get("before_utc"):
        end=min(end,datetime.fromisoformat(args["before_utc"]).timestamp())
    window=f"{utc(start)} – {utc(end)}"
    rows=[]
    engine="Local live tools"
    if planner and intent not in {"refuse","clarify_time"} and not intent.startswith("clarify_") and "now" not in text:
        from ...investigator_engines import plan_with_provider
        try:
            name, params = plan_with_provider(question, tools, **planner)
            if name in {"get_events","count_objects","get_timeline","summarize_case"} and intent in {"clarify", name}:
                intent, args = name, {**params, **args}
                engine=planner["provider"]+" · "+planner["model"]
        except (ValueError,ImportError,RuntimeError,OSError):
            pass
        except Exception as exc:
            engine += " · planner unavailable ("+type(exc).__name__+")"
    filters=args.get("filters",{})
    if intent.startswith("clarify_") or intent=="refuse" or filters.get("class") in ("bag","vehicle") or filters.get("bag"):
        message="That detail cannot be established from the live index. Ask about a recorded flag or a configured zone."
    elif ("now" in text or "currently" in text) and ("zone" in text or intent=="search_tracklets"):
        intent="get_current_occupancy"
        rules=[r for r in active_rules(repo,site,include_disabled=True) if r["type"]=="zone"]
        zones=[r for r in rules if r["label"].lower() in text or any(word in text.split() for word in re.findall(r"[a-z0-9]+",r["label"].lower()) if word not in {"zone","entry","area","the","a"})]
        requested="zone" in text
        available=0
        exact=[r for r in zones if r["label"].lower() in text]
        if exact:
            zones=exact
        ambiguous=requested and len({r["label"].lower() for r in zones})>1
        if not requested or (zones and not ambiguous):
            for camera,label in cameras.items():
                if args.get("filters",{}).get("camera") not in (None,label):
                    continue
                camera_zones=[r for r in zones if r.get("camera_id") in (None,camera)]
                if requested and not camera_zones:
                    continue
                state=manager.snapshot(site,camera)
                if state["status"]!="Online" or not state["latest"]:
                    continue
                stamp,_,tracks,_,_=state["latest"]
                available+=1
                for t in tracks:
                    if requested and not any(inside(t["center"],r) for r in camera_zones):
                        continue
                    filters=args.get("filters",{})
                    if any(filters.get(k) and filters[k]!=t["attributes"].get(k) for k in ("upper_color","lower_color")):
                        continue
                    rows.append({"UTC":utc(stamp),"Camera":label,"Local track":t["id"],"Upper clothing":t["attributes"].get("upper_color")})
        if ambiguous:
            message="Several named zones fit that wording. Enter the full zone rule name."
        elif requested and not zones:
            message="I couldn’t locate that zone. Use its configured zone rule name."
        elif not available:
            message="No current camera observation covers this request."
        else:
            message=f"{len(rows)} visible person track(s) in the latest sampled frames."
    elif intent in {"get_events","count_objects","summarize_case","get_timeline"}:
        rule_type=args.get("rule_type") or args.get("method")
        severity=next((s for s in ("critical","high","medium","low") if re.search(r"\b"+s+r"\b",text)),None)
        for e in reversed(repo.list(site,"event")):
            if not start<=e["stamp"]<=end or (rule_type and e["method"]!=rule_type):
                continue
            label=cameras.get(e["camera_id"],"Unknown")
            if (args.get("camera") and args["camera"]!=label) or (severity and severity!=e["severity"]):
                continue
            rows.append({"UTC":e["time_utc"],"Camera":label,"Rule":e["rule"],"Severity":e["severity"],"Evidence ID":e["id"]})
        message=f"{len(rows)} matching rule flag(s) recorded in the last {minutes} minutes."
    else:
        message="Ask about rule flags, line crossings or the latest occupancy in a named zone."
    return dict(answer=message,tool=intent,window=window,rows=rows[:30],engine=engine)
