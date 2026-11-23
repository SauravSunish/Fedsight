"""Traceable exports from stored evidence, never generated case narratives."""
from __future__ import annotations
import csv
import io
import json
import zipfile
from html import escape
from .core import VERSION, canonical, now, digest


def csv_bytes(rows):
    if not rows:
        return b""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        safe = {}
        for key, value in row.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=True)
            if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")):
                value = "'" + value
            safe[key] = value
        writer.writerow(safe)
    return stream.getvalue().encode("utf-8-sig")


def case_snapshot(service, case_id):
    repo = service.repo
    case = repo.get(case_id, case_id, "case")
    tracks = service.active_tracks(case_id)
    sources = repo.list(case_id, "source")
    source_map = {s["id"]: s for s in sources}
    track_rows = [{k: v for k, v in t.items() if k not in {"embedding", "embeddings", "quality", "trajectory", "preview_asset"}} for t in tracks]
    digest_items = [{"track_id": t["id"], "source_id": t["source_id"], "source_name": source_map[t["source_id"]]["name"],
                     "source_sha256": source_map[t["source_id"]].get("sha256"), "start_frame": t.get("first_frame"), "end_frame": t.get("last_frame"),
                     "start_time": t["start_time"], "end_time": t["end_time"], "time_anchor": source_map[t["source_id"]].get("time_anchor"),
                     "preview_sha256": t.get("preview_asset", {}).get("sha256")} for t in tracks]
    audit = repo.verify()
    assets = service.verify_assets(case_id, "export-verifier")
    return {"schema_version": VERSION, "generated_at": now(), "case": case, "sources": sources, "tracks": track_rows,
            "proposals": repo.active_proposals(case_id), "all_proposals": repo.list(case_id, "proposal"),
            "reviews": repo.list(case_id, "review"), "notes": repo.list(case_id, "note"), "bookmarks": repo.list(case_id, "bookmark"),
            "events": repo.list(case_id, "event"), "event_reviews": repo.list(case_id, "event_review"),
            "cameras": repo.list(case_id, "camera"), "transitions": repo.list(case_id, "transition"), "rules": repo.list(case_id, "rule"),
            "rule_states": repo.list(case_id, "rule_state"),
            "coverage_regions": repo.list(case_id, "coverage_region"), "blind_portals": repo.list(case_id, "blind_portal"),
            "source_alignments": repo.list(case_id, "source_alignment"),
            "blind_runs": repo.list(case_id, "blind_run"), "blind_hypotheses": repo.list(case_id, "blind_hypothesis"),
            "blind_reviews": repo.list(case_id, "blind_review"), "blind_clues": repo.list(case_id, "blind_clue"),
            "chat_searches": repo.list(case_id, "chat_search"),
            "investigator_sessions": repo.list(case_id, "investigator_session"),
            "investigator_messages": repo.list(case_id, "investigator_message"),
            "investigator_feedback": repo.list(case_id, "investigator_feedback"),
            "runs": repo.list(case_id, "run"), "jobs": repo.list(case_id, "job"),
            "digest": digest_items, "assets": repo.list(case_id, "asset"), "asset_verification": assets,
            "audit_verification": audit, "audit_events": [e for e in repo.audit_events() if e["case_id"] == case_id],
            "checkpoint": repo.checkpoint() if audit["ok"] else None}


def pdf_report(snapshot):
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    output = io.BytesIO()
    styles = getSampleStyleSheet()
    story = []
    def p(text, style="BodyText"):
        story.append(Paragraph(escape(str(text)), styles[style]))
        story.append(Spacer(1, 7))
    case = snapshot["case"]
    p("FEDSIGHT / CASE EVIDENCE REPORT", "Title")
    p(case["title"], "Heading1")
    p("SYNTHETIC DEMONSTRATION — NOT REAL EVIDENCE" if case["demo"] else "Human-reviewed appearance linkage; no claim of real-world identity.")
    p(f"Case: {case['id']} | Status: {case['status']} | Exported: {snapshot['generated_at']}")
    p(case["description"])
    p(f"Sources: {len(snapshot['sources'])} | Tracklets: {len(snapshot['tracks'])} | Active proposals: {len(snapshot['proposals'])} | Review actions: {len(snapshot['reviews'])}")
    p("Integrity and provenance", "Heading2")
    p(f"Audit chain: {'PASS' if snapshot['audit_verification']['ok'] else 'FAIL'}; verified assets: {sum(a['ok'] for a in snapshot['asset_verification'])}/{len(snapshot['asset_verification'])}.")
    p("Verify the supplied manifest and checkpoint independently. Hashes and keyed chains do not establish legal admissibility or protect a compromised key holder.")
    p("Original sources", "Heading2")
    for source in snapshot["sources"]:
        p(source["name"], "Heading3")
        p(f"Source ID: {source['id']}")
        p(f"SHA-256: {source.get('sha256') or 'Synthetic source: no original file'}")
        p("Time metadata: " + json.dumps(source.get("time_anchor", {})))
    p("Candidate decisions", "Heading2")
    latest = {r["proposal_id"]: r for r in snapshot["reviews"]}
    for n, proposal in enumerate(snapshot["proposals"], 1):
        review = latest.get(proposal["id"], {})
        p(f"{n}. {proposal['id']}", "Heading3")
        p(f"Left: {proposal['left_id']} / Right: {proposal['right_id']}")
        p(f"Cosine: {proposal['features']['values']['similarity']:.4f} | Model assessment: {proposal['verdict']['state']} | Human decision: {review.get('decision', 'pending')}")
        p(f"Model version: {proposal['model_version']} | Run: {proposal['run_id']}")
        p("Assessment reasons: " + " ".join(proposal['verdict']['reasons']))
        if review:
            p(f"Reviewer: {review['actor']} | Time: {review['created_at']} | Reason: {review['reason']}")
    p("Evidence digest: exact source references", "Heading2")
    for item in snapshot["digest"]:
        p(f"{item['track_id']} | {item['source_name']} | clip-relative {item['start_time']:.2f}–{item['end_time']:.2f}s | frames {item['start_frame']}–{item['end_frame']}")
    if snapshot["blind_hypotheses"]:
        p("BlindSight video-only hypotheses", "Heading2")
        blind_reviews = {r["hypothesis_id"]: r for r in snapshot["blind_reviews"]}
        for hypothesis in snapshot["blind_hypotheses"]:
            p(f"{hypothesis['id']} | {hypothesis['assessment']} | review: {blind_reviews.get(hypothesis['id'], {}).get('decision', 'pending')}")
            p(f"Tracks: {hypothesis['left_id']} / {hypothesis['right_id']} | zones: {', '.join(z['label'] for z in hypothesis['zones']) or 'unmapped'}")
            p("Supporting: " + " ".join(hypothesis["supporting_evidence"]))
            p("Contradictory or missing: " + " ".join(hypothesis["contradictory_evidence"]))
            p("Unobserved route hypothesis only; presence and identity are not established.")
    if snapshot["chat_searches"]:
        p("Investigation chat searches", "Heading2")
        for search in snapshot["chat_searches"]:
            p(f"{search['id']} | {search['engine']} | {search['created_at']}")
            p(f"Description: {search['query']} | indexed results: {len(search['result_track_ids'])}")
    p("Processing provenance", "Heading2")
    for job in snapshot["jobs"]:
        p(f"Job {job['id']} / {job['status']} / source {job['source_id']}")
        p(json.dumps(job.get("model_manifest", {})))
        if job.get("meta", {}).get("incomplete"):
            p("Incomplete source processing: " + str(job['meta'].get('stop_reason')))
    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#52657a"))
        canvas.drawString(36, 23, "FedSight | Source-linked review record")
        canvas.drawRightString(559, 23, f"Page {doc.page}")
    SimpleDocTemplate(output, pagesize=(595, 842), rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=42).build(story, onFirstPage=footer, onLaterPages=footer)
    return output.getvalue()


def export_bundle(service, case_id, actor):
    snapshot = case_snapshot(service, case_id)
    if not snapshot["audit_verification"]["ok"] or not all(x["ok"] for x in snapshot["asset_verification"]):
        raise ValueError("Export blocked because integrity verification failed. Inspect the audit workspace.")
    output = io.BytesIO()
    files = {"case.json": canonical(snapshot), "case-report.pdf": pdf_report(snapshot),
             "tracklets.csv": csv_bytes(snapshot["tracks"]), "evidence-digest.csv": csv_bytes(snapshot["digest"]),
             "reviews.csv": csv_bytes(snapshot["reviews"]), "proposals.csv": csv_bytes(snapshot["proposals"]),
             "blind-hypotheses.csv": csv_bytes(snapshot["blind_hypotheses"]),
             "blind-reviews.csv": csv_bytes(snapshot["blind_reviews"]), "blind-clues.csv": csv_bytes(snapshot["blind_clues"]),
             "source-alignments.csv": csv_bytes(snapshot["source_alignments"]),
             "chat-searches.csv": csv_bytes(snapshot["chat_searches"]),
             "audit-checkpoint.json": canonical(snapshot["checkpoint"])}
    for track in service.active_tracks(case_id):
        if track.get("preview_asset"):
            files[f"previews/{track['id']}.png"] = service.read_asset(track["preview_asset"], actor)
    for asset in snapshot["assets"]:
        if asset.get("role") == "annotated_clip":
            files[f"clips/{asset['id']}.mp4"] = service.read_asset(asset, actor)
    files["manifest.json"] = canonical({name: digest(blob) for name, blob in files.items()})
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, blob in files.items():
            archive.writestr(name, blob)
    service.repo.audit(case_id, actor, "case.exported", evidence_hash=digest(output.getvalue()))
    return output.getvalue()
