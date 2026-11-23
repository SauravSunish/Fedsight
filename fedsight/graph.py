"""Evidence graph projections and allowlisted, parameterised graph queries."""
from __future__ import annotations
import json
import networkx as nx
from .core import digest, canonical


def evidence_graph(service, case_id):
    repo = service.repo
    graph = nx.Graph()
    tracks = service.active_tracks(case_id)
    for track in tracks:
        graph.add_node(track["id"], source_id=track["source_id"], camera_id=track.get("camera_id"))
    reviews = repo.latest_reviews(case_id)
    for proposal in repo.active_proposals(case_id):
        review = reviews.get(proposal["id"], {})
        if review.get("decision") == "accept":
            graph.add_edge(proposal["left_id"], proposal["right_id"], proposal_id=proposal["id"], review_id=review["id"])
    groups = [{"dwi_id": "DWI-" + digest(canonical(sorted(group)))[:12], "track_ids": sorted(group),
               "meaning": "Human-linked appearance group; not a real-world identity."} for group in nx.connected_components(graph)]
    conflicts = []
    for proposal in repo.active_proposals(case_id):
        if reviews.get(proposal["id"], {}).get("decision") == "reject" and nx.has_path(graph, proposal["left_id"], proposal["right_id"]):
            conflicts.append({"proposal_id": proposal["id"], "reason": "A rejected pair is connected indirectly by accepted links. Review the conflicting decisions."})
    return graph, groups, conflicts


QUERY_TEMPLATES = {
    "pending proposals": "MATCH (p:CandidateLink {case_id:$case_id}) RETURN p ORDER BY p.created_at LIMIT $limit",
    "source inventory": "MATCH (s:VideoSource {case_id:$case_id}) RETURN s ORDER BY s.created_at LIMIT $limit",
    "review history": "MATCH (r:ReviewAction {case_id:$case_id}) RETURN r ORDER BY r.created_at DESC LIMIT $limit",
}


def constrained_query(service, case_id, text, limit=100):
    query = " ".join(text.lower().strip().rstrip("?").split())
    aliases = {"show pending proposals": "pending proposals", "show sources": "source inventory", "show review history": "review history"}
    query = aliases.get(query, query)
    if query not in QUERY_TEMPLATES:
        raise ValueError("Choose: pending proposals, source inventory, or review history. Free-form database commands are not allowed.")
    limit = max(1, min(int(limit), 200))
    repo = service.repo
    if query == "pending proposals":
        reviewed = repo.latest_reviews(case_id)
        rows = [p for p in repo.active_proposals(case_id) if reviewed.get(p["id"], {}).get("decision") not in {"accept", "reject"}]
    else:
        rows = repo.list(case_id, "source" if query == "source inventory" else "review")[::-1]
    return rows[:limit]


def sync_neo4j(service, case_id, uri, user, password, database="neo4j"):
    """Explicit local-to-Neo4j projection. SQLite remains the authoritative audit store."""
    from neo4j import GraphDatabase
    from urllib.parse import urlparse
    endpoint = urlparse(uri)
    if not endpoint.hostname or not (endpoint.scheme in {"neo4j+s", "bolt+s"} or
                                    (endpoint.scheme in {"bolt", "neo4j"} and endpoint.hostname in {"localhost", "127.0.0.1", "::1"})):
        raise ValueError("Use a local Neo4j instance or a TLS-verified server.")
    repo = service.repo
    case = repo.get(case_id, case_id, "case")
    kinds = {"source": "VideoSource", "camera": "Camera", "track": "Tracklet", "proposal": "CandidateLink",
             "review": "ReviewAction", "asset": "EvidenceAsset", "event": "AnomalyEvent", "transition": "Transition",
             "coverage_region": "CoverageRegion", "blind_portal": "BlindPortal", "blind_run": "BlindRun",
             "blind_hypothesis": "BlindHypothesis", "blind_review": "BlindReview", "blind_clue": "BlindClue",
             "source_alignment": "SourceAlignment", "chat_search": "ChatSearch"}
    records = {kind: repo.list(case_id, kind) for kind in kinds}
    _, groups, conflicts = evidence_graph(service, case_id)

    def write(tx):
        tx.run("MERGE (c:Case {id:$id}) SET c.case_id=$id,c.title=$title,c.status=$status", id=case_id, title=case["title"], status=case["status"]).consume()
        for kind, label in kinds.items():
            # Labels are constants from the allowlist, never user text.
            rows = [{"id": r["id"], "case_id": case_id, "created_at": r["created_at"], "payload": canonical(r).decode()} for r in records[kind]]
            tx.run(f"UNWIND $rows AS row MERGE (n:{label} {{id:row.id}}) SET n += row WITH n MATCH (c:Case {{id:$case_id}}) MERGE (c)-[:CONTAINS]->(n)", rows=rows, case_id=case_id).consume()
        for t in records["track"]:
            tx.run("MATCH (s:VideoSource {id:$source,case_id:$case}), (t:Tracklet {id:$id,case_id:$case}) MERGE (s)-[:PRODUCED]->(t) MERGE (v:Sighting {id:$sighting}) SET v.case_id=$case,v.start_time=$start,v.end_time=$end MERGE (t)-[:HAS_SIGHTING]->(v)",
                   source=t["source_id"], case=case_id, id=t["id"], sighting="sighting_" + t["id"], start=t["start_time"], end=t["end_time"]).consume()
            if t.get("camera_id"):
                tx.run("MATCH (s:Sighting {id:$sighting}),(c:Camera {id:$camera,case_id:$case}) MERGE (s)-[:OBSERVED_BY]->(c)", sighting="sighting_" + t["id"], camera=t["camera_id"], case=case_id).consume()
        for p in records["proposal"]:
            tx.run("MATCH (p:CandidateLink {id:$id,case_id:$case}),(a:Sighting {id:$left}),(b:Sighting {id:$right}) MERGE (p)-[:LEFT_SIGHTING]->(a) MERGE (p)-[:RIGHT_SIGHTING]->(b)",
                   id=p["id"], case=case_id, left="sighting_" + p["left_id"], right="sighting_" + p["right_id"]).consume()
        for r in records["review"]:
            tx.run("MATCH (r:ReviewAction {id:$id,case_id:$case}),(p:CandidateLink {id:$proposal,case_id:$case}) MERGE (r)-[:REVIEWS]->(p)", id=r["id"], case=case_id, proposal=r["proposal_id"]).consume()
        for edge in records["transition"]:
            tx.run("MATCH (a:Camera {id:$a,case_id:$case}),(b:Camera {id:$b,case_id:$case}) MERGE (a)-[t:TRANSITIONS_TO {id:$id}]->(b) SET t.min_s=$low,t.max_s=$high", a=edge["from_camera"], b=edge["to_camera"], case=case_id, id=edge["id"], low=edge["min_s"], high=edge["max_s"]).consume()
        for portal in records["blind_portal"]:
            tx.run("MATCH (p:BlindPortal {id:$id,case_id:$case}), (c:Camera {id:$camera,case_id:$case}), (z:CoverageRegion {id:$zone,case_id:$case}) MERGE (p)-[:OBSERVED_BY]->(c) MERGE (p)-[:LEADS_TO]->(z)",
                   id=portal["id"], camera=portal["camera_id"], zone=portal["blind_zone_id"], case=case_id).consume()
        for alignment in records["source_alignment"]:
            tx.run("MATCH (a:SourceAlignment {id:$id,case_id:$case}), (s:VideoSource {id:$source,case_id:$case}), (c:Camera {id:$camera,case_id:$case}) MERGE (a)-[:ALIGNS]->(s) MERGE (a)-[:MAPS_TO]->(c)",
                   id=alignment["id"], source=alignment["source_id"], camera=alignment["camera_id"], case=case_id).consume()
        for hypothesis in records["blind_hypothesis"]:
            tx.run("MATCH (h:BlindHypothesis {id:$id,case_id:$case}), (p:CandidateLink {id:$proposal,case_id:$case}), (a:Tracklet {id:$left,case_id:$case}), (b:Tracklet {id:$right,case_id:$case}) MERGE (h)-[:DERIVED_FROM]->(p) MERGE (h)-[:CONSIDERS]->(a) MERGE (h)-[:CONSIDERS]->(b)",
                   id=hypothesis["id"], proposal=hypothesis["proposal_id"], left=hypothesis["left_id"], right=hypothesis["right_id"], case=case_id).consume()
            for zone in hypothesis["zones"]:
                tx.run("MATCH (h:BlindHypothesis {id:$id,case_id:$case}), (z:CoverageRegion {id:$zone,case_id:$case}) MERGE (h)-[:POSSIBLE_ZONE]->(z)",
                       id=hypothesis["id"], zone=zone["zone_id"], case=case_id).consume()
        for kind, label in (("blind_review", "BlindReview"), ("blind_clue", "BlindClue")):
            for record in records[kind]:
                tx.run(f"MATCH (r:{label} {{id:$id,case_id:$case}}), (h:BlindHypothesis {{id:$hypothesis,case_id:$case}}) MERGE (r)-[:ANNOTATES]->(h)",
                       id=record["id"], hypothesis=record["hypothesis_id"], case=case_id).consume()
        for search in records["chat_search"]:
            for track_id in search["result_track_ids"]:
                tx.run("MATCH (q:ChatSearch {id:$id,case_id:$case}), (t:Tracklet {id:$track,case_id:$case}) MERGE (q)-[:RETURNED]->(t)",
                       id=search["id"], track=track_id, case=case_id).consume()
        for asset in records["asset"]:
            if asset.get("parent_id"):
                tx.run("MATCH (a:EvidenceAsset {id:$id,case_id:$case}),(b:EvidenceAsset {id:$parent,case_id:$case}) MERGE (a)-[:DERIVED_FROM]->(b)", id=asset["id"], parent=asset["parent_id"], case=case_id).consume()
        # Replace only the derived DWI projection, never immutable evidence or reviews.
        tx.run("MATCH (d:DWI {case_id:$case}) DETACH DELETE d", case=case_id).consume()
        for group in groups:
            tx.run("MERGE (d:DWI {id:$id}) SET d.case_id=$case WITH d UNWIND $tracks AS tid MATCH (s:Sighting {id:tid}) MERGE (d)-[:HAS_SIGHTING]->(s)",
                   id=group["dwi_id"], case=case_id, tracks=["sighting_" + t for t in group["track_ids"]]).consume()
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            session.execute_write(write)
    repo.audit(case_id, "neo4j-sync", "graph.synced")
    return {"records": sum(map(len, records.values())), "groups": len(groups), "conflicts": conflicts}
