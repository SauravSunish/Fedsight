"""Optional loopback API. Start with `uvicorn fedsight.api:create_app --factory`.

Requires an explicit bearer token. This is a single-trust-domain integration API,
not a multi-user authentication/authorisation server.
"""
from __future__ import annotations
import hmac
import os
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from .service import FedSight


class CaseInput(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=10000)
    consent: str = Field(default="", max_length=5000)


class ReviewInput(BaseModel):
    decision: Literal["accept", "reject", "defer"]
    reason: str = Field(min_length=1, max_length=5000)


def create_app(service=None, token=None):
    token = token or os.environ.get("FEDSIGHT_API_TOKEN")
    if not token or len(token) < 32:
        raise ValueError("Set FEDSIGHT_API_TOKEN to a random secret of at least 32 characters.")
    service = service or FedSight(Path(os.environ.get("FEDSIGHT_DATA_DIR", "data/workbench")))
    actor = os.environ.get("FEDSIGHT_API_ACTOR", "local-api")
    app = FastAPI(title="FedSight local integration API", version="2.0.0")
    bearer = HTTPBearer(auto_error=False)

    def authorize(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if credentials is None or credentials.scheme.lower() != "bearer" or not hmac.compare_digest(credentials.credentials, token):
            raise HTTPException(401, "Invalid API token", headers={"WWW-Authenticate": "Bearer"})

    def scoped_case(cid):
        try:
            return service.repo.get(cid, cid, "case")
        except ValueError:
            raise HTTPException(404, "Case not found")

    @app.get("/health")
    def health():
        return {"status": "ok", "version": "2.0.0"}

    @app.get("/cases", dependencies=[Depends(authorize)])
    def cases():
        return service.repo.list(None, "case")

    @app.post("/cases", dependencies=[Depends(authorize)], status_code=201)
    def create_case(value: CaseInput):
        try:
            return service.repo.create_case(value.title, value.description, actor, consent=value.consent)
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.get("/cases/{cid}/sources", dependencies=[Depends(authorize)])
    def sources(cid: str):
        scoped_case(cid)
        return service.repo.list(cid, "source")

    @app.get("/cases/{cid}/proposals", dependencies=[Depends(authorize)])
    def proposals(cid: str):
        scoped_case(cid)
        return service.repo.active_proposals(cid)

    @app.post("/cases/{cid}/proposals/{pid}/reviews", dependencies=[Depends(authorize)], status_code=201)
    def review(cid: str, pid: str, value: ReviewInput):
        scoped_case(cid)
        try:
            return service.repo.review(cid, pid, actor, value.decision, value.reason)
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.get("/integrity", dependencies=[Depends(authorize)])
    def integrity():
        return service.repo.verify()

    return app
