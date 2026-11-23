"""Optional Sign in with ChatGPT for a local, single-user workbench.

Uses OpenAI's public loopback OAuth/PKCE flow. Tokens are encrypted with the
workbench key and never stored in case records or browser session state.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import queue
import secrets
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

from .core import canonical
from .security import encrypt, decrypt


AUTH = "https://auth.openai.com/api/accounts/authorize"
TOKEN = "https://auth.openai.com/api/accounts/oauth/token"
JWKS = "https://auth.openai.com/.well-known/jwks.json"
DISCOVERY = "https://auth.openai.com/.well-known/openid-configuration"
RESOURCE = "https://api.openai.com/v1"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
CONTEXT = {"purpose": "fedsight_siwc_credentials_v1"}


def _post_form(url: str, values: dict) -> dict:
    request = Request(url, data=urlencode(values).encode(),
                      headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
                      method="POST")
    with urlopen(request, timeout=20) as response:
        return json.load(response)


def _validate_id_token(token: str, client_id: str, nonce: str) -> dict:
    try:
        import jwt
    except ImportError as exc:
        raise RuntimeError("Install requirements-ai.txt for ChatGPT sign-in.") from exc
    key = jwt.PyJWKClient(JWKS, timeout=10).get_signing_key_from_jwt(token).key
    claims = jwt.decode(token, key, algorithms=["RS256", "ES256"], audience=client_id,
                        issuer="https://auth.openai.com", leeway=5,
                        options={"require": ["sub", "exp", "iat"]})
    if claims.get("nonce") != nonce or not isinstance(claims.get("sub"), str) or not claims["sub"]:
        raise ValueError("ChatGPT identity token did not match the sign-in attempt.")
    return claims


class CredentialStore:
    def __init__(self, directory: Path, encryption_key: bytes):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "siwc-credentials.aes"
        self.host_path = self.directory / "siwc-host-id.txt"
        self.key = encryption_key

    def host_id(self) -> str:
        try:
            with self.host_path.open("x", encoding="utf-8") as stream:
                stream.write("urn:uuid:" + str(uuid4()))
            os.chmod(self.host_path, 0o600)
        except FileExistsError:
            pass
        value = self.host_path.read_text(encoding="utf-8").strip()
        if not value.startswith("urn:uuid:"):
            raise ValueError("Invalid saved ChatGPT host identifier.")
        return value

    def load(self) -> dict | None:
        if not self.path.exists():
            return None
        return json.loads(decrypt(self.path.read_bytes(), self.key, CONTEXT))

    def save(self, credential: dict) -> None:
        payload = encrypt(canonical(credential), self.key, CONTEXT)
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            stream.write(payload)
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.path)

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)

    def valid_access_token(self) -> str:
        credential = self.load()
        if not credential or "chatgpt.tokens.use.direct" not in credential.get("scopes", []):
            raise ValueError("Connect an eligible ChatGPT account with plan usage permission first.")
        if credential.get("expires_at", 0) <= time.time() + 60:
            if not credential.get("refresh_token"):
                raise ValueError("ChatGPT sign-in expired. Connect again.")
            refreshed = _post_form(TOKEN, {"grant_type": "refresh_token", "client_id": credential["client_id"],
                                           "refresh_token": credential["refresh_token"], "resource": RESOURCE})
            credential.update(access_token=refreshed["access_token"],
                              refresh_token=refreshed.get("refresh_token", credential["refresh_token"]),
                              expires_at=time.time() + int(refreshed.get("expires_in", 3600)))
            if "scope" in refreshed:
                credential["scopes"] = refreshed["scope"].split()
            self.save(credential)
        return credential["access_token"]

    def disconnect(self) -> None:
        credential = self.load()
        if credential and credential.get("refresh_token"):
            try:
                with urlopen(DISCOVERY, timeout=10) as response:
                    endpoint = json.load(response).get("revocation_endpoint", "")
                if endpoint.startswith("https://auth.openai.com/"):
                    request = Request(endpoint, data=urlencode({"token": credential["refresh_token"],
                                          "token_type_hint": "refresh_token", "client_id": credential["client_id"]}).encode(),
                                      headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST")
                    with urlopen(request, timeout=10):
                        pass
            finally:
                self.clear()
        else:
            self.clear()


@dataclass
class SignInAttempt:
    server: HTTPServer
    thread: threading.Thread
    received: queue.Queue
    state: str
    nonce: str
    verifier: str
    redirect_uri: str
    client_id: str
    started_at: float
    authorization_url: str

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def start_sign_in(store: CredentialStore) -> SignInAttempt:
    credential = store.load()
    client_id = credential["client_id"] if credential else "dynamic_agent_client"
    state, nonce, verifier = secrets.token_urlsafe(32), secrets.token_urlsafe(32), secrets.token_urlsafe(64)
    received: queue.Queue = queue.Queue(maxsize=1)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path != "/auth/callback":
                self.send_error(404)
                return
            try:
                received.put_nowait({k: v[0] for k, v in parse_qs(parsed.query).items()})
            except queue.Full:
                self.send_error(409)
                return
            body = b"<html><body><h2>FedSight sign-in received</h2><p>Return to the FedSight tab to finish connecting.</p></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass  # Never log the authorization code or callback query.

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="fedsight-siwc-callback")
    thread.start()
    redirect_uri = f"http://127.0.0.1:{server.server_port}/auth/callback"
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    params = {"client_id": client_id, "response_type": "code", "redirect_uri": redirect_uri,
              "scope": SCOPES, "resource": RESOURCE, "state": state, "nonce": nonce,
              "code_challenge_method": "S256", "code_challenge": challenge,
              "ext_agent_host_id": store.host_id()}
    if not credential:
        params["agent_name_hint"] = "FedSight"
    return SignInAttempt(server, thread, received, state, nonce, verifier, redirect_uri,
                         client_id, time.time(), AUTH + "?" + urlencode(params))


def finish_sign_in(attempt: SignInAttempt, store: CredentialStore) -> dict | None:
    if time.time() - attempt.started_at > 300:
        attempt.close()
        raise ValueError("ChatGPT sign-in timed out. Start again.")
    try:
        callback = attempt.received.get_nowait()
    except queue.Empty:
        return None
    attempt.close()
    if not secrets.compare_digest(callback.get("state", ""), attempt.state):
        raise ValueError("ChatGPT sign-in state mismatch.")
    if callback.get("error"):
        raise ValueError("ChatGPT authorization was not granted.")
    issued_id = callback.get("client_id", "")
    if attempt.client_id == "dynamic_agent_client":
        if not issued_id.startswith("oaiapp_"):
            raise ValueError("ChatGPT did not return an issued client ID.")
        client_id = issued_id
    else:
        if issued_id and issued_id != attempt.client_id:
            raise ValueError("ChatGPT returned a different client registration.")
        client_id = attempt.client_id
    if not callback.get("code"):
        raise ValueError("ChatGPT did not return an authorization code.")
    tokens = _post_form(TOKEN, {"grant_type": "authorization_code", "code": callback["code"],
                                "redirect_uri": attempt.redirect_uri, "client_id": client_id,
                                "code_verifier": attempt.verifier, "resource": RESOURCE})
    claims = _validate_id_token(tokens["id_token"], client_id, attempt.nonce)
    prior = store.load()
    if prior and prior["client_id"] == client_id and prior["subject"] != claims["sub"]:
        raise ValueError("The returned ChatGPT account differs from the saved registration.")
    scopes = tokens.get("scope", "").split()
    if "chatgpt.tokens.use.direct" not in scopes:
        raise ValueError("ChatGPT plan usage was not authorized for FedSight.")
    credential = {"client_id": client_id, "subject": claims["sub"], "email": claims.get("email"),
                  "id_token": tokens["id_token"], "access_token": tokens["access_token"],
                  "refresh_token": tokens.get("refresh_token"), "expires_at": time.time() + int(tokens.get("expires_in", 3600)),
                  "scopes": scopes, "host_id": store.host_id()}
    store.save(credential)
    return {"email": credential["email"], "client_id": client_id, "scopes": scopes}


def available_models(store: CredentialStore) -> list[str]:
    token = store.valid_access_token()
    request = Request(RESOURCE + "/models", headers={"Authorization": "Bearer " + token,
                                                       "Accept": "application/json"})
    with urlopen(request, timeout=15) as response:
        payload = json.load(response)
    rows = payload.get("models", payload.get("data", []))
    return sorted({row["id"] for row in rows if isinstance(row, dict) and isinstance(row.get("id"), str)})
