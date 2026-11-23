import json
from urllib.parse import parse_qs, urlparse, urlencode
from urllib.request import urlopen

import pytest

from fedsight.siwc import CredentialStore, finish_sign_in, start_sign_in


def test_credentials_encrypted_and_host_stable(tmp_path):
    store = CredentialStore(tmp_path, b"k" * 32)
    first = store.host_id()
    assert first == store.host_id()
    credential = {"client_id": "oaiapp_test", "access_token": "secret-access-token", "scopes": ["chatgpt.tokens.use.direct"]}
    store.save(credential)
    assert store.load() == credential
    assert b"secret-access-token" not in store.path.read_bytes()


def test_loopback_sign_in_validates_state_id_token_and_scope(tmp_path, monkeypatch):
    import fedsight.siwc as siwc
    store = CredentialStore(tmp_path, b"k" * 32)
    attempt = start_sign_in(store)
    params = parse_qs(urlparse(attempt.authorization_url).query)
    assert params["client_id"] == ["dynamic_agent_client"]
    assert params["redirect_uri"] == [attempt.redirect_uri]
    assert params["code_challenge_method"] == ["S256"]
    monkeypatch.setattr(siwc, "_post_form", lambda url, values: {
        "id_token": "signed-id-token", "access_token": "access", "refresh_token": "refresh",
        "scope": siwc.SCOPES, "expires_in": 3600})
    monkeypatch.setattr(siwc, "_validate_id_token", lambda token, client_id, nonce: {
        "sub": "account-1", "email": "user@example.com", "nonce": nonce})
    callback = attempt.redirect_uri + "?" + urlencode({"state": attempt.state, "code": "one-time-code", "client_id": "oaiapp_test"})
    with urlopen(callback, timeout=5) as response:
        assert response.status == 200
    result = finish_sign_in(attempt, store)
    assert result["client_id"] == "oaiapp_test"
    assert store.load()["access_token"] == "access"
    assert store.valid_access_token() == "access"


def test_loopback_rejects_wrong_state(tmp_path):
    store = CredentialStore(tmp_path, b"k" * 32)
    attempt = start_sign_in(store)
    callback = attempt.redirect_uri + "?" + urlencode({"state": "wrong", "code": "one-time-code", "client_id": "oaiapp_test"})
    with urlopen(callback, timeout=5):
        pass
    with pytest.raises(ValueError, match="state mismatch"):
        finish_sign_in(attempt, store)
    assert store.load() is None


def test_refresh_replaces_rotating_token(tmp_path, monkeypatch):
    import fedsight.siwc as siwc
    store = CredentialStore(tmp_path, b"k" * 32)
    store.save({"client_id": "oaiapp_test", "access_token": "old", "refresh_token": "old-refresh",
                "expires_at": 0, "scopes": ["chatgpt.tokens.use.direct"]})
    monkeypatch.setattr(siwc, "_post_form", lambda url, values: {
        "access_token": "new", "refresh_token": "new-refresh", "expires_in": 3600})
    assert store.valid_access_token() == "new"
    assert store.load()["refresh_token"] == "new-refresh"
