import os
import pytest
from fedsight.security import encrypt, decrypt, IntegrityError, Keyring
from fedsight.repository import Repository


@pytest.fixture
def repo(tmp_path):
    return Repository(tmp_path / "cases.db", Keyring(tmp_path / "keys"))


def test_authenticated_encryption_and_four_failure_modes():
    key, context = os.urandom(32), {"case_id": "a", "asset_id": "x", "sequence": 1}
    blob = encrypt(b"original evidence", key, context)
    assert decrypt(blob, key, context) == b"original evidence"
    assert encrypt(b"original evidence", key, context) != blob
    for altered, wrong_key, wrong_context in [(blob[:-1] + bytes([blob[-1] ^ 1]), key, context),
                                              (blob[:-4], key, context), (blob, os.urandom(32), context),
                                              (blob, key, {**context, "case_id": "b"})]:
        with pytest.raises(IntegrityError):
            decrypt(altered, wrong_key, wrong_context)


@pytest.mark.parametrize("attack", ["edit", "delete", "reorder", "tail"])
def test_audit_tampering_detected(repo, attack):
    case = repo.create_case("test", "", "reviewer")
    repo.audit(case["id"], "reviewer", "test.one")
    repo.audit(case["id"], "reviewer", "test.two")
    checkpoint = repo.checkpoint()
    with repo.connect() as conn:
        if attack == "edit":
            conn.execute("UPDATE audit SET actor='attacker' WHERE seq=2")
        elif attack == "delete":
            conn.execute("DELETE FROM audit WHERE seq=2")
        elif attack == "reorder":
            conn.execute("UPDATE audit SET seq=20 WHERE seq=2")
        else:
            conn.execute("DELETE FROM audit WHERE seq=3")
    assert not repo.verify(checkpoint)["ok"]


def test_replay_persists_and_rejections_are_audited(repo):
    cid = repo.create_case("replay", "", "reviewer")["id"]
    repo.accept_message("node", cid, 4, "message", "reviewer")
    reopened = Repository(repo.path, repo.keys)
    for seq, msg in [(4, "message"), (3, "other"), (5, "message")]:
        with pytest.raises(IntegrityError):
            reopened.accept_message("node", cid, seq, msg, "reviewer")
    assert sum(e["action"] == "replay.rejected" for e in repo.audit_events()) == 3
    assert repo.verify()["ok"]


def test_deleted_record_inventory_detected(repo):
    case = repo.create_case("case", "", "reviewer")
    with repo.connect() as conn:
        conn.execute("DELETE FROM records WHERE id=?", (case["id"],))
    assert not repo.verify()["ok"]


def test_ciphertext_not_plaintext_and_case_scope(repo):
    case = repo.create_case("private title", "secret description", "reviewer")
    with repo.connect() as conn:
        payload = conn.execute("SELECT payload FROM records").fetchone()[0]
    assert b"secret description" not in payload
    with pytest.raises(ValueError):
        repo.get("another-case", case["id"])
