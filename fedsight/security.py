"""Authenticated evidence storage and keyed, checkpointable audit chains."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .core import canonical, digest


class IntegrityError(ValueError):
    pass


class Keyring:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.encryption = self._load("evidence.key")
        self.audit = self._load("audit.key")
        self.key_id = digest(self.encryption)[:16]

    def _load(self, name: str) -> bytes:
        path = self.directory / name
        try:
            with path.open("xb") as stream:
                stream.write(os.urandom(32))
            os.chmod(path, 0o600)
        except FileExistsError:
            pass
        key = path.read_bytes()
        if len(key) != 32:
            raise IntegrityError("Key file must contain exactly 32 bytes.")
        return key


def encrypt(data: bytes, key: bytes, context: dict) -> bytes:
    if len(key) != 32:
        raise ValueError("AES-256 requires a 32-byte key.")
    nonce = os.urandom(12)
    return b"FS2\x01" + nonce + AESGCM(key).encrypt(nonce, data, canonical(context))


def decrypt(blob: bytes, key: bytes, context: dict) -> bytes:
    if len(blob) < 32 or blob[:4] != b"FS2\x01":
        raise IntegrityError("Invalid encrypted evidence envelope.")
    try:
        return AESGCM(key).decrypt(blob[4:16], blob[16:], canonical(context))
    except Exception as exc:
        raise IntegrityError("Evidence authentication failed: wrong key, context, or altered ciphertext.") from exc


def audit_mac(key: bytes, event: dict) -> str:
    return hmac.new(key, canonical(event), hashlib.sha3_256).hexdigest()


def verify_events(events: list[dict], key: bytes, checkpoint: dict | None = None) -> dict:
    previous = "0" * 64
    for expected, row in enumerate(events, 1):
        event = {k: row[k] for k in ("seq", "at", "actor", "action", "case_id", "record_id", "evidence_hash", "previous")}
        if event["seq"] != expected or event["previous"] != previous or not hmac.compare_digest(row["mac"], audit_mac(key, event)):
            return {"ok": False, "first_broken_sequence": expected, "reason": "Sequence, chain, or HMAC mismatch."}
        previous = row["mac"]
    if checkpoint:
        signed = {k: checkpoint[k] for k in ("seq", "mac")}
        if not hmac.compare_digest(checkpoint.get("signature", ""), audit_mac(key, signed)):
            return {"ok": False, "reason": "Checkpoint signature mismatch."}
        seq = checkpoint["seq"]
        if seq > len(events) or seq < 0 or (seq and events[seq - 1]["mac"] != checkpoint["mac"]):
            return {"ok": False, "reason": "Audit history differs from the external checkpoint."}
    return {"ok": True, "events": len(events), "head": previous,
            "checkpoint_checked": checkpoint is not None,
            "limitation": "Tail truncation requires a previously exported checkpoint; protect keys and checkpoint separately."}


class EvidenceVault:
    def __init__(self, root: Path, keyring: Keyring):
        self.root, self.keys = Path(root), keyring
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, data: bytes, case_id: str, asset_id: str) -> dict:
        context = {"case_id": case_id, "asset_id": asset_id, "sha256": digest(data)}
        name = digest(canonical(context)) + ".aes"
        target = self.root / name
        blob = encrypt(data, self.keys.encryption, context)
        try:
            with target.open("xb") as stream:
                stream.write(blob)
            os.chmod(target, 0o400)
        except FileExistsError:
            if self.read({**context, "storage_name": name}) != data:
                raise IntegrityError("An existing evidence asset differs.")
        return {**context, "storage_name": name, "bytes": len(data), "key_id": self.keys.key_id,
                "encryption": "AES-256-GCM"}

    def read(self, asset: dict) -> bytes:
        name = asset["storage_name"]
        if Path(name).name != name or not name.endswith(".aes"):
            raise IntegrityError("Invalid asset reference.")
        context = {k: asset[k] for k in ("case_id", "asset_id", "sha256")}
        data = decrypt((self.root / name).read_bytes(), self.keys.encryption, context)
        if not hmac.compare_digest(digest(data), asset["sha256"]):
            raise IntegrityError("Original content hash mismatch.")
        return data

    def remove(self, case_id: str, asset_id: str, sha256: str) -> None:
        context = {"case_id": case_id, "asset_id": asset_id, "sha256": sha256}
        path = self.root / (digest(canonical(context)) + ".aes")
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except PermissionError:
            # Vault originals are marked read-only on Windows.
            if path.is_symlink():
                raise IntegrityError("Unexpected link in the evidence vault.")
            path.chmod(0o600)
            path.unlink(missing_ok=True)
