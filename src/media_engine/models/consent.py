"""Voice-owner consent. Records stay on the local studio database."""
from __future__ import annotations

import hashlib
import sqlite3
import uuid
from datetime import datetime, timezone


class ConsentError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class ConsentStore:
    def __init__(self, path: str) -> None:
        self.path = path
        conn = sqlite3.connect(path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS voice_consents (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    speaker_name TEXT NOT NULL,
                    statement_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def record(self, user_id: str, payload: dict) -> dict:
        if payload.get("subject_is_owner") is not True:
            raise ConsentError("CONSENT_REQUIRED")
        if payload.get("third_party_impersonation") is True:
            raise ConsentError("IMPERSONATION_REFUSED")
        statement = payload.get("statement")
        speaker = payload.get("speaker_name")
        if not isinstance(statement, str) or len(statement.strip()) < 20:
            raise ConsentError("CONSENT_REQUIRED")
        if not isinstance(speaker, str) or not speaker.strip():
            raise ConsentError("SPEAKER_REQUIRED")
        consent_id = "consent_" + uuid.uuid4().hex
        digest = hashlib.sha256(statement.strip().encode("utf-8")).hexdigest()
        created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        conn = sqlite3.connect(self.path)
        try:
            conn.execute(
                """
                INSERT INTO voice_consents (id, user_id, speaker_name, statement_sha256, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (consent_id, user_id, speaker.strip(), digest, created),
            )
            conn.commit()
        finally:
            conn.close()
        return {
            "consent_id": consent_id,
            "user_id": user_id,
            "speaker_name": speaker.strip(),
            "statement_sha256": digest,
            "created_at": created,
        }

    def get(self, user_id: str, consent_id: str) -> dict | None:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                """
                SELECT id, user_id, speaker_name, statement_sha256, created_at
                FROM voice_consents WHERE id = ? AND user_id = ?
                """,
                (consent_id, user_id),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return {
            "consent_id": row["id"],
            "user_id": row["user_id"],
            "speaker_name": row["speaker_name"],
            "statement_sha256": row["statement_sha256"],
            "created_at": row["created_at"],
        }
