"""Upload checks. Bytes stay under the owning user and can be deleted."""
from __future__ import annotations

import uuid
from pathlib import Path

LIMITS = {
    "image": 15 * 1024 * 1024,
    "audio": 25 * 1024 * 1024,
    "video": 100 * 1024 * 1024,
}
DURATION_LIMITS = {"audio": 600, "video": 30}
_MAGIC = {
    "image": (b"\x89PNG\r\n\x1a\n",),
    "audio": (b"RIFF",),
    "video": (b"\x00\x00\x00",),
}


class UploadError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class UploadStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._meta: dict[str, dict] = {}

    def accept(self, user_id: str, *, filename: str, header: bytes, declared_bytes: int,
               kind: str, duration_seconds: int | None = None) -> dict:
        if kind not in LIMITS:
            raise UploadError("UPLOAD_KIND")
        if not isinstance(declared_bytes, int) or declared_bytes < 1:
            raise UploadError("UPLOAD_SIZE")
        if declared_bytes > LIMITS[kind]:
            raise UploadError("UPLOAD_TOO_LARGE")
        if kind in DURATION_LIMITS:
            if not isinstance(duration_seconds, int) or duration_seconds < 1:
                raise UploadError("DURATION_REQUIRED")
            if duration_seconds > DURATION_LIMITS[kind]:
                raise UploadError("DURATION_LIMIT")
        if not _matches(kind, header):
            raise UploadError("UPLOAD_TYPE")
        upload_id = "upload_" + uuid.uuid4().hex
        folder = self.root / user_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / upload_id
        path.write_bytes(header)
        record = {
            "upload_id": upload_id,
            "user_id": user_id,
            "filename": Path(filename).name,
            "kind": kind,
            "declared_bytes": declared_bytes,
            "duration_seconds": duration_seconds,
        }
        self._meta[upload_id] = record
        return dict(record)

    def owned(self, user_id: str, upload_id: str) -> dict | None:
        row = self._meta.get(upload_id)
        if row is None or row["user_id"] != user_id:
            return None
        return dict(row)

    def delete(self, user_id: str, upload_id: str) -> bool:
        row = self.owned(user_id, upload_id)
        if row is None:
            return False
        path = self.root / user_id / upload_id
        if path.exists():
            path.unlink()
        self._meta.pop(upload_id, None)
        return True


def _matches(kind: str, header: bytes) -> bool:
    if kind == "video":
        return len(header) >= 8 and header[4:8] == b"ftyp"
    return any(header.startswith(magic) for magic in _MAGIC[kind])
