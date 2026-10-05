"""Local filesystem storage. The URI is a file URI."""
from __future__ import annotations

import re
from pathlib import Path

_SEGMENT = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,80}$")
_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,80}$")


from media_engine.storage.base import Storage


class LocalStorage(Storage):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def put(self, *, client_id: str, job_id: str, name: str, data: bytes) -> str:
        if _SEGMENT.fullmatch(client_id) is None or _SEGMENT.fullmatch(job_id) is None:
            raise ValueError("storage key is not acceptable")
        if _NAME.fullmatch(name) is None:
            raise ValueError("artifact name is not acceptable")
        path = self.root / client_id / job_id / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path.resolve().as_uri()
