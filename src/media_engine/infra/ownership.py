"""Job ownership stays on the existing repository. This module does not open Neon."""
from __future__ import annotations

CONSUMED_PREFIX = "live8-"


class SharedOwnership:
    """Claim and settlement go through SqliteRepository or PostgresRepository."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def claim(self, worker_id: str, now: str, until: str) -> str | None:
        claimed = self.repository.claim(worker_id, now, until)
        return None if claimed is None else str(claimed)

    def authorizes_rental(self) -> bool:
        return False

    def note_authorization(self, execution_id: str) -> bool:
        """Remember an id locally. A consumed LIVE-8 id cannot be reused, and none rents."""
        token = str(execution_id or "")
        if not token or token.startswith(CONSUMED_PREFIX):
            return False
        key = "gpu_authorization:" + token
        if self.repository.config(key) == "consumed":
            return False
        self.repository.set_config(key, "consumed")
        return False
