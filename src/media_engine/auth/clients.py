"""Known API clients. Phase 1 seeds film-studio only."""
from __future__ import annotations

import re

_CLIENT_RE = re.compile(r"^[a-z][a-z0-9-]{1,63}$")
SEEDED_CLIENTS = frozenset({"film-studio"})


class UnknownClient(Exception):
    code = "UNKNOWN_CLIENT"

    def __init__(self, client_id: str) -> None:
        super().__init__(client_id)
        self.client_id = client_id


class ClientDirectory:
    """In-memory allow-list. `allow` exists so a later client can be added
    without an account system. Phase 1 does not seed any client except
    film-studio."""

    def __init__(self, clients: frozenset[str] | set[str] | None = None) -> None:
        self._clients = set(SEEDED_CLIENTS if clients is None else clients)

    def require(self, client_id: str) -> str:
        if not isinstance(client_id, str) or client_id not in self._clients:
            raise UnknownClient(str(client_id))
        return client_id

    def allow(self, client_id: str) -> None:
        if not isinstance(client_id, str) or _CLIENT_RE.fullmatch(client_id) is None:
            raise UnknownClient(str(client_id))
        self._clients.add(client_id)

    def known(self) -> frozenset[str]:
        return frozenset(self._clients)
