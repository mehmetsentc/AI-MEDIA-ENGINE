"""Storage contract. Objects live outside the metadata database."""
from __future__ import annotations

import abc


class Storage(abc.ABC):
    @abc.abstractmethod
    def put(self, *, client_id: str, job_id: str, name: str, data: bytes) -> str:
        raise NotImplementedError
