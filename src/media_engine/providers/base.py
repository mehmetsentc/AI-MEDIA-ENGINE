"""Vendor-neutral GPU lifecycle."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional


class ProviderError(Exception):
    code = "PROVIDER_ERROR"


class ProviderCapacityError(ProviderError):
    code = "MAX_GPU_WORKERS"


@dataclass(frozen=True)
class Quote:
    hourly_price_usd: Decimal
    provider: str


@dataclass(frozen=True)
class CreateResult:
    outcome: str
    resource_id: Optional[str] = None


OBSERVE_PRESENT = "present"
OBSERVE_GONE = "gone"
OBSERVE_FOREIGN = "foreign"
OBSERVE_UNKNOWN = "unknown"


class GPUProvider(abc.ABC):
    """quote, create, status, terminate, list_owned.

    create performs one attempt. It does not retry. An ambiguous outcome
    means the caller must treat the resource as unresolved.
    """

    @abc.abstractmethod
    def quote(self, owner: str) -> Quote:
        raise NotImplementedError

    @abc.abstractmethod
    def create(self, owner: str) -> CreateResult:
        raise NotImplementedError

    @abc.abstractmethod
    def status(self, resource_id: str) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def terminate(self, resource_id: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def list_owned(self) -> list[dict]:
        raise NotImplementedError

    @abc.abstractmethod
    def observe(self, resource_id: str, owner: str) -> str:
        """Return present, gone, foreign, or unknown.

        present requires both the recorded id and the owner to match.
        unknown means the provider could not be asked. Never guess.
        """

    def begin_busy(self, resource_id: str) -> None:
        """Local lifecycle hook. Remote providers can no-op until a job starts."""

    def end_busy(self, resource_id: str) -> None:
        """Return a finished job's worker to idle."""
