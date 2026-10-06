"""Provider-neutral GPU offers. This catalog cannot provision a machine."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional


@dataclass(frozen=True)
class GpuRequirements:
    """Generic search constraints. No model, country, or host is implied."""

    min_vram_gb: Optional[Decimal] = None
    max_hourly_price_usd: Optional[Decimal] = None
    min_reliability: Optional[Decimal] = None
    gpu_count: Optional[int] = None
    gpu_model: Optional[str] = None
    offer_id: Optional[str] = None
    allocated_storage_gb: Optional[int] = None
    sort_by_hourly: bool = False


@dataclass(frozen=True)
class GpuOffer:
    provider: str
    offer_id: str
    gpu_model: str
    gpu_count: int
    vram_gb: Decimal
    hourly_price_usd: Decimal
    reliability: Optional[Decimal] = None
    performance: Optional[Decimal] = None
    cuda: Optional[str] = None
    disk_gb: Optional[Decimal] = None
    location: Optional[str] = None
    availability: Optional[str] = None
    gpu_hourly_price_usd: Optional[Decimal] = None
    storage_price_per_gb_month: Optional[Decimal] = None
    machine_id: Optional[str] = None

    def as_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "offer_id": self.offer_id,
            "gpu_model": self.gpu_model,
            "gpu_count": self.gpu_count,
            "vram_gb": str(self.vram_gb),
            "hourly_price_usd": str(self.hourly_price_usd),
            "reliability": None if self.reliability is None else str(self.reliability),
            "performance": None if self.performance is None else str(self.performance),
            "cuda": self.cuda,
            "disk_gb": None if self.disk_gb is None else str(self.disk_gb),
            "location": self.location,
            "availability": self.availability,
            "machine_id": self.machine_id,
        }


def model_key(value: str) -> str:
    """Compare GPU names without treating punctuation as a different model."""
    return " ".join(value.replace("_", " ").casefold().split())
