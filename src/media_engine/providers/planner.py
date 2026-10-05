"""Small deterministic planner. Performance is a marketplace heuristic, not inference speed."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from media_engine.providers.offers import GpuOffer, GpuRequirements, model_key


@dataclass(frozen=True)
class ResourceRequirements:
    """What a workload needs. Providers are not named here."""

    min_vram_gb: Decimal
    gpu_count: int
    max_hourly_price_usd: Decimal
    min_reliability: Decimal
    min_performance: Optional[Decimal] = None
    compatible_gpu_models: tuple[str, ...] = ()
    excluded_gpu_models: tuple[str, ...] = ()
    max_job_seconds: Optional[int] = None

    def to_search(self, gpu_model: Optional[str] = None) -> GpuRequirements:
        return GpuRequirements(
            min_vram_gb=self.min_vram_gb,
            max_hourly_price_usd=self.max_hourly_price_usd,
            min_reliability=self.min_reliability,
            gpu_count=self.gpu_count,
            gpu_model=gpu_model,
        )


@dataclass(frozen=True)
class Plan:
    eligible: tuple[GpuOffer, ...]
    shortlist: tuple[GpuOffer, ...]
    selected: Optional[GpuOffer]
    explanation: str
    max_estimated_cost_usd: Optional[Decimal]
    max_job_seconds: Optional[int]
    status: str


def max_estimated_gpu_cost(hourly_price_usd: Decimal, max_job_seconds: int) -> Decimal:
    if max_job_seconds <= 0:
        raise ValueError("max_job_seconds must be positive")
    return hourly_price_usd * Decimal(max_job_seconds) / Decimal(3600)


def plan(offers: list[GpuOffer], requirements: ResourceRequirements, *, shortlist_size: int = 5) -> Plan:
    eligible = tuple(offer for offer in offers if _eligible(offer, requirements))
    ranked = tuple(sorted(eligible, key=_rank_key))
    shortlist = ranked[:shortlist_size]
    selected = shortlist[0] if shortlist else None
    cost = None
    if selected is not None and requirements.max_job_seconds is not None:
        cost = max_estimated_gpu_cost(selected.hourly_price_usd, requirements.max_job_seconds)
    if selected is None:
        return Plan(
            eligible=(),
            shortlist=(),
            selected=None,
            explanation="rejected: " + (_rejection(offers, requirements) or "no offer"),
            max_estimated_cost_usd=None,
            max_job_seconds=requirements.max_job_seconds,
            status="NO_ELIGIBLE_OFFER",
        )
    return Plan(
        eligible=ranked,
        shortlist=shortlist,
        selected=selected,
        explanation=(
            f"selected {selected.offer_id} by marketplace performance-per-dollar heuristic; "
            "not measured inference speed"
        ),
        max_estimated_cost_usd=cost,
        max_job_seconds=requirements.max_job_seconds,
        status="HUMAN_APPROVAL_REQUIRED",
    )


def format_preflight(result: Plan) -> str:
    lines = ["SHORTLIST:"]
    if not result.shortlist:
        lines.append("(none)")
    for index, offer in enumerate(result.shortlist, start=1):
        lines.append(f"{index}. {offer.offer_id} {offer.gpu_model} {offer.hourly_price_usd}/h")
    if result.selected is None:
        lines.append(f"STATUS: {result.status}")
        lines.append(result.explanation)
        return "\n".join(lines) + "\n"
    offer = result.selected
    cost = "unknown" if result.max_estimated_cost_usd is None else f"${result.max_estimated_cost_usd}"
    lines.extend([
        f"PROVIDER: {offer.provider}",
        f"GPU: {offer.gpu_model}",
        f"VRAM: {offer.vram_gb} GB",
        f"OFFER ID: {offer.offer_id}",
        f"PRICE/HOUR: {offer.hourly_price_usd}",
        f"RELIABILITY: {offer.reliability}",
        f"PERFORMANCE HEURISTIC: {offer.performance}",
        f"MAX JOB SECONDS: {result.max_job_seconds}",
        f"MAX ESTIMATED GPU COST: {cost}",
        f"STATUS: {result.status}",
        result.explanation,
    ])
    return "\n".join(lines) + "\n"


def _eligible(offer: GpuOffer, requirements: ResourceRequirements) -> bool:
    return _reason(offer, requirements) is None


def _reason(offer: GpuOffer, requirements: ResourceRequirements) -> Optional[str]:
    if offer.gpu_count != requirements.gpu_count:
        return "gpu_count"
    if offer.vram_gb < requirements.min_vram_gb:
        return "vram"
    if offer.hourly_price_usd > requirements.max_hourly_price_usd:
        return "price"
    if offer.reliability is None or offer.reliability < requirements.min_reliability:
        return "reliability"
    if requirements.min_performance is not None:
        if offer.performance is None or offer.performance < requirements.min_performance:
            return "performance"
    if requirements.compatible_gpu_models:
        allowed = {model_key(name) for name in requirements.compatible_gpu_models}
        if model_key(offer.gpu_model) not in allowed:
            return "incompatible"
    if any(model_key(offer.gpu_model) == model_key(name) for name in requirements.excluded_gpu_models):
        return "incompatible"
    return None


def _rejection(offers: list[GpuOffer], requirements: ResourceRequirements) -> str:
    reasons = []
    for offer in offers:
        reason = _reason(offer, requirements)
        if reason and reason not in reasons:
            reasons.append(reason)
    return ",".join(reasons)


def _rank_key(offer: GpuOffer) -> tuple:
    if offer.performance is None or offer.performance <= 0 or offer.hourly_price_usd <= 0:
        ratio = Decimal("-1")
    else:
        ratio = offer.performance / offer.hourly_price_usd
    return (-ratio, offer.hourly_price_usd, offer.offer_id)
