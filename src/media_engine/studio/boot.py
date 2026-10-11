"""Studio process setup. Development stays on the fake provider."""
from __future__ import annotations

import os
import time
from decimal import Decimal
from pathlib import Path
from typing import Optional

from media_engine.engines.image.base import ImageEngine, ImageEngineError
from media_engine.orchestrator.controller import MediaController, WallClock
from media_engine.platform.service import Platform
from media_engine.providers.fake import FakeGPUProvider
from media_engine.safety.limits import SafetyLimits, limits_from_env
from media_engine.studio.dev_engine import DevImageEngine


class RealProviderRequired(ImageEngine):
    """Refuses to paint a preview image when the studio is in real mode."""

    def render(self, prompt: str, width: int = 1024, height: int = 1024, seed: int = 0) -> bytes:
        raise ImageEngineError("REAL_PROVIDER_REQUIRED")


def resolve_studio_mode(environ: Optional[dict] = None) -> str:
    """development uses the fake provider. real and production use Qwen."""
    source = os.environ if environ is None else environ
    raw = str(source.get("STUDIO_MODE", "development")).strip().casefold()
    if raw in {"real", "production"}:
        return "real"
    if raw in {"", "development", "dev"}:
        return "development"
    raise ValueError("STUDIO_MODE")


def build_controller(root: Path, mode: Optional[str] = None,
                     environ: Optional[dict] = None) -> MediaController:
    """Build the local studio controller. Real mode does not use the preview engine."""
    selected = resolve_studio_mode(environ) if mode is None else mode
    root.mkdir(parents=True, exist_ok=True)
    clock = WallClock()
    if selected == "development":
        return MediaController(
            str(root / "studio.sqlite3"),
            str(root / "artifacts"),
            clock=clock,
            provider=FakeGPUProvider(time.time),
            image_engine=DevImageEngine(),
        )
    if selected != "real":
        raise ValueError("STUDIO_MODE")
    source = os.environ if environ is None else environ
    profile = str(source.get("MEDIA_ENGINE_EXECUTION_PROFILE", "")).strip()
    paid_postgres = (
        str(source.get("STUDIO_MODE", "")).strip().casefold() == "production"
        or profile in {"live7", "live8"}
    )
    stores = None
    if paid_postgres:
        from media_engine.platform.postgres_app import open_production_stores
        stores = open_production_stores(source, limits=limits_from_env(SafetyLimits(
            live_external_providers=True, max_gpu_workers=1, max_job_attempts=1,
        )))
    key = str(source.get("VAST_API_KEY", "")).strip()
    if not key:
        raise ValueError("VAST_API_KEY")
    _load_r2_if_needed(source)
    from media_engine.providers.vast_image import VastImageProvider
    limits = limits_from_env(SafetyLimits(
        live_external_providers=True, max_gpu_workers=1, max_job_attempts=1,
    ))
    provider = VastImageProvider(api_key=key, limits=limits, now=clock.now, runtime=root)
    if profile == "live7":
        from media_engine.providers.live7_prepare import claim_execution_id, grant_from_environ
        grant = grant_from_environ(source)
        claim_execution_id(root / "live7-executions", grant.execution_id)
        provider.bind_execution(grant, execution_id=grant.execution_id)
    elif profile == "live8":
        from media_engine.providers.live8_prepare import bind_from_environ, head_sha
        bind_from_environ(provider, source, runtime=root, code_sha=head_sha())
    elif profile in {"", "live6"}:
        from media_engine.providers.live6_prepare import claim_execution_id, grant_from_environ
        grant = grant_from_environ(source)
        if grant is not None:
            claim_execution_id(root / "live6-executions", grant.execution_id)
            provider.bind_execution(grant, execution_id=grant.execution_id)
    else:
        raise ValueError("MEDIA_ENGINE_EXECUTION_PROFILE")
    controller = MediaController(
        str(root / "studio.sqlite3"),
        str(root / "artifacts"),
        clock=clock,
        limits=limits,
        provider=provider,
        image_engine=RealProviderRequired(),
        stores=stores,
    )
    if stores is None:
        from media_engine.storage.r2_media import UnavailableMedia, open_media_store
        try:
            media = open_media_store()
        except Exception:
            media = UnavailableMedia()
        controller.platform = Platform(str(root / "platform.sqlite3"), media=media)
    return controller


def _local_credit_platform(root: Path) -> Platform:
    """Local ledger for the development studio. This does not grant credits."""
    platform = Platform(str(root / "platform.sqlite3"))
    if platform.repo.price_at("image", "qwen-image", platform.now_iso()) is None:
        platform.add_price(
            version="daidoi-e2e",
            service="image",
            model="qwen-image",
            quality="standard",
            unit="image",
            credit_price=2,
            internal_cost_units=0,
            effective_at="2026-01-01T00:00:00Z",
        )
    return platform


def budget_usd(environ: Optional[dict] = None) -> Optional[Decimal]:
    source = os.environ if environ is None else environ
    raw = str(source.get("STUDIO_GPU_BUDGET_USD", "")).strip()
    if not raw:
        return None
    amount = Decimal(raw)
    if amount <= 0:
        raise ValueError("STUDIO_GPU_BUDGET_USD")
    return amount


def _load_r2_if_needed(source: dict) -> None:
    from media_engine.providers.vast_image import cold_stage_allowed
    flag = str(source.get("MEDIA_ENGINE_ALLOW_COLD_STAGE", "")).strip().casefold()
    allowed = flag in {"1", "true", "yes"} if source is not os.environ else cold_stage_allowed()
    if not allowed:
        return
    path = Path(__file__).resolve().parents[3] / ".env.r2"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip("'\""))
