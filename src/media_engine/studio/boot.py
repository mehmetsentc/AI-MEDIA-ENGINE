"""Studio process setup. Development stays on the fake provider."""
from __future__ import annotations

import os
import time
from decimal import Decimal
from pathlib import Path
from typing import Optional

from media_engine.engines.image.base import ImageEngine, ImageEngineError
from media_engine.orchestrator.controller import MediaController, WallClock
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
    key = str(source.get("VAST_API_KEY", "")).strip()
    if not key:
        raise ValueError("VAST_API_KEY")
    _load_r2_if_needed(source)
    from media_engine.providers.vast_image import VastImageProvider
    limits = limits_from_env(SafetyLimits(
        live_external_providers=True, max_gpu_workers=1, max_job_attempts=1,
    ))
    provider = VastImageProvider(api_key=key, limits=limits, now=clock.now, runtime=root)
    return MediaController(
        str(root / "studio.sqlite3"),
        str(root / "artifacts"),
        clock=clock,
        limits=limits,
        provider=provider,
        image_engine=RealProviderRequired(),
    )


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
