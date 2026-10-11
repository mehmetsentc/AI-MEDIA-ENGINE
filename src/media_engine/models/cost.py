"""Credit estimates. The existing image price is not recomputed here."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING

# Planning constant for non-image estimates. It is not the image credit price.
PLANNING_CREDIT_USD = Decimal("0.05")

# Local ledger keys. These do not change the published image price.
CREDIT_RULES = {
    "audio.transcribe": ("transcribe", "faster-whisper-large-v3", 1),
    "audio.tts": ("tts", "qwen3-tts-custom", 1),
    "audio.voice_clone": ("voice-clone", "qwen3-tts-base", 2),
    "audio.translate": ("translate", "qwen2.5-1.5b-instruct", 1),
    "image.edit": ("image-edit", "qwen-image-edit", 2),
    "music.generate": ("music", "ace-step-v1-3.5b", 2),
    "video.text_to_video": ("video", "wan2.1-t2v-1.3b", 8),
}


@dataclass(frozen=True)
class CostProfile:
    name: str
    startup_seconds: Decimal
    load_seconds: Decimal
    inference_seconds: Decimal
    storage_overhead_usd: Decimal
    margin: Decimal
    fixed_image_price: bool = False


PROFILES = {
    "qwen-image-1024": CostProfile(
        "qwen-image-1024", Decimal("0"), Decimal("0"), Decimal("0"),
        Decimal("0"), Decimal("0"), True,
    ),
    "wan-t2v-5s": CostProfile(
        "wan-t2v-5s", Decimal("30"), Decimal("90"), Decimal("240"),
        Decimal("0.02"), Decimal("0.35"), False,
    ),
    "unpriced": CostProfile(
        "unpriced", Decimal("0"), Decimal("0"), Decimal("0"),
        Decimal("0"), Decimal("0"), False,
    ),
}


def estimate(profile_name: str, *, hourly_usd: Decimal, image_price: int = 2,
             credit_value_usd: Decimal = PLANNING_CREDIT_USD) -> dict:
    profile = PROFILES[profile_name]
    if profile.fixed_image_price:
        return {
            "profile": profile.name,
            "credits": int(image_price),
            "priced_by": "existing_image_price",
            "usd": None,
        }
    if profile.name == "unpriced":
        return {
            "profile": profile.name,
            "credits": None,
            "priced_by": "unpriced",
            "usd": None,
        }
    seconds = profile.startup_seconds + profile.load_seconds + profile.inference_seconds
    gpu = Decimal(hourly_usd) * seconds / Decimal(3600)
    usd = (gpu + profile.storage_overhead_usd) * (Decimal(1) + profile.margin)
    usd = usd.quantize(Decimal("0.0001"), rounding=ROUND_CEILING)
    if credit_value_usd <= 0:
        raise ValueError("CREDIT_VALUE")
    credits = int((usd / credit_value_usd).to_integral_value(rounding=ROUND_CEILING))
    return {
        "profile": profile.name,
        "credits": credits,
        "priced_by": "gpu_time",
        "usd": format(usd, "f"),
        "gpu_seconds": format(seconds, "f"),
    }
