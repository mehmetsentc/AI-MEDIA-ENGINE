"""Qwen-Image engine contract. The weights are not downloaded on import."""
from __future__ import annotations

import hashlib
import struct
import zlib
from typing import Callable

from media_engine.engines.image.base import ImageEngine, ImageEngineError

MODEL_ID = "Qwen/Qwen-Image"
LICENSE = "apache-2.0"
PIPELINE = "QwenImagePipeline"
FIRST_WIDTH = 1024
FIRST_HEIGHT = 1024
FIRST_SEED = 20260804
FIRST_PROMPT = (
    "A cinematic editorial photograph of a futuristic Mediterranean "
    "newsroom at night, large glass windows overlooking the sea, "
    "professional broadcast lighting, realistic architecture, "
    "subtle screens and newsroom desks, photorealistic, "
    "high detail, natural colors, no logos, no readable text"
)
INFERENCE_STEPS = 20

Pipeline = Callable[..., bytes]


class QwenImageEngine(ImageEngine):
    """Render one PNG through an injected pipeline.

    The live GPU runner supplies the real QwenImagePipeline. Tests supply bytes.
    """

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    def render(
        self,
        prompt: str,
        *,
        width: int = FIRST_WIDTH,
        height: int = FIRST_HEIGHT,
        seed: int = FIRST_SEED,
    ) -> bytes:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ImageEngineError("PROMPT_REQUIRED")
        if width != FIRST_WIDTH or height != FIRST_HEIGHT:
            raise ImageEngineError("RESOLUTION_UNSUPPORTED")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ImageEngineError("SEED_INVALID")
        raw = self._pipeline(prompt=prompt, width=width, height=height, seed=seed)
        if not isinstance(raw, (bytes, bytearray)) or not raw:
            raise ImageEngineError("PNG_INVALID")
        got_width, got_height = validate_png(bytes(raw))
        if got_width != width or got_height != height:
            raise ImageEngineError("RESOLUTION_MISMATCH")
        return bytes(raw)


def validate_png(data: bytes) -> tuple[int, int]:
    """Return width and height after the PNG signature and pixel data decode."""
    if len(data) < 8 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ImageEngineError("PNG_INVALID")
    pos = 8
    width = None
    height = None
    idat: list[bytes] = []
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        tag = data[pos + 4:pos + 8]
        start = pos + 8
        end = start + length
        if end + 4 > len(data):
            raise ImageEngineError("PNG_INVALID")
        chunk = data[start:end]
        if tag == b"IHDR":
            if length < 13:
                raise ImageEngineError("PNG_INVALID")
            width, height, bit_depth, color = struct.unpack(">IIBB", chunk[:10])
            if bit_depth != 8 or color not in (2, 6) or width < 1 or height < 1:
                raise ImageEngineError("PNG_INVALID")
        elif tag == b"IDAT":
            idat.append(chunk)
        elif tag == b"IEND":
            break
        pos = end + 4
    if width is None or height is None or not idat:
        raise ImageEngineError("PNG_INVALID")
    try:
        zlib.decompress(b"".join(idat))
    except zlib.error as exc:
        raise ImageEngineError("PNG_INVALID") from exc
    return width, height


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
