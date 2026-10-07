"""Local preview image. It is not the Qwen model and it does not contact a provider."""
from __future__ import annotations

import hashlib
import struct
import zlib

from media_engine.engines.image.base import ImageEngine


class DevImageEngine(ImageEngine):
    def render(self, prompt: str, width: int = 1024, height: int = 1024, seed: int = 0) -> bytes:
        digest = hashlib.sha256(f"{seed}:{prompt}".encode("utf-8")).digest()
        color = digest[:3]
        return _png(width, height, color, digest)


def _png(width: int, height: int, color: bytes, digest: bytes) -> bytes:
    row = b"\x00" + color * width
    compressed = zlib.compress(row * height)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    text = b"Comment\x00" + digest.hex().encode("ascii")
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"tEXt", text) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")
