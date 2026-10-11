"""MP4 container and Wan frame limits. Encoding a playable H.264 stream is a GPU step."""
from __future__ import annotations

import struct

from media_engine.models.errors import AdapterError

ASPECT_SIZE = {"16:9": (832, 480), "9:16": (480, 832), "1:1": (480, 480)}
FPS = 16
MAX_SECONDS = 5


def frame_count(duration_seconds: int) -> int:
    if isinstance(duration_seconds, bool) or not isinstance(duration_seconds, int):
        raise AdapterError("DURATION_LIMIT")
    if duration_seconds < 1 or duration_seconds > MAX_SECONDS:
        raise AdapterError("DURATION_LIMIT")
    return duration_seconds * FPS + 1


def frame_size(aspect_ratio: str) -> tuple[int, int]:
    try:
        return ASPECT_SIZE[aspect_ratio]
    except KeyError as exc:
        raise AdapterError("ASPECT_UNSUPPORTED") from exc


def write_mp4(payload: bytes) -> bytes:
    """ISO BMFF container. The GPU worker replaces the payload with an encoded stream."""
    if not payload:
        raise AdapterError("VIDEO_INVALID")
    brand = b"isom"
    ftyp_body = brand + b"\x00\x00\x00\x00" + brand
    ftyp = struct.pack(">I", 8 + len(ftyp_body)) + b"ftyp" + ftyp_body
    mdat = struct.pack(">I", 8 + len(payload)) + b"mdat" + payload
    data = ftyp + mdat
    validate_mp4(data)
    return data


def validate_mp4(data: bytes) -> None:
    if len(data) < 16 or data[4:8] != b"ftyp":
        raise AdapterError("VIDEO_INVALID")
