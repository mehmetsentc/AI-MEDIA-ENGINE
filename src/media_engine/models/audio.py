"""WAV decode, 16 kHz mono normalization, and subtitle text. No model weights."""
from __future__ import annotations

import array
import io
import wave

from media_engine.models.errors import AdapterError


def decode_wav(data: bytes) -> tuple[list[float], int]:
    if not data.startswith(b"RIFF") or b"WAVE" not in data[:16]:
        raise AdapterError("AUDIO_FORMAT")
    try:
        handle = wave.open(io.BytesIO(data), "rb")
    except wave.Error as exc:
        raise AdapterError("AUDIO_FORMAT") from exc
    try:
        channels = handle.getnchannels()
        rate = handle.getframerate()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    finally:
        handle.close()
    if width != 2 or channels < 1 or rate <= 0 or not frames:
        raise AdapterError("AUDIO_FORMAT")
    samples = array.array("h")
    samples.frombytes(frames[: len(frames) - (len(frames) % 2)])
    mono: list[float] = []
    step = channels
    for index in range(0, len(samples) - channels + 1, step):
        total = 0
        for channel in range(channels):
            total += samples[index + channel]
        mono.append(total / channels / 32768.0)
    if not mono:
        raise AdapterError("AUDIO_FORMAT")
    return mono, rate


def resample(samples: list[float], source_rate: int, target_rate: int = 16000) -> list[float]:
    if source_rate <= 0 or target_rate <= 0 or not samples:
        raise AdapterError("AUDIO_FORMAT")
    if source_rate == target_rate:
        return list(samples)
    count = max(1, int(len(samples) * target_rate / source_rate))
    output: list[float] = []
    last = len(samples) - 1
    for index in range(count):
        position = index * source_rate / target_rate
        left = int(position)
        if left >= last:
            output.append(samples[last])
            continue
        fraction = position - left
        output.append(samples[left] * (1 - fraction) + samples[left + 1] * fraction)
    return output


def normalize_speech(data: bytes) -> list[float]:
    samples, rate = decode_wav(data)
    return resample(samples, rate, 16000)


def encode_wav(samples: list[float], rate: int) -> bytes:
    if rate <= 0 or not samples:
        raise AdapterError("AUDIO_INVALID")
    clipped = array.array("h")
    for sample in samples:
        value = int(max(-1.0, min(1.0, float(sample))) * 32767)
        clipped.append(value)
    buffer = io.BytesIO()
    handle = wave.open(buffer, "wb")
    try:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(clipped.tobytes())
    finally:
        handle.close()
    data = buffer.getvalue()
    if not data.startswith(b"RIFF") or b"WAVE" not in data[:16]:
        raise AdapterError("AUDIO_INVALID")
    return data


def duration_seconds(samples: list[float], rate: int) -> float:
    if rate <= 0:
        raise AdapterError("AUDIO_FORMAT")
    return len(samples) / rate


def to_srt(segments: list[dict]) -> str:
    lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        lines.append(str(index))
        lines.append(_stamp(float(segment["start"])) + " --> " + _stamp(float(segment["end"])))
        lines.append(str(segment["text"]).strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def to_vtt(segments: list[dict]) -> str:
    body = to_srt(segments).replace(",", ".")
    return "WEBVTT\n\n" + body


def parse_srt(text: str) -> list[dict]:
    cues: list[dict] = []
    blocks = [block.strip() for block in text.replace("\r\n", "\n").split("\n\n") if block.strip()]
    for block in blocks:
        rows = block.split("\n")
        if len(rows) < 3 or "-->" not in rows[1]:
            raise AdapterError("SUBTITLE_INVALID")
        start, end = [part.strip() for part in rows[1].split("-->")]
        cues.append({"start": start, "end": end, "text": "\n".join(rows[2:]).strip()})
    if not cues:
        raise AdapterError("SUBTITLE_INVALID")
    return cues


def render_srt(cues: list[dict]) -> str:
    lines: list[str] = []
    for index, cue in enumerate(cues, start=1):
        lines.append(str(index))
        lines.append(cue["start"] + " --> " + cue["end"])
        lines.append(cue["text"])
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _stamp(seconds: float) -> str:
    whole = max(0, int(seconds))
    millis = int(round((seconds - whole) * 1000))
    if millis == 1000:
        whole += 1
        millis = 0
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
