"""Model invocation. Each function calls the runtime method the real library exposes."""
from __future__ import annotations

import json

from media_engine.engines.image.qwen import validate_png
from media_engine.models.audio import (
    decode_wav,
    duration_seconds,
    encode_wav,
    normalize_speech,
    parse_srt,
    render_srt,
    to_srt,
    to_vtt,
)
from media_engine.models.errors import AdapterError
from media_engine.models.video import frame_count, frame_size, validate_mp4, write_mp4

SPEAKERS = ("Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden")
TTS_RATE = 24000
TRANSLATION_QUALITY_VERIFIED = False
TRANSLATION_FIXTURES = (
    {
        "source_language": "English",
        "target_language": "Chinese",
        "text": "Hello",
        "accepted": ("你好",),
    },
)


class Invocation:
    def __init__(self, deadline: float, cancel) -> None:
        self.deadline = deadline
        self.cancel = cancel

    def check(self) -> None:
        import time
        if self.cancel is not None and self.cancel.is_set():
            raise AdapterError("CANCELLED")
        if time.monotonic() > self.deadline:
            raise AdapterError("TIMEOUT")


def invoke_transcribe(runtime, wav: bytes, ctx: Invocation) -> dict:
    ctx.check()
    samples = normalize_speech(wav)
    try:
        import numpy
        audio = numpy.asarray(samples, dtype="float32")
    except ImportError:
        audio = samples
    segments, info = runtime.transcribe(audio, word_timestamps=True)
    ctx.check()
    rows = [_segment(item) for item in segments]
    if not rows:
        raise AdapterError("TRANSCRIPT_EMPTY")
    language = str(getattr(info, "language", "") or "")
    if not language:
        raise AdapterError("LANGUAGE_UNDETECTED")
    text = " ".join(row["text"] for row in rows).strip()
    return {
        "text": text,
        "language": language,
        "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
        "segments": rows,
        "srt": to_srt(rows),
        "vtt": to_vtt(rows),
    }


def invoke_tts(runtime, payload: dict, ctx: Invocation) -> bytes:
    ctx.check()
    speaker = str(payload.get("voice") or "")
    if speaker not in SPEAKERS:
        raise AdapterError("SPEAKER_UNSUPPORTED")
    wavs, rate = runtime.generate_custom_voice(
        text=str(payload["text"]),
        language=str(payload["language"]),
        speaker=speaker,
    )
    ctx.check()
    return _wav_from_runtime(wavs, rate)


def invoke_voice_clone(runtime, payload: dict, reference: bytes, ctx: Invocation) -> bytes:
    import os
    import tempfile
    ctx.check()
    samples, rate = decode_wav(reference)
    heard = duration_seconds(samples, rate)
    if heard < 1 or heard > 30:
        raise AdapterError("REFERENCE_DURATION")
    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    handle.write(reference)
    handle.close()
    try:
        wavs, out_rate = runtime.generate_voice_clone(
            text=str(payload["text"]),
            language=str(payload["language"]),
            ref_audio=handle.name,
            ref_text=str(payload.get("reference_text") or ""),
        )
    finally:
        os.unlink(handle.name)
    ctx.check()
    return _wav_from_runtime(wavs, out_rate)


def translation_messages(source: str, target: str, text: str) -> list[dict]:
    return [{
        "role": "user",
        "content": (
            "Translate the text from " + source + " to " + target + ". "
            'Return one JSON object with a "translation" string and no other keys.\n\n'
            + text
        ),
    }]


def parse_translation(raw: str) -> str:
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise AdapterError("TRANSLATION_INVALID")
    try:
        payload = json.loads(raw[start:end + 1])
    except json.JSONDecodeError as exc:
        raise AdapterError("TRANSLATION_INVALID") from exc
    text = payload.get("translation") if isinstance(payload, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise AdapterError("TRANSLATION_INVALID")
    return text.strip()


def invoke_translation(runtime, payload: dict, ctx: Invocation) -> dict:
    ctx.check()
    source = str(payload["source_language"])
    target = str(payload["target_language"])
    if source == target:
        raise AdapterError("LANGUAGE_UNSUPPORTED")
    body = str(payload["text"])
    if "-->" in body:
        translated = translate_subtitles(runtime, body, source, target, ctx)
        return {"translation": translated, "format": "srt", "quality_verified": False}
    messages = translation_messages(source, target, body)
    raw = runtime.translate(messages) if hasattr(runtime, "translate") else _generate_text(runtime, messages)
    ctx.check()
    text = parse_translation(raw)
    return {"translation": text, "format": "text", "quality_verified": TRANSLATION_QUALITY_VERIFIED}


def translate_subtitles(runtime, srt: str, source: str, target: str, ctx: Invocation) -> str:
    cues = parse_srt(srt)
    translated = []
    for cue in cues:
        ctx.check()
        messages = translation_messages(source, target, cue["text"])
        raw = runtime.translate(messages) if hasattr(runtime, "translate") else _generate_text(runtime, messages)
        translated.append({
            "start": cue["start"],
            "end": cue["end"],
            "text": parse_translation(raw),
        })
    rendered = render_srt(translated)
    original = parse_srt(srt)
    current = parse_srt(rendered)
    if [(item["start"], item["end"]) for item in original] != [(item["start"], item["end"]) for item in current]:
        raise AdapterError("SUBTITLE_TIMING")
    return rendered


def evaluate_translation(runtime, ctx: Invocation) -> dict:
    """Score the injected runtime on fixtures. This does not mark the model GPU-verified."""
    results = []
    for fixture in TRANSLATION_FIXTURES:
        produced = invoke_translation(runtime, fixture, ctx)
        results.append({
            "text": fixture["text"],
            "passed": produced["translation"] in fixture["accepted"],
            "quality_verified": False,
        })
    return {"results": results, "quality_verified": False, "gpu_verified": False}


def invoke_image_edit(runtime, image: bytes, prompt: str, ctx: Invocation) -> bytes:
    ctx.check()
    validate_png(image)
    if callable(runtime) and not hasattr(runtime, "images"):
        data = runtime(image=image, prompt=prompt)
    else:
        result = runtime(image=image, prompt=prompt)
        data = getattr(result, "images", None)
        data = data[0] if isinstance(data, (list, tuple)) else data
        if hasattr(data, "save"):
            import io
            buffer = io.BytesIO()
            data.save(buffer, format="PNG")
            data = buffer.getvalue()
    ctx.check()
    if not isinstance(data, (bytes, bytearray)):
        raise AdapterError("PNG_INVALID")
    validate_png(bytes(data))
    return bytes(data)


def invoke_music(runtime, payload: dict, ctx: Invocation) -> bytes:
    ctx.check()
    duration = int(payload["duration_seconds"])
    samples, rate = runtime.generate(
        prompt=str(payload["prompt"]),
        duration_seconds=duration,
        style=str(payload["style"]),
    )
    ctx.check()
    wav = _wav_from_runtime(samples if not isinstance(samples, (bytes, bytearray)) else samples, rate)
    heard = duration_seconds(decode_wav(wav)[0], decode_wav(wav)[1])
    if abs(heard - duration) > 0.25:
        raise AdapterError("AUDIO_DURATION")
    return wav


def invoke_video(runtime, payload: dict, ctx: Invocation) -> bytes:
    ctx.check()
    width, height = frame_size(str(payload["aspect_ratio"]))
    frames = frame_count(int(payload["duration_seconds"]))
    produced = runtime(
        prompt=str(payload["prompt"]),
        width=width,
        height=height,
        num_frames=frames,
    )
    ctx.check()
    if isinstance(produced, (bytes, bytearray)):
        data = bytes(produced)
        validate_mp4(data)
        return data
    return write_mp4(b"".join(produced))


def _segment(item) -> dict:
    words = []
    for word in getattr(item, "words", None) or []:
        words.append({
            "word": str(getattr(word, "word", "")).strip(),
            "start": float(getattr(word, "start", 0.0)),
            "end": float(getattr(word, "end", 0.0)),
        })
    text = str(getattr(item, "text", "")).strip()
    if not text:
        raise AdapterError("TRANSCRIPT_EMPTY")
    return {
        "start": float(getattr(item, "start", 0.0)),
        "end": float(getattr(item, "end", 0.0)),
        "text": text,
        "words": words,
    }


def _wav_from_runtime(samples, rate) -> bytes:
    if isinstance(samples, (bytes, bytearray)):
        data = bytes(samples)
        decode_wav(data)
        return data
    if isinstance(samples, (list, tuple)) and samples and isinstance(samples[0], (list, tuple)):
        samples = samples[0]
    if not isinstance(rate, int) or isinstance(rate, bool) or rate <= 0:
        raise AdapterError("AUDIO_INVALID")
    return encode_wav([float(sample) for sample in samples], rate)


def _generate_text(runtime, messages: list[dict]) -> str:
    generated = runtime.generate(messages)
    if not isinstance(generated, str):
        raise AdapterError("TRANSLATION_INVALID")
    return generated
