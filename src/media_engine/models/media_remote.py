"""Run one media job on a rented worker, from weights already staged there.

This is the remote half the GPU media path was missing. The image engine had
its own worker script and the speech side had none, so every capability but
images stopped at the local runtime.

Nothing here rents, creates or bills anything: by the time this runs the
authorization, the budget window and the instance already exist. It reads a job
file, loads one model from a directory of weights, runs it, and writes the
bytes plus a receipt. The caller fetches both.

Two rules this module keeps:

* **Weights are verified before they are loaded.** ``speech_stage`` writes a
  ``manifest.json`` beside the files it stages, with a size and a SHA-256 per
  file. A truncated transfer or a substituted checkpoint would otherwise show
  up as a strange-sounding voice on a GPU that is being paid for by the minute.
  The manifest travels with the weights, so this check needs no inventory and
  no network.
* **The dispatch is the local one.** It calls ``invoke`` rather than
  reimplementing it, so a fix to a capability reaches the worker too. The
  package is staged to the worker for exactly this reason.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

MANIFEST = "manifest.json"
CHUNK = 1024 * 1024

#: capability -> the extension its bytes are written with. A receipt names the
#: file, so the caller never guesses.
EXTENSIONS = {
    "audio.transcribe": "json",
    "audio.tts": "wav",
    "audio.voice_clone": "wav",
    "audio.translate": "json",
    "image.edit": "png",
    "music.generate": "wav",
    "video.text_to_video": "mp4",
}

#: capability -> the job field naming the input file, when it takes one.
INPUTS = {
    "audio.transcribe": "input",
    "audio.voice_clone": "reference",
    "image.edit": "source",
}


class RemoteJobError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def verify_weights(directory) -> dict:
    """Check a staged tree against the manifest staging left beside it.

    Returns a report rather than raising so a caller can log precisely what is
    wrong with a worker it is paying for.
    """
    directory = Path(directory)
    manifest = directory / MANIFEST
    if not manifest.is_file():
        return {"status": "NO_MANIFEST", "complete": False, "missing": [], "corrupt": []}
    try:
        body = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "MANIFEST_UNREADABLE", "complete": False, "missing": [], "corrupt": []}
    files = body.get("files")
    if not isinstance(files, list) or not files:
        return {"status": "MANIFEST_EMPTY", "complete": False, "missing": [], "corrupt": []}
    missing: list[str] = []
    corrupt: list[str] = []
    present = 0
    for item in files:
        relative = str(item.get("path") or "")
        path = directory / relative
        if not relative or ".." in Path(relative).parts or Path(relative).is_absolute():
            corrupt.append(relative)
            continue
        if not path.is_file():
            missing.append(relative)
            continue
        if path.stat().st_size != int(item.get("bytes", -1)):
            corrupt.append(relative)
            continue
        if _sha256(path) != str(item.get("sha256") or ""):
            corrupt.append(relative)
            continue
        present += 1
    complete = present == len(files) and not missing and not corrupt
    return {
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "complete": complete,
        "model_id": str(body.get("model_id") or ""),
        "revision": str(body.get("revision") or ""),
        "files_expected": len(files),
        "files_present": present,
        "missing": missing,
        "corrupt": corrupt,
    }


def run_job(
    job: dict,
    *,
    root,
    out_dir,
    in_dir=None,
    loaders=None,
    timeout_seconds: float = 900.0,
    clock=time.monotonic,
    verify=verify_weights,
) -> dict:
    """Run one job and write its bytes and receipt. Returns the receipt."""
    from media_engine.models.errors import AdapterError
    from media_engine.models.invoke import Invocation

    job_id = str(job.get("id") or "")
    capability = str(job.get("capability") or "")
    model_id = str(job.get("model_id") or "")
    if not job_id or not job_id.replace("-", "").replace("_", "").isalnum():
        raise RemoteJobError("JOB_ID_INVALID")
    if capability not in EXTENSIONS:
        raise RemoteJobError("CAPABILITY_UNKNOWN")
    if not model_id:
        raise RemoteJobError("MODEL_REQUIRED")
    payload = job.get("payload")
    if not isinstance(payload, dict):
        raise RemoteJobError("PAYLOAD_INVALID")

    directory = Path(root) / model_id
    report = verify(directory)
    if not report.get("complete"):
        raise RemoteJobError("WEIGHTS_" + str(report.get("status") or "INCOMPLETE"))
    revision = str(job.get("revision") or "")
    if revision and report.get("revision") and revision != report["revision"]:
        # Paying for a GPU to run a checkpoint nobody authorized is worse than
        # not running at all.
        raise RemoteJobError("REVISION_MISMATCH")

    table = _loaders() if loaders is None else loaders
    loader = table.get(model_id)
    if loader is None:
        raise RemoteJobError("MODEL_UNSUPPORTED")

    started = float(clock())
    runtime = loader(str(directory))
    loaded = float(clock())
    ctx = Invocation(float(clock()) + float(timeout_seconds), None)
    source = _input_bytes(job, capability, in_dir)
    try:
        result = _dispatch(capability, runtime, payload, source, ctx)
    except AdapterError as exc:
        raise RemoteJobError(exc.code) from exc

    data = result if isinstance(result, (bytes, bytearray)) else json.dumps(
        result, separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")
    data = bytes(data)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = job_id + "." + EXTENSIONS[capability]
    partial = out_dir / (name + ".partial")
    partial.write_bytes(data)
    partial.replace(out_dir / name)
    receipt = {
        "id": job_id,
        "capability": capability,
        "model_id": model_id,
        "revision": report.get("revision", ""),
        "file": name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "load_seconds": round(loaded - started, 3),
        "run_seconds": round(float(clock()) - loaded, 3),
        "files_verified": report.get("files_present", 0),
    }
    _write_json(out_dir / (job_id + ".json"), receipt)
    return receipt


def _dispatch(capability: str, runtime, payload: dict, source: bytes | None, ctx):
    from media_engine.models import invoke

    if capability == "audio.transcribe":
        return invoke.invoke_transcribe(runtime, _require(source), ctx)
    if capability == "audio.tts":
        return invoke.invoke_tts(runtime, payload, ctx)
    if capability == "audio.voice_clone":
        return invoke.invoke_voice_clone(runtime, payload, _require(source), ctx)
    if capability == "audio.translate":
        return invoke.invoke_translation(runtime, payload, ctx)
    if capability == "image.edit":
        return invoke.invoke_image_edit(
            runtime, _require(source), str(payload.get("prompt") or ""), ctx,
        )
    if capability == "music.generate":
        return invoke.invoke_music(runtime, payload, ctx)
    if capability == "video.text_to_video":
        return invoke.invoke_video(runtime, payload, ctx)
    raise RemoteJobError("CAPABILITY_UNKNOWN")


def _require(source: bytes | None) -> bytes:
    if not source:
        raise RemoteJobError("INPUT_REQUIRED")
    return source


def _input_bytes(job: dict, capability: str, in_dir) -> bytes | None:
    field = INPUTS.get(capability)
    if field is None:
        return None
    relative = str(job.get(field) or "")
    if not relative:
        raise RemoteJobError("INPUT_REQUIRED")
    if in_dir is None:
        raise RemoteJobError("INPUT_REQUIRED")
    path = Path(in_dir) / relative
    if ".." in Path(relative).parts or Path(relative).is_absolute():
        raise RemoteJobError("INPUT_PATH_INVALID")
    if not path.is_file():
        raise RemoteJobError("INPUT_MISSING")
    return path.read_bytes()


def _loaders() -> dict:
    from media_engine.models.local_bind import MODEL_LOADERS

    return MODEL_LOADERS


def _write_json(path: Path, body: dict) -> None:
    partial = Path(str(path) + ".partial")
    partial.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
    partial.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str]) -> int:
    """``python -m media_engine.models.media_remote <job.json> <root> <out> [in]``"""
    if len(argv) < 4:
        print("USAGE", flush=True)
        return 2
    job_path = Path(argv[1])
    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print("JOB_UNREADABLE", flush=True)
        return 3
    out_dir = Path(argv[3])
    try:
        receipt = run_job(
            job, root=argv[2], out_dir=out_dir, in_dir=argv[4] if len(argv) > 4 else None,
        )
    except RemoteJobError as exc:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(out_dir / (str(job.get("id") or "job") + ".error.json"), {"error": exc.code})
        print("JOB_FAILED " + exc.code, flush=True)
        return 4
    except Exception as exc:  # the worker is paid by the minute; say why and stop
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            out_dir / (str(job.get("id") or "job") + ".error.json"),
            {"error": "GENERATION_FAILED", "detail": type(exc).__name__},
        )
        print("JOB_FAILED GENERATION_FAILED", flush=True)
        return 5
    print("JOB_DONE " + json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
