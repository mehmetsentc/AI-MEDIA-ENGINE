"""Runs on the rented GPU. Importing this file does not load the model."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path


def _phase_seconds(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _write_report(report: dict) -> None:
    Path("/workspace/phase2d_report.json").write_text(json.dumps(report), encoding="utf-8")


def _record_exception(report: dict, error: str, exc: BaseException) -> None:
    """Keep the exception class. The message is one short line for the host log."""
    report["error"] = error
    report["detail"] = type(exc).__name__
    report["exception"] = type(exc).__name__
    report["exception_detail"] = " ".join(str(exc).split())[:180]
    _write_report(report)


PINNED_QWEN_REVISION = "75e0b4be04f60ec59a75f475837eced720f823b6"
WORKER_MOUNT = "/models"


def resolve_worker_mount(job: dict | None = None, default: str = WORKER_MOUNT) -> Path:
    """The worker reads MEDIA_ENGINE_MODEL_CACHE. A missing value stays on /models."""
    configured = os.environ.get("MEDIA_ENGINE_MODEL_CACHE", "").strip()
    if configured:
        return Path(configured)
    if isinstance(job, dict) and job.get("cache_mount"):
        return Path(str(job["cache_mount"]))
    return Path(default)


def cache_is_ready(mount: Path, *, min_bytes: int | None = None) -> bool:
    """True only when this worker directory already has the pinned Qwen snapshot."""
    assess, read_manifest, complete, minimum = _worker_cache_api()
    floor = minimum if min_bytes is None else int(min_bytes)
    if assess(mount, min_bytes=floor) != complete:
        return False
    manifest = read_manifest(mount) or {}
    return str(manifest.get("revision") or "") == PINNED_QWEN_REVISION


class SessionServer:
    """One process. The model is loaded on the first job and kept for the rest."""

    def __init__(self, mount: Path, *, cache_ready, sync, load, generate, limit: int = 3) -> None:
        self.mount = mount
        self.cache_ready = cache_ready
        self.sync = sync
        self.load = load
        self.generate = generate
        self.limit = int(limit)
        self.pipe = None
        self.downloads = 0
        self.loads = 0
        self.done: dict[str, object] = {}

    def submit(self, job: dict):
        identity = str(job.get("id") or "")
        if not identity:
            raise RuntimeError("JOB_ID_REQUIRED")
        if identity in self.done:
            return self.done[identity]
        if len(self.done) >= self.limit:
            raise RuntimeError("SESSION_IMAGE_LIMIT")
        if self.pipe is None:
            if not self.cache_ready(self.mount):
                self.sync(self.mount)
                self.downloads = 1
                if not self.cache_ready(self.mount):
                    raise RuntimeError("MODEL_CACHE_NOT_READY")
            self.pipe = self.load(self.mount)
            self.loads = 1
        image = self.generate(self.pipe, job)
        self.done[identity] = image
        return image


def run_session(jobs: list[dict], *, mount: Path, cache_ready, sync, load, generate) -> dict:
    """One load for every job. Download only when the worker cache is not ready."""
    if not jobs:
        raise RuntimeError("JOBS_REQUIRED")
    downloads = 0
    if not cache_ready(mount):
        sync(mount)
        downloads = 1
        if not cache_ready(mount):
            raise RuntimeError("MODEL_CACHE_NOT_READY")
    pipe = load(mount)
    outputs = [generate(pipe, job) for job in jobs]
    return {"downloads": downloads, "loads": 1, "jobs": len(outputs), "outputs": outputs}


def main() -> int:
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    if phase == "serve":
        return _serve()
    job = json.loads(Path("/workspace/phase2d_job.json").read_text(encoding="utf-8"))
    report: dict[str, object] = {
        "model_id": "Qwen/Qwen-Image",
        "strategy": "bf16-cpu-offload",
        "operation": "PHASE2D_GENERATE",
    }
    prepare_generation, cache_error, mount_path = _cache_api()
    mount = resolve_worker_mount(job, mount_path)
    cache = mount / "hf-cache"
    os.environ["HF_HOME"] = str(cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache)
    if phase == "cache-sync":
        report["stage"] = "CACHE_SYNC"
        try:
            if cache_is_ready(mount):
                report["cache"] = "HIT"
            elif os.environ.get("MODEL_CACHE_PROVIDER") == "r2":
                _sync_r2(mount, deadline_seconds=_phase_seconds("PHASE2D_CACHE_SYNC_SECONDS", 1200))
                report["cache"] = "STAGED"
            else:
                raise cache_error("INVALID")
        except TimeoutError:
            report["error"] = "MODEL_DOWNLOAD_FAILED"
            report["detail"] = "CACHE_SYNC_DEADLINE"
            _write_report(report)
            return 1
        except cache_error as exc:
            report["error"] = "MODEL_CACHE_NOT_READY"
            report["detail"] = exc.state
            _write_report(report)
            return 1
        except Exception as exc:
            report["error"] = "MODEL_CACHE_NOT_READY"
            report["detail"] = type(exc).__name__
            _write_report(report)
            return 1
        report["state"] = "COMPLETE"
        _write_report(report)
        return 0
    try:
        import resource
        import torch
        from diffusers import QwenImagePipeline
        from huggingface_hub import snapshot_download
        import diffusers
        import transformers
    except Exception as exc:
        _record_exception(report, "IMPORT_FAILED", exc)
        return 1
    report["torch"] = torch.__version__
    report["cuda"] = torch.version.cuda
    report["diffusers"] = diffusers.__version__
    report["transformers"] = transformers.__version__
    try:
        if phase != "generate" and os.environ.get("MODEL_CACHE_PROVIDER") == "r2" and not cache_is_ready(mount):
            _sync_r2(mount, deadline_seconds=_phase_seconds("PHASE2D_CACHE_SYNC_SECONDS", 1200))
        local = prepare_generation(mount, snapshot_download)
    except TimeoutError:
        report["error"] = "MODEL_DOWNLOAD_FAILED"
        report["detail"] = "CACHE_SYNC_DEADLINE"
        _write_report(report)
        return 1
    except cache_error as exc:
        report["error"] = "MODEL_CACHE_NOT_READY"
        report["detail"] = exc.state
        _write_report(report)
        return 1
    except Exception as exc:
        report["error"] = "MODEL_CACHE_NOT_READY"
        report["detail"] = type(exc).__name__
        _write_report(report)
        return 1
    load_limit = _phase_seconds("PHASE2D_MODEL_LOAD_SECONDS", 300)
    load_started = time.perf_counter()
    try:
        pipe = QwenImagePipeline.from_pretrained(
            local, torch_dtype=torch.bfloat16, local_files_only=True,
        )
        pipe.enable_model_cpu_offload()
        if time.perf_counter() - load_started > load_limit:
            report["error"] = "LOAD_FAILED"
            report["detail"] = "MODEL_LOAD_DEADLINE"
            report["exception"] = "MODEL_LOAD_DEADLINE"
            _write_report(report)
            return 1
    except Exception as exc:
        _record_exception(report, "LOAD_FAILED", exc)
        return 1
    report["load_seconds"] = round(time.perf_counter() - load_started, 3)
    report["peak_rss_kb"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    gen_limit = _phase_seconds("PHASE2D_GENERATION_SECONDS", 300)
    started = time.perf_counter()
    box: dict[str, object] = {}

    def _run_pipe() -> None:
        try:
            generator = torch.Generator(device="cpu").manual_seed(int(job["seed"]))
            box["image"] = pipe(
                prompt=job["prompt"],
                width=int(job["width"]),
                height=int(job["height"]),
                num_inference_steps=int(job["steps"]),
                generator=generator,
            ).images[0]
        except Exception as exc:
            box["error"] = exc

    worker = threading.Thread(target=_run_pipe, daemon=True)
    worker.start()
    worker.join(gen_limit)
    if worker.is_alive():
        report["error"] = "GENERATION_DEADLINE"
        report["detail"] = "GENERATION_DEADLINE"
        report["inference_seconds"] = round(time.perf_counter() - started, 3)
        _write_report(report)
        os._exit(1)
    if "error" in box:
        report["error"] = type(box["error"]).__name__
        report["detail"] = type(box["error"]).__name__
        _write_report(report)
        return 1
    image = box["image"]
    report["inference_seconds"] = round(time.perf_counter() - started, 3)
    if torch.cuda.is_available():
        report["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
    out = Path("/workspace/phase2d_output.png")
    image.save(out, format="PNG")
    report["output_bytes"] = out.stat().st_size
    report["generation_seconds"] = report["inference_seconds"]
    _write_report(report)
    return 0


def _load_workspace_module(name: str, path: str):
    """Load a copied worker module. dataclass needs it in sys.modules first."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def session_image_limit() -> int:
    """The rented process stops at the authorized count. A missing value does not become three."""
    raw = os.environ.get("MEDIA_ENGINE_SESSION_IMAGE_LIMIT", "").strip()
    if raw not in {"1", "2", "3"}:
        raise RuntimeError("SESSION_IMAGE_LIMIT")
    return int(raw)


def _serve() -> int:
    """Stay up, load Qwen once, and write one PNG per inbox job. Not the one-shot command."""
    try:
        limit = session_image_limit()
    except RuntimeError:
        return 2
    mount = resolve_worker_mount(None, WORKER_MOUNT)
    os.environ["HF_HOME"] = str(mount / "hf-cache")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(mount / "hf-cache")
    inbox = Path("/workspace/session-inbox")
    output = Path("/workspace/session-out")
    inbox.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    if os.environ.get("MODEL_CACHE_PROVIDER") == "r2" and not cache_is_ready(mount):
        _sync_r2(mount, deadline_seconds=_phase_seconds("PHASE2D_CACHE_SYNC_SECONDS", 1200))
    import torch
    from diffusers import QwenImagePipeline
    from huggingface_hub import snapshot_download
    prepare_generation, _cache_error, _mount = _cache_api()
    local = prepare_generation(mount, snapshot_download)
    pipe = QwenImagePipeline.from_pretrained(local, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.enable_model_cpu_offload()
    Path("/workspace/session.ready").write_text("ready\n", encoding="utf-8")
    seen: set[str] = set()
    completed = 0
    while completed < limit and not Path("/workspace/session.stop").exists():
        progressed = False
        for path in sorted(inbox.glob("*.json")):
            if path.name in seen:
                continue
            job = json.loads(path.read_text(encoding="utf-8"))
            identity = str(job.get("id") or path.stem)
            if (output / (identity + ".png")).is_file():
                seen.add(path.name)
                continue
            generator = torch.Generator(device="cpu").manual_seed(int(job["seed"]))
            image = pipe(
                prompt=job["prompt"],
                width=int(job["width"]),
                height=int(job["height"]),
                num_inference_steps=int(job.get("steps") or 30),
                generator=generator,
            ).images[0]
            image.save(output / (identity + ".png"), format="PNG")
            seen.add(path.name)
            completed += 1
            progressed = True
            if completed >= limit:
                break
        if not progressed:
            time.sleep(1)
    return 0


def _worker_cache_api():
    try:
        from media_engine.engines.image.model_cache import (
            COMPLETE,
            QWEN_CACHE_MIN_BYTES,
            assess_cache,
            read_manifest,
        )
    except ImportError:
        module = _load_workspace_module("phase2d_model_cache", "/workspace/model_cache.py")
        return module.assess_cache, module.read_manifest, module.COMPLETE, module.QWEN_CACHE_MIN_BYTES
    return assess_cache, read_manifest, COMPLETE, QWEN_CACHE_MIN_BYTES


def _cache_api():
    try:
        from media_engine.engines.image.model_cache import (
            MOUNT_PATH,
            ModelCacheNotReady,
            prepare_generation,
        )
    except ImportError:
        module = _load_workspace_module("phase2d_model_cache", "/workspace/model_cache.py")
        return module.prepare_generation, module.ModelCacheNotReady, module.MOUNT_PATH
    return prepare_generation, ModelCacheNotReady, MOUNT_PATH


def _sync_r2(mount: Path, deadline_seconds: float | None = None) -> None:
    progress_path = Path("/workspace/r2-sync-progress.json")
    if not progress_path.parent.exists():
        progress_path = None

    def on_progress(record: dict) -> None:
        print(json.dumps(record), flush=True)

    try:
        from media_engine.engines.image.r2_cache import sync_from_env
    except ImportError:
        module = _load_workspace_module("phase2d_r2_cache", "/workspace/r2_cache.py")
        sync_from_env = module.sync_from_env
    sync_from_env(
        mount,
        on_progress=on_progress,
        progress_path=progress_path,
        deadline_seconds=deadline_seconds,
    )


def download_progress_snapshot(cache: Path) -> dict[str, object]:
    """Bytes and incomplete blob names. The blob name is a hash, not a secret."""
    usage = shutil.disk_usage(cache if cache.exists() else cache.parent)
    cache_bytes = 0
    complete = 0
    last_incomplete = ""
    last_mtime = -1.0
    if cache.exists():
        for path in cache.rglob("*"):
            if not path.is_file():
                continue
            stat = path.stat()
            cache_bytes += stat.st_size
            if path.name.endswith(".incomplete"):
                if stat.st_mtime >= last_mtime:
                    last_mtime = stat.st_mtime
                    last_incomplete = path.name
            else:
                complete += 1
    return {
        "stage": "DOWNLOAD",
        "cache_bytes": cache_bytes,
        "free_bytes": usage.free,
        "files_complete": complete,
        "last_active_file": last_incomplete,
    }


def _watch_download(cache: Path, stop: threading.Event) -> None:
    while not stop.wait(15):
        print(json.dumps({"at": time.time(), **download_progress_snapshot(cache)}), flush=True)


def _cache_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return 0
    for path in root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


if __name__ == "__main__":
    raise SystemExit(main())
