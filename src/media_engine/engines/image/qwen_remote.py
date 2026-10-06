"""Runs on the rented GPU. Importing this file does not load the model."""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path


def main() -> int:
    job = json.loads(Path("/workspace/phase2d_job.json").read_text(encoding="utf-8"))
    report: dict[str, object] = {
        "model_id": "Qwen/Qwen-Image",
        "strategy": "bf16-cpu-offload",
        "operation": "PHASE2D_GENERATE",
    }
    prepare_generation, cache_error, mount_path = _cache_api()
    mount = Path(str(job.get("cache_mount") or mount_path))
    cache = mount / "hf-cache"
    os.environ["HF_HOME"] = str(cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache)
    try:
        import resource
        import torch
        from diffusers import QwenImagePipeline
        from huggingface_hub import snapshot_download
        import diffusers
        import transformers
    except Exception as exc:
        report["error"] = "IMPORT_FAILED"
        report["detail"] = type(exc).__name__
        Path("/workspace/phase2d_report.json").write_text(json.dumps(report), encoding="utf-8")
        return 1
    report["torch"] = torch.__version__
    report["cuda"] = torch.version.cuda
    report["diffusers"] = diffusers.__version__
    report["transformers"] = transformers.__version__
    try:
        if os.environ.get("MODEL_CACHE_PROVIDER") == "r2":
            _sync_r2(mount)
        local = prepare_generation(mount, snapshot_download)
    except cache_error as exc:
        report["error"] = "MODEL_CACHE_NOT_READY"
        report["detail"] = exc.state
        Path("/workspace/phase2d_report.json").write_text(json.dumps(report), encoding="utf-8")
        return 1
    except Exception as exc:
        report["error"] = "MODEL_CACHE_NOT_READY"
        report["detail"] = type(exc).__name__
        Path("/workspace/phase2d_report.json").write_text(json.dumps(report), encoding="utf-8")
        return 1
    load_started = time.perf_counter()
    try:
        pipe = QwenImagePipeline.from_pretrained(
            local, torch_dtype=torch.bfloat16, local_files_only=True,
        )
        pipe.enable_model_cpu_offload()
    except Exception as exc:
        report["error"] = "LOAD_FAILED"
        report["detail"] = type(exc).__name__
        Path("/workspace/phase2d_report.json").write_text(json.dumps(report), encoding="utf-8")
        return 1
    report["load_seconds"] = round(time.perf_counter() - load_started, 3)
    report["peak_rss_kb"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    generator = torch.Generator(device="cpu").manual_seed(int(job["seed"]))
    image = pipe(
        prompt=job["prompt"],
        width=int(job["width"]),
        height=int(job["height"]),
        num_inference_steps=int(job["steps"]),
        generator=generator,
    ).images[0]
    report["inference_seconds"] = round(time.perf_counter() - started, 3)
    if torch.cuda.is_available():
        report["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
    out = Path("/workspace/phase2d_output.png")
    image.save(out, format="PNG")
    report["output_bytes"] = out.stat().st_size
    Path("/workspace/phase2d_report.json").write_text(json.dumps(report), encoding="utf-8")
    return 0


def _cache_api():
    try:
        from media_engine.engines.image.model_cache import (
            MOUNT_PATH,
            ModelCacheNotReady,
            prepare_generation,
        )
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location("phase2d_model_cache", "/workspace/model_cache.py")
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.prepare_generation, module.ModelCacheNotReady, module.MOUNT_PATH
    return prepare_generation, ModelCacheNotReady, MOUNT_PATH


def _sync_r2(mount: Path) -> None:
    try:
        from media_engine.engines.image.r2_cache import sync_from_env
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location("phase2d_r2_cache", "/workspace/r2_cache.py")
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sync_from_env = module.sync_from_env
    sync_from_env(mount)


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
