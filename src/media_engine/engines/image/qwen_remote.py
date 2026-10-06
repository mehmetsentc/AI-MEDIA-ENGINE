"""Runs on the rented GPU. Importing this file does not load the model."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


def main() -> int:
    job = json.loads(Path("/workspace/phase2d_job.json").read_text(encoding="utf-8"))
    report: dict[str, object] = {"model_id": "Qwen/Qwen-Image", "strategy": "bf16-cpu-offload"}
    cache = Path("/workspace/hf-cache")
    cache.mkdir(parents=True, exist_ok=True)
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
    download_started = time.perf_counter()
    try:
        local = snapshot_download("Qwen/Qwen-Image", cache_dir=str(cache))
    except Exception as exc:
        report["error"] = "DOWNLOAD_FAILED"
        report["detail"] = type(exc).__name__
        Path("/workspace/phase2d_report.json").write_text(json.dumps(report), encoding="utf-8")
        return 1
    report["download_seconds"] = round(time.perf_counter() - download_started, 3)
    report["downloaded_bytes"] = _cache_bytes(cache)
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
