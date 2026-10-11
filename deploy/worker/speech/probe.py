"""Import probe for the speech image. A successful import is not CUDA proof."""
from __future__ import annotations

import importlib
import json


def main() -> None:
    payload = {"imports_ok": False, "torch_cuda_is_available": False}
    modules = {
        "faster_whisper": "faster_whisper",
        "ctranslate2": "ctranslate2",
        "qwen_tts": "qwen_tts",
        "transformers": "transformers",
        "accelerate": "accelerate",
        "torch": "torch",
        "torchaudio": "torchaudio",
    }
    import importlib.metadata as meta
    names = {
        "faster_whisper": "faster-whisper",
        "ctranslate2": "ctranslate2",
        "qwen_tts": "qwen-tts",
        "transformers": "transformers",
        "accelerate": "accelerate",
        "torch": "torch",
        "torchaudio": "torchaudio",
    }
    for key, module_name in modules.items():
        importlib.import_module(module_name)
        payload[key] = meta.version(names[key])
    torch = importlib.import_module("torch")
    payload["torch_cuda_is_available"] = bool(torch.cuda.is_available())
    payload["imports_ok"] = True
    payload["cuda_gpu_support"] = False
    payload["reason"] = "imports_are_not_gpu_proof"
    expected = {
        "faster_whisper": "1.2.1",
        "ctranslate2": "4.8.2",
        "qwen_tts": "0.1.1",
        "transformers": "4.57.3",
        "accelerate": "1.12.0",
        "torchaudio": "2.11.0",
    }
    for key, version in expected.items():
        if payload[key] != version:
            raise SystemExit("VERSION_MISMATCH")
    if not str(payload["torch"]).startswith("2.11."):
        raise SystemExit("TORCH_MISMATCH")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
