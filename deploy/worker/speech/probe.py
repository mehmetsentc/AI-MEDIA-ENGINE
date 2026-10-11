"""Import probe for the speech image. A successful import is not CUDA proof."""
from __future__ import annotations

import importlib
import json

METHOD = "importlib.metadata"
EXPECTED = {
    "faster_whisper": "1.2.1",
    "ctranslate2": "4.8.2",
    "qwen_tts": "0.1.1",
    "transformers": "4.57.3",
    "accelerate": "1.12.0",
    "torchaudio": "2.11.0",
}
DISTRIBUTIONS = {
    "faster_whisper": "faster-whisper",
    "ctranslate2": "ctranslate2",
    "qwen_tts": "qwen-tts",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "torch": "torch",
    "torchaudio": "torchaudio",
}
MODULES = {
    "faster_whisper": "faster_whisper",
    "ctranslate2": "ctranslate2",
    "qwen_tts": "qwen_tts",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "torch": "torch",
    "torchaudio": "torchaudio",
}


def release_matches(actual: str, expected: str) -> bool:
    """Accept the pinned release or that release plus one local label.

    The CUDA base image publishes torchaudio as 2.11.0+cu128. That is the
    2.11.0 distribution, not a different release.
    """
    if actual == expected:
        return True
    release, separator, local = actual.partition("+")
    return separator == "+" and release == expected and bool(local) and "+" not in local


def version_failure(code: str, package: str, expected: str, actual: str) -> SystemExit:
    return SystemExit(
        f"{code} package={package} expected={expected} actual={actual} method={METHOD}"
    )


def check_distribution_versions(actual: dict[str, str]) -> dict[str, str]:
    """Return the verified release for each pin. Mismatches name the package."""
    verified: dict[str, str] = {}
    for key, expected in EXPECTED.items():
        package = DISTRIBUTIONS[key]
        if key not in actual:
            raise version_failure("MISSING_DISTRIBUTION", package, expected, "absent")
        found = actual[key]
        if not release_matches(found, expected):
            raise version_failure("VERSION_MISMATCH", package, expected, found)
        verified[key] = expected
        verified[key + "_distribution"] = found
    torch_version = actual.get("torch")
    if torch_version is None:
        raise version_failure("MISSING_DISTRIBUTION", "torch", "2.11.x", "absent")
    if not str(torch_version).startswith("2.11."):
        raise version_failure("TORCH_MISMATCH", "torch", "2.11.x", torch_version)
    verified["torch"] = torch_version
    verified["torch_distribution"] = torch_version
    return verified


def main() -> None:
    import importlib.metadata as meta
    payload = {"imports_ok": False, "torch_cuda_is_available": False}
    found: dict[str, str] = {}
    for key, module_name in MODULES.items():
        importlib.import_module(module_name)
        package = DISTRIBUTIONS[key]
        try:
            found[key] = meta.version(package)
        except meta.PackageNotFoundError as exc:
            expected = EXPECTED.get(key, "2.11.x")
            raise version_failure("MISSING_DISTRIBUTION", package, expected, "absent") from exc
    torch = importlib.import_module("torch")
    payload.update(check_distribution_versions(found))
    payload["torch_cuda_is_available"] = bool(torch.cuda.is_available())
    payload["imports_ok"] = True
    payload["cuda_gpu_support"] = False
    payload["reason"] = "imports_are_not_gpu_proof"
    payload["version_method"] = METHOD
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
