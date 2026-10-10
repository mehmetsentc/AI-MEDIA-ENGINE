"""Pick one interpreter and keep it for import, load, and inference.

The pinned worker image installs torch and diffusers on system python3.
It also writes /opt/ai-media-engine/runtime-ready and does not create
/workspace/phase2d-venv. The ready marker is not an interpreter.

Selection probes a fixed list. A missing virtualenv is skipped. The first
executable that exists is accepted only when every required package imports.
A broken executable is a failure. The probe does not continue to another
Python, and it does not create a virtualenv or install packages.
"""
from __future__ import annotations

import json
import shlex

REQUIRED_MODULES = ("torch", "diffusers", "transformers", "accelerate", "safetensors")
VENV_PYTHON = "/workspace/phase2d-venv/bin/python"
SYSTEM_PYTHON = "python3"
READY_MARKER = "/opt/ai-media-engine/runtime-ready"
RUNTIME_RECORD = "/workspace/runtime-python.json"

SELECTOR_SCRIPT = r"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REQUIRED = ("torch", "diffusers", "transformers", "accelerate", "safetensors")
venv = os.environ.get("RUNTIME_VENV_PYTHON", "/workspace/phase2d-venv/bin/python")
system = os.environ.get("RUNTIME_SYSTEM_PYTHON", "python3")
marker = Path(os.environ.get("RUNTIME_MARKER", "/opt/ai-media-engine/runtime-ready"))
record_path = Path(os.environ.get("RUNTIME_RECORD", "/workspace/runtime-python.json"))


def resolve(executable):
    if "/" not in executable:
        return shutil.which(executable) or ""
    if os.path.isfile(executable) and os.access(executable, os.X_OK):
        return executable
    return ""


def run(executable, code):
    try:
        completed = subprocess.run(
            [executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, "", type(exc).__name__
    err = (completed.stderr or "").strip().replace("\n", " ")[:400]
    out = (completed.stdout or "").strip()
    return completed.returncode, out, err


def probe(executable):
    resolved = resolve(executable)
    row = {
        "executable": executable,
        "resolved": resolved,
        "state": "ABSENT",
        "version": "",
        "imports": {},
        "pipeline": False,
        "stderr": "",
    }
    if not resolved:
        return row
    code, out, err = run(resolved, "import sys; print(sys.version.split()[0])")
    if code != 0:
        row["state"] = "NOT_RUNNABLE"
        row["stderr"] = err or "NOT_RUNNABLE"
        return row
    row["version"] = out.split()[0] if out else ""
    missing = []
    for name in REQUIRED:
        imported, _out, err = run(resolved, "import " + name)
        row["imports"][name] = imported == 0
        if imported != 0:
            missing.append(name)
            row["stderr"] = err or ("missing " + name)
    pipe_code, pipe_out, pipe_err = run(
        resolved,
        "from diffusers import QwenImagePipeline; print(QwenImagePipeline.__name__)",
    )
    row["pipeline"] = pipe_code == 0 and pipe_out == "QwenImagePipeline"
    if pipe_code != 0 and not row["stderr"]:
        row["stderr"] = pipe_err or "QwenImagePipeline"
    if missing or not row["pipeline"]:
        row["state"] = "DEPENDENCY_MISSING"
        return row
    row["state"] = "OK"
    row["stderr"] = ""
    return row


candidates = []
selected = ""
failure = ""
version = ""
for executable in (venv, system):
    row = probe(executable)
    candidates.append(row)
    if row["state"] == "ABSENT":
        continue
    if row["state"] == "OK":
        selected = row["resolved"]
        version = row["version"]
        break
    failure = row["state"]
    break
if not selected and not failure:
    failure = "NO_INTERPRETER"
document = {
    "selected": selected,
    "version": version,
    "ready_marker": marker.is_file(),
    "failure": failure,
    "candidates": candidates,
}
text = json.dumps(document)
try:
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(text + "\n", encoding="utf-8")
except OSError:
    pass
print(text)
raise SystemExit(0 if selected else 1)
"""

# The marker is how the host recognizes this command. The script itself is stdlib only.
RUNTIME_SELECT_COMMAND = "python3 - << 'PY'\n# RUNTIME_PYTHON_SELECT\n" + SELECTOR_SCRIPT + "\nPY\n"


def parse_runtime_selection(text: str) -> dict:
    """Read the last JSON object printed by the remote probe."""
    for line in reversed(str(text).splitlines()):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {
        "selected": "",
        "version": "",
        "ready_marker": False,
        "failure": "SELECTION_UNREADABLE",
        "candidates": [],
    }


def accepted_executable(document: dict) -> str:
    """Return the selected path only when that same candidate imported the packages."""
    if not isinstance(document, dict) or document.get("failure"):
        return ""
    selected = str(document.get("selected") or "")
    if not selected:
        return ""
    for item in document.get("candidates") or []:
        if not isinstance(item, dict) or item.get("state") != "OK" or item.get("pipeline") is not True:
            continue
        allowed = {str(item.get("executable") or ""), str(item.get("resolved") or "")}
        allowed.discard("")
        if selected in allowed:
            return selected
    return ""


def import_probe_command(python: str) -> str:
    """Import QwenImagePipeline with the interpreter chosen above. Save its stderr."""
    if not python or python != python.strip() or any(char in python for char in "\n\r\x00"):
        raise ValueError("interpreter")
    quoted = shlex.quote(python)
    return (
        "# RUNTIME_IMPORT_PROBE\n"
        "set +e\n"
        f"{quoted} -c 'from diffusers import QwenImagePipeline; print(QwenImagePipeline.__name__)' "
        "> /workspace/import-probe.out 2> /workspace/import-probe.err\n"
        "code=$?\n"
        "printf '%s\\n' \"$code\" > /workspace/import-probe.exit\n"
        "cat /workspace/import-probe.out\n"
        "exit \"$code\"\n"
    )


def pinned_image_selection(executable: str = SYSTEM_PYTHON, version: str = "3.12.3") -> dict:
    """A successful system-python result, for tests that mock the remote probe."""
    return {
        "selected": executable,
        "version": version,
        "ready_marker": True,
        "failure": "",
        "candidates": [
            {
                "executable": VENV_PYTHON,
                "resolved": "",
                "state": "ABSENT",
                "version": "",
                "imports": {},
                "pipeline": False,
                "stderr": "",
            },
            {
                "executable": SYSTEM_PYTHON,
                "resolved": executable,
                "state": "OK",
                "version": version,
                "imports": {name: True for name in REQUIRED_MODULES},
                "pipeline": True,
                "stderr": "",
            },
        ],
    }
