"""Interpreter selection and model-load evidence. No GPU and no model weights."""
from __future__ import annotations

import base64
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from media_engine.orchestrator.controller import _DESTROY_ON
from media_engine.providers.base import WorkerStageError
from media_engine.providers.runtime_python import (
    REQUIRED_MODULES,
    RUNTIME_SELECT_COMMAND,
    SELECTOR_SCRIPT,
    VENV_PYTHON,
    accepted_executable,
    import_probe_command,
    parse_runtime_selection,
    pinned_image_selection,
)
from media_engine.providers.stage_evidence import (
    CAPTURE_TIMEOUT_SECONDS,
    parse_runtime_bundle,
    preserve_runtime_evidence,
)
from media_engine.providers.vast_image import VastImageProvider, _generate_command
from tests.test_phase2e import PNG


def _fake_python(path: Path, *, version: str = "3.12.3", missing: tuple[str, ...] = (),
                 pipeline: bool = True, dead: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    missing_names = " ".join(missing)
    script = f"""#!/bin/sh
code="$2"
if [ "{1 if dead else 0}" = "1" ]; then
  echo dead >&2
  exit 1
fi
case "$code" in
  *sys.version*)
    printf '%s\\n' "{version}"
    ;;
  *QwenImagePipeline*)
    if [ "{0 if pipeline else 1}" = "1" ]; then
      echo 'ImportError: QwenImagePipeline' >&2
      exit 1
    fi
    printf '%s\\n' QwenImagePipeline
    ;;
  import\\ *)
    name="${{code#import }}"
    case " {missing_names} " in
      *" $name "*)
        echo "ImportError: $name" >&2
        exit 1
        ;;
    esac
    ;;
  *)
    echo "unexpected $code" >&2
    exit 2
    ;;
esac
exit 0
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _run_selector(**env: str) -> subprocess.CompletedProcess[str]:
    full = os.environ.copy()
    full.update(env)
    return subprocess.run(
        [sys.executable, "-c", SELECTOR_SCRIPT],
        env=full,
        capture_output=True,
        text=True,
        check=False,
    )


class SelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.marker = self.root / "runtime-ready"
        self.record = self.root / "runtime-python.json"
        self.venv = self.root / "missing-venv" / "bin" / "python"
        self.system = self.root / "system" / "python3"

    def _env(self, **extra: str) -> dict[str, str]:
        values = {
            "RUNTIME_VENV_PYTHON": str(self.venv),
            "RUNTIME_SYSTEM_PYTHON": str(self.system),
            "RUNTIME_MARKER": str(self.marker),
            "RUNTIME_RECORD": str(self.record),
        }
        values.update(extra)
        return values

    def test_system_python_is_selected_when_the_virtualenv_is_missing(self) -> None:
        _fake_python(self.system)
        completed = _run_selector(**self._env())
        self.assertEqual(completed.returncode, 0, completed.stderr)
        document = json.loads(completed.stdout)
        self.assertEqual(document["selected"], str(self.system))
        self.assertEqual(document["version"], "3.12.3")
        self.assertEqual(document["failure"], "")
        self.assertFalse(self.venv.exists())
        self.assertEqual(document["candidates"][0]["state"], "ABSENT")
        self.assertEqual(document["candidates"][1]["imports"]["torch"], True)
        self.assertTrue(document["candidates"][1]["pipeline"])

    def test_a_valid_virtualenv_is_kept_and_system_python_is_not_probed(self) -> None:
        _fake_python(self.venv, version="3.11.9")
        _fake_python(self.system, version="9.9.9")
        completed = _run_selector(**self._env())
        document = json.loads(completed.stdout)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(document["selected"], str(self.venv))
        self.assertEqual(document["version"], "3.11.9")
        self.assertEqual(len(document["candidates"]), 1)
        self.assertNotEqual(document["selected"], str(self.system))

    def test_ready_marker_does_not_invent_a_virtualenv(self) -> None:
        self.marker.parent.mkdir(parents=True, exist_ok=True)
        self.marker.write_text("ready\n", encoding="utf-8")
        _fake_python(self.system)
        completed = _run_selector(**self._env())
        document = json.loads(completed.stdout)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(document["ready_marker"])
        self.assertEqual(document["selected"], str(self.system))
        self.assertNotEqual(document["selected"], VENV_PYTHON)
        self.assertFalse(self.venv.exists())
        saved = json.loads(self.record.read_text(encoding="utf-8"))
        self.assertEqual(saved["selected"], str(self.system))

    def test_missing_dependency_stops_without_installing_or_switching(self) -> None:
        _fake_python(self.venv, missing=("diffusers",), pipeline=False)
        _fake_python(self.system)
        before = set(self.root.rglob("*"))
        completed = _run_selector(**self._env())
        document = json.loads(completed.stdout)
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(document["selected"], "")
        self.assertEqual(document["failure"], "DEPENDENCY_MISSING")
        self.assertEqual(len(document["candidates"]), 1)
        self.assertIn("diffusers", document["candidates"][0]["stderr"])
        self.assertFalse(document["candidates"][0]["imports"]["diffusers"])
        self.assertEqual(set(self.root.rglob("*")) - before, {self.record})
        self.assertNotIn("pip install", RUNTIME_SELECT_COMMAND)
        self.assertNotIn("python3 -m venv", RUNTIME_SELECT_COMMAND)

    def test_incorrect_python_path_is_a_clear_failure(self) -> None:
        completed = _run_selector(**self._env(
            RUNTIME_VENV_PYTHON=str(self.root / "no" / "venv"),
            RUNTIME_SYSTEM_PYTHON=str(self.root / "no" / "python3"),
        ))
        document = json.loads(completed.stdout)
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(document["failure"], "NO_INTERPRETER")
        self.assertEqual([row["state"] for row in document["candidates"]], ["ABSENT", "ABSENT"])
        self.assertEqual(accepted_executable(document), "")
        forged = pinned_image_selection("/tmp/not-the-worker-python")
        forged["selected"] = "/tmp/not-the-worker-python"
        forged["candidates"][1]["state"] = "ABSENT"
        self.assertEqual(accepted_executable(forged), "")

    def test_unreadable_probe_output_is_refused(self) -> None:
        self.assertEqual(accepted_executable(parse_runtime_selection("not json")), "")
        self.assertEqual(parse_runtime_selection("not json")["failure"], "SELECTION_UNREADABLE")


class ProviderTests(unittest.TestCase):
    def _provider(self, ssh) -> VastImageProvider:
        provider = VastImageProvider(
            api_key="fixture-key",
            transport=lambda *args: (200, b'{"instances": []}'),
            sleep=lambda _seconds: None,
            ssh_run=ssh,
            runtime=Path(tempfile.mkdtemp()),
        )
        provider._host = "worker"
        provider._port = "22"
        return provider

    def test_import_and_inference_use_the_selected_interpreter(self) -> None:
        venv = "/workspace/phase2d-venv/bin/python"
        selected = {
            "selected": venv,
            "version": "3.12.3",
            "ready_marker": False,
            "failure": "",
            "candidates": [{
                "executable": venv,
                "resolved": venv,
                "state": "OK",
                "version": "3.12.3",
                "imports": {name: True for name in REQUIRED_MODULES},
                "pipeline": True,
                "stderr": "",
            }],
        }
        seen: list[str] = []

        def ssh(_host, _port, command, _timeout):
            seen.append(command)
            if "RUNTIME_PYTHON_SELECT" in command:
                return 0, json.dumps(selected) + "\n", ""
            if "RUNTIME_IMPORT_PROBE" in command:
                return 0, "QwenImagePipeline\n", ""
            if "qwen_remote.py generate" in command:
                return 0, "PNG_READY 12\n", ""
            if command.startswith("base64 "):
                return 0, base64.b64encode(PNG).decode("ascii") + "\n", ""
            return 0, "", ""

        provider = self._provider(ssh)
        provider.runtime_ready = True
        provider.prepare_runtime("1")
        provider.ensure_model("1")
        image = provider.generate(prompt="a terrace", width=1024, height=1024, seed=3)
        self.assertEqual(image[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(provider.python_executable, venv)
        import_commands = [item for item in seen if "RUNTIME_IMPORT_PROBE" in item]
        generate_commands = [item for item in seen if "qwen_remote.py generate" in item]
        self.assertEqual(len(import_commands), 1)
        self.assertEqual(len(generate_commands), 1)
        self.assertIn(f"{venv} -c 'from diffusers import QwenImagePipeline", import_commands[0])
        self.assertIn(f"{venv} -u /workspace/qwen_remote.py generate", generate_commands[0])
        self.assertEqual(generate_commands[0].count(venv), 2)
        self.assertFalse(any("pip install" in item or "python3 -m venv" in item for item in seen))
        direct = _generate_command("a terrace", 1024, 1024, 3, python=venv)
        self.assertIn(f"{venv} -u /workspace/qwen_remote.py generate", direct)

    def test_a_broken_probe_is_saved_and_does_not_install(self) -> None:
        failed = pinned_image_selection()
        failed["selected"] = ""
        failed["failure"] = "DEPENDENCY_MISSING"
        failed["candidates"][1]["state"] = "DEPENDENCY_MISSING"
        failed["candidates"][1]["pipeline"] = False
        failed["candidates"][1]["imports"]["diffusers"] = False
        failed["candidates"][1]["stderr"] = "ImportError: diffusers bearer fixture-key"

        def ssh(_host, _port, command, _timeout):
            if "RUNTIME_PYTHON_SELECT" in command:
                return 1, json.dumps(failed) + "\n", ""
            if "---PYTHON---" in command:
                return 0, (
                    "---PYTHON---\n" + json.dumps(failed) + "\n"
                    "---IMPORT_EXIT---\n\n---IMPORT_ERR---\n\n---REPORT---\n\n"
                    "---STAMP---\n2026-10-10T12:40:00Z\n"
                ), ""
            return 0, "", ""

        provider = self._provider(ssh)
        with self.assertRaises(WorkerStageError) as raised:
            provider.prepare_runtime("1")
        self.assertEqual(raised.exception.code, "RUNTIME_PREPARE_FAILED")
        self.assertEqual(provider.python_executable, "")
        saved = json.loads((provider.runtime / "runtime-evidence.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["model_load_exception"], "DEPENDENCY_MISSING")
        self.assertEqual(saved["failure_timestamp"], "2026-10-10T12:40:00Z")
        self.assertEqual(saved["python_version"], "3.12.3")
        self.assertFalse(saved["dependencies"][1]["imports"]["diffusers"])
        self.assertNotIn("fixture-key", json.dumps(saved))
        self.assertIn("[redacted]", json.dumps(saved))

    def test_model_load_failure_keeps_the_exception_off_the_instance(self) -> None:
        report = {
            "error": "LOAD_FAILED",
            "exception": "OutOfMemoryError",
            "exception_detail": "CUDA out of memory",
        }

        def ssh(_host, _port, command, _timeout):
            if "RUNTIME_PYTHON_SELECT" in command:
                return 0, json.dumps(pinned_image_selection()) + "\n", ""
            if "RUNTIME_IMPORT_PROBE" in command:
                return 0, "QwenImagePipeline\n", ""
            if "qwen_remote.py generate" in command:
                return 1, "LOAD_FAILED\n", "CUDA out of memory"
            if "---PYTHON---" in command:
                return 0, (
                    "---PYTHON---\n" + json.dumps(pinned_image_selection()) + "\n"
                    "---IMPORT_EXIT---\n0\n---IMPORT_ERR---\n\n---REPORT---\n"
                    + json.dumps(report) + "\n---STAMP---\n2026-10-10T12:41:00Z\n"
                ), "Authorization: Bearer fixture-key"
            return 0, "", ""

        provider = self._provider(ssh)
        provider.ensure_model("1")
        with self.assertRaises(WorkerStageError) as raised:
            provider.generate(prompt="a terrace", width=1024, height=1024, seed=3)
        self.assertEqual(raised.exception.code, "MODEL_LOAD_FAILED")
        saved = json.loads((provider.runtime / "runtime-evidence.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["python_executable"], "python3")
        self.assertEqual(saved["import_exit_code"], "0")
        self.assertEqual(saved["model_load_exception"], "OutOfMemoryError")
        self.assertEqual(saved["failure_timestamp"], "2026-10-10T12:41:00Z")
        self.assertNotIn("fixture-key", saved["stderr"])
        self.assertIn("OutOfMemoryError", saved["report"])

    def test_hung_evidence_still_raises_and_the_gpu_can_be_deleted(self) -> None:
        self.assertEqual(CAPTURE_TIMEOUT_SECONDS, 8)
        self.assertIn("MODEL_LOAD_FAILED", _DESTROY_ON)
        self.assertIn("RUNTIME_PREPARE_FAILED", _DESTROY_ON)
        deleted = {"done": False}
        calls: list[str] = []

        def transport(method, url, body, headers):
            if method == "DELETE":
                deleted["done"] = True
                calls.append(url)
                return 200, b"{}"
            if deleted["done"]:
                return 200, b'{"instances": []}'
            return 200, json.dumps({"instances": [{
                "id": 9,
                "label": "media-engine:live8-run",
                "machine_id": 142658,
            }]}).encode()

        def ssh(_host, _port, command, _timeout):
            if "RUNTIME_PYTHON_SELECT" in command:
                return 0, json.dumps(pinned_image_selection()) + "\n", ""
            if "RUNTIME_IMPORT_PROBE" in command:
                return 1, "", "ImportError: QwenImagePipeline"
            if "---PYTHON---" in command:
                time.sleep(30)
                return 0, "late", ""
            return 0, "", ""

        provider = VastImageProvider(
            api_key="fixture-key",
            transport=transport,
            sleep=lambda _seconds: None,
            ssh_run=ssh,
            runtime=Path(tempfile.mkdtemp()),
        )
        provider._host = "worker"
        provider._port = "22"
        provider.machine_id = "142658"
        started = time.monotonic()
        with self.assertRaises(WorkerStageError) as raised:
            provider.ensure_model("9")
        elapsed = time.monotonic() - started
        self.assertEqual(raised.exception.code, "MODEL_LOAD_FAILED")
        self.assertGreaterEqual(elapsed, 7.5)
        self.assertLess(elapsed, 9.0)
        saved = json.loads((provider.runtime / "runtime-evidence.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["capture_error"], "CAPTURE_TIMEOUT")
        self.assertEqual(saved["python_executable"], "python3")
        self.assertEqual(saved["import_exit_code"], "1")
        provider.terminate("9")
        self.assertTrue(deleted["done"])
        self.assertTrue(any(item.endswith("/9/") for item in calls))
        self.assertIn("9", provider._gone)

    def test_capture_timeout_returns_without_raising(self) -> None:
        root = Path(tempfile.mkdtemp())

        def fetch():
            time.sleep(2)
            return "---STAMP---\nlate\n", ""

        started = time.monotonic()
        record = preserve_runtime_evidence(
            destination=root / "runtime-evidence.json",
            fetch=fetch,
            python_executable="python3",
            model_load_exception="IMPORT_FAILED",
            timeout_seconds=0.05,
        )
        self.assertLess(time.monotonic() - started, 1)
        self.assertEqual(record["capture_error"], "CAPTURE_TIMEOUT")
        self.assertEqual(record["python_executable"], "python3")
        parsed = parse_runtime_bundle("---IMPORT_EXIT---\n1\n---STAMP---\n2026-10-10T00:00:00Z\n")
        self.assertEqual(parsed["import_exit"], "1")
        self.assertEqual(parsed["stamp"], "2026-10-10T00:00:00Z")

    def test_import_command_rejects_a_blank_interpreter(self) -> None:
        with self.assertRaises(ValueError):
            import_probe_command("")
        provider = self._provider(lambda *args: (_ for _ in ()).throw(AssertionError("ssh")))
        with self.assertRaises(WorkerStageError) as raised:
            provider.generate(prompt="a terrace", width=1024, height=1024, seed=3)
        self.assertEqual(raised.exception.code, "RUNTIME_PREPARE_FAILED")
