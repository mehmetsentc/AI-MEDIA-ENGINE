"""The remote half of the GPU media path.

These run no GPU and rent nothing. What they pin is the behaviour that is
expensive to get wrong on a machine billed by the minute: weights are proven
before a model is loaded, and the bytes that come back are the bytes the
receipt names.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from media_engine.models import media_remote
from media_engine.models.media_remote import RemoteJobError, run_job, verify_weights


def _stage(directory: Path, files: dict[str, bytes], *, model_id="m/x", revision="rev1") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, body in files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        entries.append({
            "path": name,
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        })
    (directory / "manifest.json").write_text(
        json.dumps({"model_id": model_id, "revision": revision, "files": entries}),
        encoding="utf-8",
    )


class _Voice:
    """Stands in for a loaded TTS checkpoint."""

    def __init__(self) -> None:
        self.calls = 0

    def generate_custom_voice(self, text, language, speaker):
        self.calls += 1
        return [0.0, 0.25, -0.25, 0.5] * 240, 24000


class VerifyWeightsTests(unittest.TestCase):
    def test_a_complete_tree_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "m"
            _stage(root, {"config.json": b"{}", "weights.bin": b"0123456789"})
            report = verify_weights(root)
            self.assertTrue(report["complete"])
            self.assertEqual(report["files_present"], 2)

    def test_a_truncated_file_is_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "m"
            _stage(root, {"weights.bin": b"0123456789"})
            (root / "weights.bin").write_bytes(b"01234")
            report = verify_weights(root)
            self.assertFalse(report["complete"])
            self.assertEqual(report["corrupt"], ["weights.bin"])

    def test_a_substituted_file_of_the_same_size_is_caught(self):
        """The case a size check alone would pass: different weights, same bytes long."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "m"
            _stage(root, {"weights.bin": b"0123456789"})
            (root / "weights.bin").write_bytes(b"9876543210")
            report = verify_weights(root)
            self.assertFalse(report["complete"])
            self.assertEqual(report["corrupt"], ["weights.bin"])

    def test_a_missing_file_is_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "m"
            _stage(root, {"a.bin": b"aa", "b.bin": b"bb"})
            (root / "b.bin").unlink()
            report = verify_weights(root)
            self.assertEqual(report["missing"], ["b.bin"])

    def test_no_manifest_is_not_treated_as_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "m"
            root.mkdir(parents=True)
            (root / "weights.bin").write_bytes(b"anything")
            report = verify_weights(root)
            self.assertEqual(report["status"], "NO_MANIFEST")
            self.assertFalse(report["complete"])

    def test_a_manifest_path_escaping_the_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "m"
            root.mkdir(parents=True)
            (root / "manifest.json").write_text(json.dumps({
                "model_id": "m/x", "revision": "r",
                "files": [{"path": "../escape.bin", "bytes": 2, "sha256": "x"}],
            }), encoding="utf-8")
            report = verify_weights(root)
            self.assertFalse(report["complete"])
            self.assertEqual(report["corrupt"], ["../escape.bin"])


    def test_a_manifest_path_that_is_absolute_is_refused(self):
        """The guard this pins was briefly inverted: it tested the joined path,
        which is always absolute, and so condemned every file as corrupt."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "m"
            root.mkdir(parents=True)
            (root / "manifest.json").write_text(json.dumps({
                "model_id": "m/x", "revision": "r",
                "files": [{"path": "/etc/passwd", "bytes": 1, "sha256": "x"}],
            }), encoding="utf-8")
            report = verify_weights(root)
            self.assertFalse(report["complete"])
            self.assertEqual(report["corrupt"], ["/etc/passwd"])


class RunJobTests(unittest.TestCase):
    def _job(self, **over):
        job = {
            "id": "job1",
            "capability": "audio.tts",
            "model_id": "m/x",
            "revision": "rev1",
            "payload": {"text": "merhaba", "language": "English", "voice": "Vivian"},
        }
        job.update(over)
        return job

    def test_bad_weights_stop_the_run_before_the_model_is_loaded(self):
        """A GPU is being paid for. Loading first and failing later wastes it."""
        loaded = []

        def loader(directory):
            loaded.append(directory)
            return _Voice()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "weights"
            _stage(root / "m/x", {"w.bin": b"0123456789"})
            (root / "m/x/w.bin").write_bytes(b"9876543210")
            with self.assertRaises(RemoteJobError) as caught:
                run_job(
                    self._job(), root=root, out_dir=Path(tmp) / "out",
                    loaders={"m/x": loader},
                )
            self.assertEqual(caught.exception.code, "WEIGHTS_INCOMPLETE")
            self.assertEqual(loaded, [], "the loader must not run on unverified weights")

    def test_a_run_writes_the_bytes_and_a_receipt_that_matches_them(self):
        voice = _Voice()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "weights"
            out = Path(tmp) / "out"
            _stage(root / "m/x", {"w.bin": b"0123456789"})
            receipt = run_job(
                self._job(), root=root, out_dir=out, loaders={"m/x": lambda d: voice},
            )
            produced = out / receipt["file"]
            self.assertTrue(produced.is_file())
            data = produced.read_bytes()
            self.assertEqual(receipt["bytes"], len(data))
            self.assertEqual(receipt["sha256"], hashlib.sha256(data).hexdigest())
            self.assertTrue(data.startswith(b"RIFF"))
            self.assertEqual(voice.calls, 1)
            side = json.loads((out / "job1.json").read_text(encoding="utf-8"))
            self.assertEqual(side, receipt)

    def test_a_revision_the_caller_did_not_authorize_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "weights"
            _stage(root / "m/x", {"w.bin": b"0123456789"}, revision="rev1")
            with self.assertRaises(RemoteJobError) as caught:
                run_job(
                    self._job(revision="rev2"), root=root, out_dir=Path(tmp) / "out",
                    loaders={"m/x": lambda d: _Voice()},
                )
            self.assertEqual(caught.exception.code, "REVISION_MISMATCH")

    def test_an_unknown_capability_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RemoteJobError) as caught:
                run_job(
                    self._job(capability="audio.sing"), root=Path(tmp),
                    out_dir=Path(tmp) / "out", loaders={},
                )
            self.assertEqual(caught.exception.code, "CAPABILITY_UNKNOWN")

    def test_a_model_without_a_loader_is_refused_after_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "weights"
            _stage(root / "m/x", {"w.bin": b"0123456789"})
            with self.assertRaises(RemoteJobError) as caught:
                run_job(self._job(), root=root, out_dir=Path(tmp) / "out", loaders={})
            self.assertEqual(caught.exception.code, "MODEL_UNSUPPORTED")

    def test_an_input_path_escaping_the_inbox_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "weights"
            _stage(root / "m/x", {"w.bin": b"0123456789"})
            with self.assertRaises(RemoteJobError) as caught:
                run_job(
                    self._job(capability="audio.transcribe", input="../../etc/passwd"),
                    root=root, out_dir=Path(tmp) / "out", in_dir=Path(tmp) / "in",
                    loaders={"m/x": lambda d: _Voice()},
                )
            self.assertEqual(caught.exception.code, "INPUT_PATH_INVALID")

    def test_a_capability_that_needs_input_refuses_without_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "weights"
            _stage(root / "m/x", {"w.bin": b"0123456789"})
            with self.assertRaises(RemoteJobError) as caught:
                run_job(
                    self._job(capability="audio.transcribe"), root=root,
                    out_dir=Path(tmp) / "out", loaders={"m/x": lambda d: _Voice()},
                )
            self.assertEqual(caught.exception.code, "INPUT_REQUIRED")


class DriftTests(unittest.TestCase):
    def test_the_remote_runner_covers_exactly_the_local_capabilities(self):
        """A capability added locally and forgotten here would silently have no GPU path."""
        from media_engine.models.runner import _MEDIA

        self.assertEqual(set(media_remote.EXTENSIONS), set(_MEDIA))

    def test_each_capability_writes_the_extension_the_local_path_uses(self):
        from media_engine.models.runner import _MEDIA

        for capability, (_kind, ext, _scene, _mime) in _MEDIA.items():
            self.assertEqual(media_remote.EXTENSIONS[capability], ext, capability)

    def test_every_input_capability_is_a_known_capability(self):
        for capability in media_remote.INPUTS:
            self.assertIn(capability, media_remote.EXTENSIONS)


if __name__ == "__main__":
    unittest.main()


class StageFromR2Tests(unittest.TestCase):
    def test_missing_r2_settings_refuse_before_any_transfer(self):
        from media_engine.models.media_remote import stage_from_r2

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RemoteJobError) as caught:
                stage_from_r2("m/x", "", Path(tmp), env={})
            self.assertEqual(caught.exception.code, "R2_NOT_CONFIGURED")

    def test_a_revision_other_than_the_pinned_one_is_refused(self):
        """Unauthorized weights must not land on the machine at all."""
        from media_engine.models.media_remote import stage_from_r2
        from media_engine.models.speech_stage import all_targets

        pinned = all_targets()[0]
        env = {
            "R2_ENDPOINT": "https://example.invalid", "R2_BUCKET": "b",
            "R2_ACCESS_KEY_ID": "k", "R2_SECRET_ACCESS_KEY": "s",
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RemoteJobError) as caught:
                stage_from_r2(pinned["model_id"], "0" * 40, Path(tmp), env=env)
            self.assertEqual(caught.exception.code, "REVISION_MISMATCH")

    def test_an_unknown_model_is_refused(self):
        from media_engine.models.media_remote import stage_from_r2

        env = {
            "R2_ENDPOINT": "https://example.invalid", "R2_BUCKET": "b",
            "R2_ACCESS_KEY_ID": "k", "R2_SECRET_ACCESS_KEY": "s",
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RemoteJobError) as caught:
                stage_from_r2("nobody/nothing", "", Path(tmp), env=env)
            self.assertEqual(caught.exception.code, "MODEL_UNKNOWN")
