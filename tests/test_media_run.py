"""Host side of the remote media run.

Nothing here opens a connection. What it pins is that a prompt cannot become a
shell command, that the archive digest means something, and that bytes fetched
off a worker are checked against what the worker said it produced.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

from media_engine.providers.media_run import (
    RemoteRunError,
    install_command,
    job_command,
    package_archive,
    parse_result,
    remote_artifact_path,
    verify_artifact,
)


def _job(**over):
    job = {
        "id": "job1",
        "capability": "audio.tts",
        "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "revision": "abc",
        "payload": {"text": "merhaba", "language": "Turkish", "voice": "Vivian"},
    }
    job.update(over)
    return job


class ArchiveTests(unittest.TestCase):
    def test_the_real_package_packs_and_contains_the_remote_runner(self):
        source = Path(__file__).resolve().parents[1] / "src" / "media_engine"
        with tempfile.TemporaryDirectory() as tmp:
            report = package_archive(source, Path(tmp) / "pkg.tar")
            self.assertGreater(report["files"], 50)
            with tarfile.open(report["path"]) as archive:
                names = archive.getnames()
            self.assertIn("media_engine/models/media_remote.py", names)
            self.assertIn("media_engine/models/invoke.py", names)
            self.assertFalse([n for n in names if "__pycache__" in n])

    def test_two_packs_of_one_tree_have_one_digest(self):
        """Otherwise the digest cannot answer 'does the worker have this code'."""
        source = Path(__file__).resolve().parents[1] / "src" / "media_engine"
        with tempfile.TemporaryDirectory() as tmp:
            first = package_archive(source, Path(tmp) / "a.tar")
            second = package_archive(source, Path(tmp) / "b.tar")
            self.assertEqual(first["sha256"], second["sha256"])

    def test_a_changed_file_changes_the_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "pkg"
            (source).mkdir()
            (source / "__init__.py").write_text("x = 1\n")
            before = package_archive(source, Path(tmp) / "a.tar")["sha256"]
            (source / "__init__.py").write_text("x = 2\n")
            after = package_archive(source, Path(tmp) / "b.tar")["sha256"]
            self.assertNotEqual(before, after)

    def test_a_directory_that_is_not_a_package_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RemoteRunError) as caught:
                package_archive(Path(tmp), Path(tmp) / "a.tar")
            self.assertEqual(caught.exception.code, "PACKAGE_NOT_FOUND")

    def test_the_archive_unpacks_to_an_importable_package(self):
        source = Path(__file__).resolve().parents[1] / "src" / "media_engine"
        with tempfile.TemporaryDirectory() as tmp:
            report = package_archive(source, Path(tmp) / "pkg.tar")
            root = Path(tmp) / "unpacked"
            root.mkdir()
            with tarfile.open(report["path"]) as archive:
                archive.extractall(root, filter="data")
            proof = subprocess.run(
                ["python3", "-c",
                 "import media_engine.models.media_remote as m; print(len(m.EXTENSIONS))"],
                cwd=root, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(proof.returncode, 0, proof.stderr)
            self.assertEqual(proof.stdout.strip(), "7")


class CommandTests(unittest.TestCase):
    def test_a_prompt_cannot_end_the_command_and_start_another(self):
        """Run the write half in a real shell: string matching proves nothing,
        because the hostile text legitimately appears inside the quoted JSON."""
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "pwned"
            hostile = "merhaba'; touch " + str(marker) + "; echo '"
            command = job_command(
                _job(payload={"text": hostile, "language": "Turkish", "voice": "Vivian"}),
                weights_root="/models", out_dir=str(Path(tmp) / "out"),
            )
            # Everything up to the python invocation: mkdir and the job write.
            write_half = " && ".join(command.split(" && ")[:2]).replace(
                "/workspace/media-jobs", str(Path(tmp) / "jobs"),
            )
            run = subprocess.run(["sh", "-c", write_half], capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertFalse(marker.exists(), "the injected command ran")
            written = json.loads((Path(tmp) / "jobs" / "job1.json").read_text(encoding="utf-8"))
            self.assertEqual(written["payload"]["text"], hostile)

    def test_a_newline_in_a_prompt_does_not_split_the_command(self):
        command = job_command(
            _job(payload={"text": "bir\nyeni satir\n&& whoami", "language": "Turkish", "voice": "Vivian"}),
            weights_root="/models", out_dir="/workspace/out",
        )
        self.assertNotIn("\n", command)

    def test_a_job_id_that_is_not_a_plain_name_is_refused(self):
        for bad in ("../../etc/passwd", "a b", "a;b", "", "a/b"):
            with self.assertRaises(RemoteRunError, msg=bad) as caught:
                job_command(_job(id=bad), weights_root="/m", out_dir="/o")
            self.assertEqual(caught.exception.code, "JOB_ID_INVALID")

    def test_the_command_runs_the_module_offline(self):
        command = job_command(_job(), weights_root="/models", out_dir="/workspace/out")
        self.assertIn("HF_HUB_OFFLINE=1", command)
        self.assertIn("TRANSFORMERS_OFFLINE=1", command)
        self.assertIn("-m media_engine.models.media_remote", command)

    def test_an_inbox_is_passed_only_when_there_is_one(self):
        without = job_command(_job(), weights_root="/m", out_dir="/o")
        withbox = job_command(_job(), weights_root="/m", out_dir="/o", in_dir="/in")
        self.assertTrue(without.rstrip().endswith("/o"))
        self.assertTrue(withbox.rstrip().endswith("/o /in"))

    def test_the_install_command_proves_the_import_before_use(self):
        self.assertIn("REMOTE_READY", install_command())


class ResultTests(unittest.TestCase):
    def test_a_receipt_is_read_from_the_last_line(self):
        receipt = {"id": "job1", "file": "job1.wav", "bytes": 4, "sha256": "ab"}
        out = "loading\nnoise\nJOB_DONE " + json.dumps(receipt, sort_keys=True)
        self.assertEqual(parse_result(out)["file"], "job1.wav")

    def test_a_failure_line_becomes_its_code(self):
        with self.assertRaises(RemoteRunError) as caught:
            parse_result("working\nJOB_FAILED SPEAKER_UNSUPPORTED")
        self.assertEqual(caught.exception.code, "SPEAKER_UNSUPPORTED")

    def test_silence_is_not_success(self):
        with self.assertRaises(RemoteRunError) as caught:
            parse_result("loading weights\nstill loading\n")
        self.assertEqual(caught.exception.code, "NO_RECEIPT")

    def test_a_truncated_fetch_is_caught(self):
        data = b"0123456789"
        receipt = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        with self.assertRaises(RemoteRunError) as caught:
            verify_artifact(data[:5], receipt)
        self.assertEqual(caught.exception.code, "ARTIFACT_TRUNCATED")

    def test_bytes_of_the_right_length_but_the_wrong_content_are_caught(self):
        data = b"0123456789"
        receipt = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        with self.assertRaises(RemoteRunError) as caught:
            verify_artifact(b"9876543210", receipt)
        self.assertEqual(caught.exception.code, "ARTIFACT_CORRUPT")

    def test_matching_bytes_pass(self):
        data = b"RIFFdata"
        verify_artifact(data, {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})

    def test_a_receipt_filename_cannot_walk_out_of_the_out_directory(self):
        for bad in ("../../etc/passwd", "a/b.wav", ""):
            with self.assertRaises(RemoteRunError, msg=bad):
                remote_artifact_path("/workspace/out", {"file": bad})


if __name__ == "__main__":
    unittest.main()
