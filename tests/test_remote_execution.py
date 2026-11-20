"""The production media run. Nothing here rents: the switch stays off by default.

The invariants these pin are the ones that cost money when they break: the
refusal happens before an instance exists, the checkpoint that runs is the one
the authorization names, and the machine is deleted whatever else goes wrong.
"""
from __future__ import annotations

import decimal
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from media_engine.models.remote_execution import (
    RENTAL_ENV,
    authorized_models,
    rental_approved,
    run_media_job,
)
from media_engine.models.speech_stage import SpeechPrepError
from media_engine.providers.base import WorkerStageError
from media_engine.providers.execution_authorization import issue_authorization
from media_engine.models.speech_image import SPEECH_REPOSITORY
from media_engine.providers.fake import FakeGPUProvider

SHA = "1b07d08b467b8def7943ab57771128f734e3283d"
DIGEST = "sha256:" + "ab" * 32
WHISPER = "Systran/faster-whisper-large-v3"
TTS = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
WHISPER_REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"
TTS_REVISION = "0c0e3051f131929182e2c023b9537f8b1c68adfe"
APPROVED = {RENTAL_ENV: "approved"}
WAV = b"RIFF....WAVEfakebytes"


def _verified_ref(digest: str = DIGEST) -> dict:
    return {
        "repository": SPEECH_REPOSITORY,
        "tag": "sha-" + SHA,
        "digest": digest,
        "pinned_reference": SPEECH_REPOSITORY + "@" + digest,
        "previous_digest": "",
        "cuda_gpu_support": False,
    }


class _Clock:
    def __init__(self) -> None:
        self.wall = 1_700_000_000.0
        self.mono = 1_000.0

    def now(self) -> float:
        return self.mono


def _speech(**overrides):
    fields = dict(
        authorization_id="speech-auth-live",
        execution_id="speech-run-live",
        code_sha=SHA,
        allowed_machine_ids=(501,),
        allowed_offer_ids=(1001,),
        allowed_pairs=((501, 1001),),
        selection_policy="just_in_time",
        max_hourly_usd="0.74",
        max_total_usd="1.50",
        max_instances=1,
        max_creates=1,
        max_gpu_instances=1,
        max_inference_attempts=2,
        max_image_count=0,
        max_model_stages=2,
        max_model_loads=2,
        max_disk_gb=140,
        model=WHISPER,
        revision=WHISPER_REVISION,
        speech_models=((WHISPER, WHISPER_REVISION), (TTS, TTS_REVISION)),
        speech_session=True,
        worker_digest=DIGEST,
        user_approval={"approval_id": "speech-approval-live", "approved_at": 1_500},
        provider_billing_cap=False,
        issued_at=1_000,
        expires_at=20_000,
    )
    fields.update(overrides)
    return issue_authorization(**fields)


class _Worker:
    """Stands in for one rented machine reached over ssh."""

    def __init__(self, *, payload: bytes = WAV, fail_run: bool = False, short: bool = False) -> None:
        self.payload = payload
        self.fail_run = fail_run
        self.short = short
        self.commands: list[str] = []
        self.staged = 0

    def stage_media(self, host, port):
        self.staged += 1
        return {"sha256": "c" * 64, "bytes": 1, "files": 1}

    def run(self, host, port, command, timeout):
        self.commands.append(command)
        if self.fail_run:
            raise WorkerStageError("WORKER_TIMEOUT")
        receipt = {
            "id": "job1",
            "capability": "audio.tts",
            "model_id": TTS,
            "revision": TTS_REVISION,
            "file": "job1.wav",
            "bytes": len(self.payload),
            "sha256": hashlib.sha256(self.payload).hexdigest(),
        }
        return 0, "loading\nJOB_DONE " + json.dumps(receipt, sort_keys=True), ""

    def fetch(self, host, port, remote, local):
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(self.payload[:4] if self.short else self.payload)


class RemoteExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.clock = _Clock()
        self.provider = FakeGPUProvider(self.clock.now, hourly_price=decimal.Decimal("0.74"))
        self.staged: list[str] = []

    def _stage_weights(self, worker, host, port, model_id):
        self.staged.append(model_id)
        return {"status": "COMPLETE", "model_id": model_id}

    def _run(self, *, worker=None, env=APPROVED, job=None, authorization=None):
        return run_media_job(
            authorization or _speech(),
            job=job or {
                "id": "job1", "capability": "audio.tts",
                "model_id": TTS, "revision": TTS_REVISION,
                "payload": {"text": "merhaba", "language": "Turkish", "voice": "Vivian"},
            },
            provider=self.provider,
            worker=worker or _Worker(),
            platform=None,
            user_id="nahaber",
            now=5_000,
            code_sha=SHA,
            recorded_digest=DIGEST,
            claim_path=self.root / "claims",
            machine_id=501,
            offer_id=1001,
            instance_id="i-1",
            clock=self.clock,
            host="1.2.3.4",
            port="22",
            artifact_path=self.root / "out" / "job1.wav",
            stage_weights=self._stage_weights,
            image_ref=_verified_ref(),
            env=env,
        )

    # --- the switch -------------------------------------------------------

    def test_without_approval_nothing_is_rented(self):
        with self.assertRaises(SpeechPrepError) as caught:
            self._run(env={})
        self.assertEqual(caught.exception.code, "GPU_RENTAL_REFUSED")
        self.assertEqual(self.provider.create_calls, 0, "an instance was created anyway")

    def test_a_truthy_value_is_not_approval(self):
        """'1' and 'true' are what a stale shell export looks like."""
        for value in ("1", "true", "yes", "on", "APPROVE", " "):
            self.assertFalse(rental_approved({RENTAL_ENV: value}), value)
        self.assertTrue(rental_approved({RENTAL_ENV: "approved"}))
        self.assertTrue(rental_approved({RENTAL_ENV: " Approved "}))

    def test_an_absent_variable_is_refusal(self):
        self.assertFalse(rental_approved({}))

    # --- the pins ---------------------------------------------------------

    def test_a_model_the_authorization_does_not_name_never_reaches_a_machine(self):
        job = {
            "id": "job1", "capability": "audio.tts",
            "model_id": "someone/else-tts", "revision": TTS_REVISION,
            "payload": {"text": "x", "language": "Turkish", "voice": "Vivian"},
        }
        with self.assertRaises(SpeechPrepError) as caught:
            self._run(job=job)
        self.assertEqual(caught.exception.code, "MODEL_NOT_PINNED")
        self.assertEqual(self.provider.create_calls, 0)

    def test_the_right_model_at_the_wrong_revision_is_refused(self):
        """Matching on the model name alone would let this through."""
        job = {
            "id": "job1", "capability": "audio.tts",
            "model_id": TTS, "revision": "0" * 40,
            "payload": {"text": "x", "language": "Turkish", "voice": "Vivian"},
        }
        with self.assertRaises(SpeechPrepError) as caught:
            self._run(job=job)
        self.assertEqual(caught.exception.code, "MODEL_NOT_PINNED")
        self.assertEqual(self.provider.create_calls, 0)

    def test_both_pinned_models_are_authorized(self):
        pairs = authorized_models(_speech())
        self.assertIn((WHISPER, WHISPER_REVISION), pairs)
        self.assertIn((TTS, TTS_REVISION), pairs)

    def test_a_machine_outside_the_authorized_pairs_is_refused(self):
        with self.assertRaises(WorkerStageError):
            run_media_job(
                _speech(), job={
                    "id": "job1", "capability": "audio.tts", "model_id": TTS,
                    "revision": TTS_REVISION, "payload": {"text": "x"},
                },
                provider=self.provider, worker=_Worker(), platform=None, user_id="nahaber",
                now=5_000, code_sha=SHA, recorded_digest=DIGEST,
                claim_path=self.root / "claims", machine_id=999, offer_id=1001,
                instance_id="i-1", clock=self.clock, host="1.2.3.4", port="22",
                artifact_path=self.root / "out" / "a.wav",
                stage_weights=self._stage_weights, image_ref=_verified_ref(),
                env=APPROVED,
            )
        self.assertEqual(self.provider.create_calls, 0)

    # --- the run ----------------------------------------------------------

    def test_an_approved_run_brings_the_artifact_home_and_deletes_the_machine(self):
        worker = _Worker()
        report = self._run(worker=worker)
        self.assertEqual(report["status"], "COMPLETED")
        self.assertEqual(report["bytes"], len(WAV))
        self.assertEqual(report["sha256"], hashlib.sha256(WAV).hexdigest())
        self.assertTrue(report["deleted"])
        self.assertFalse(report["rehearsal"])
        self.assertEqual(self.provider.create_calls, 1)
        self.assertEqual(worker.staged, 1)
        self.assertEqual(self.staged, [TTS])
        self.assertEqual((self.root / "out" / "job1.wav").read_bytes(), WAV)

    def test_the_command_sent_to_the_worker_names_the_pinned_model(self):
        worker = _Worker()
        self._run(worker=worker)
        self.assertEqual(len(worker.commands), 1)
        self.assertIn(TTS, worker.commands[0])
        self.assertIn("media_engine.models.media_remote", worker.commands[0])

    def test_a_failed_job_still_deletes_the_machine(self):
        """The most expensive bug available here: raise after create, bill forever."""
        worker = _Worker(fail_run=True)
        with self.assertRaises(WorkerStageError):
            self._run(worker=worker)
        self.assertEqual(self.provider.create_calls, 1)
        self.assertEqual(self.provider.terminate_calls, 1)

    def test_a_truncated_artifact_fails_and_still_deletes_the_machine(self):
        worker = _Worker(short=True)
        with self.assertRaises(Exception) as caught:
            self._run(worker=worker)
        self.assertEqual(getattr(caught.exception, "code", ""), "ARTIFACT_TRUNCATED")
        self.assertEqual(self.provider.terminate_calls, 1)

    def test_a_job_without_a_capability_is_refused_before_create(self):
        with self.assertRaises(SpeechPrepError) as caught:
            self._run(job={"id": "job1", "model_id": TTS, "revision": TTS_REVISION})
        self.assertEqual(caught.exception.code, "JOB_INCOMPLETE")
        self.assertEqual(self.provider.create_calls, 0)


if __name__ == "__main__":
    unittest.main()
