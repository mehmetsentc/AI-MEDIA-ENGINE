"""Speech authorization and one-instance rehearsal. No rental and no weight download."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from media_engine.engines.image.qwen import MODEL_ID
from media_engine.infra.volume_audit import VOLUME_ID
from media_engine.models.audio import encode_wav
from media_engine.models.cost import CREDIT_RULES
from media_engine.models.speech_execution import classify_probe, run_speech_rehearsal
from media_engine.models.speech_image import SPEECH_REPOSITORY, assess_speech_image_ref, load_image_ref, speech_image_ref_path
from media_engine.models.speech_stage import SpeechPrepError, stage_from_r2, verify_tree
from media_engine.platform.service import Platform
from media_engine.providers.base import WorkerStageError
from media_engine.providers.execution_authorization import (
    accept_record,
    accept_speech_record,
    authorization_from_json,
    issue_authorization,
    make_live8_grant,
)
from media_engine.providers.fake import FakeGPUProvider
from media_engine.providers.live7_contract import REVISION

SHA = "1b07d08b467b8def7943ab57771128f734e3283d"
DIGEST = "sha256:" + "ab" * 32
WHISPER = "Systran/faster-whisper-large-v3"
TTS = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
WHISPER_REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"
TTS_REVISION = "0c0e3051f131929182e2c023b9537f8b1c68adfe"
ROOT = Path(__file__).resolve().parents[1]


class _Clock:
    def __init__(self) -> None:
        self.wall = 1_700_000_000.0
        self.mono = 1_000.0

    def now(self) -> float:
        return self.mono


class _Word:
    def __init__(self) -> None:
        self.word = "known"
        self.start = 0.0
        self.end = 0.4


class _Segment:
    def __init__(self) -> None:
        self.start = 0.0
        self.end = 1.0
        self.text = "known courtyard line"
        self.words = [_Word()]


class _Info:
    language = "en"
    language_probability = 0.99


class _Whisper:
    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, audio, word_timestamps=True):
        self.calls += 1
        if len(audio) < 1:
            raise RuntimeError("empty")
        return [_Segment()], _Info()


class _Voice:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def generate_custom_voice(self, text, language, speaker):
        self.calls += 1
        if self.fail:
            raise RuntimeError("GENERATION_FAILED")
        return encode_wav([0.1, -0.1, 0.1], 24000), 24000


class _Client:
    def __init__(self) -> None:
        self.calls = 0

    def _request(self, *args, **kwargs):
        self.calls += 1
        return 200, {}, b""


def _verified_ref(digest: str) -> dict:
    return {
        "repository": SPEECH_REPOSITORY,
        "tag": "sha-" + SHA,
        "digest": digest,
        "pinned_reference": SPEECH_REPOSITORY + "@" + digest,
        "previous_digest": "",
        "cuda_gpu_support": False,
    }


def _speech(**overrides):
    fields = dict(
        authorization_id="speech-auth-dry",
        execution_id="speech-run-dry",
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
        user_approval={"approval_id": "speech-approval-dry", "approved_at": 1_500},
        provider_billing_cap=False,
        issued_at=1_000,
        expires_at=20_000,
    )
    fields.update(overrides)
    return issue_authorization(**fields)


class SpeechExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.platform = Platform(str(root / "platform.sqlite3"))
        for service, model, credits in CREDIT_RULES.values():
            self.platform.add_price(
                version="mm-" + service, service=service, model=model, quality="standard",
                unit="job", credit_price=credits, internal_cost_units=0,
                effective_at="2026-01-01T00:00:00Z",
            )
        self.platform.grant("nahaber", "topup", 20, "topup_purchase", idempotency_key="speech-grant")
        self.clock = _Clock()
        self.provider = FakeGPUProvider(self.clock.now, hourly_price=__import__("decimal").Decimal("0.74"))
        self.claim = root / "claims"
        self.weights = root / "empty-weights"
        self.weights.mkdir()

    def _run(self, authorization=None, **overrides):
        fields = dict(
            authorization=authorization or _speech(),
            now=5_000,
            code_sha=SHA,
            recorded_digest=DIGEST,
            provider=self.provider,
            platform=self.platform,
            user_id="nahaber",
            whisper_runtime=_Whisper(),
            tts_runtime=_Voice(),
            claim_path=self.claim,
            machine_id=501,
            offer_id=1001,
            instance_id="77",
            clock=self.clock,
            weights_root=self.weights,
            image_ref=_verified_ref(DIGEST),
        )
        fields.update(overrides)
        return run_speech_rehearsal(**fields)

    def test_rehearsal_uses_one_instance_then_deletes_it(self) -> None:
        report = self._run()
        self.assertEqual(report["transcript"], "known courtyard line")
        self.assertTrue(report["wav"].startswith(b"RIFF"))
        self.assertEqual(report["creates"], 1)
        self.assertEqual(report["stages"], 2)
        self.assertEqual(report["loads"], 2)
        self.assertEqual(report["transcriptions"], 1)
        self.assertEqual(report["tts"], 1)
        self.assertEqual(report["weights"], "ABSENT")
        self.assertFalse(report["gpu_verified"])
        self.assertFalse(report["cuda_gpu_support"])
        self.assertFalse(report["provider_billing_cap"])
        self.assertTrue(report["deleted"])
        self.assertEqual(self.provider.create_calls, 1)
        self.assertEqual(self.provider.terminate_calls, 1)
        self.assertIsNone(report["resident"])
        self.assertEqual(self.platform.balance("nahaber"), 18)
        self.assertEqual(report["deletion_reserve_seconds"], 840)
        self.assertEqual(report["heartbeat_interval_seconds"], 20.0)
        self.assertEqual(report["heartbeat_stale_seconds"], 300.0)
        with self.assertRaises(WorkerStageError) as second:
            self._run()
        self.assertEqual(second.exception.code, "STALE_AUTHORIZATION")
        self.assertEqual(self.provider.create_calls, 1)

    def test_deadline_refunds_tts_and_still_deletes_the_one_instance(self) -> None:
        def pause(budget) -> None:
            self.clock.wall = budget.spending_deadline_wall
            self.clock.mono = budget.spending_deadline_monotonic

        report = self._run(pause_after=pause)
        self.assertEqual(report["transcriptions"], 1)
        self.assertEqual(report["tts"], 0)
        self.assertEqual(report["tts_status"], "BUDGET_STOP")
        self.assertEqual(report["stages"], 1)
        self.assertEqual(report["loads"], 1)
        self.assertEqual(self.platform.balance("nahaber"), 19)
        self.assertEqual(self.provider.create_calls, 1)
        self.assertTrue(report["deleted"])

    def test_tts_failure_refunds_and_volume_expiry_and_substitution_are_refused(self) -> None:
        with self.assertRaises(RuntimeError):
            self._run(tts_runtime=_Voice(fail=True))
        self.assertEqual(self.platform.balance("nahaber"), 19)
        self.assertEqual(self.provider.create_calls, 1)
        fresh = FakeGPUProvider(self.clock.now)
        with self.assertRaises(WorkerStageError) as volume:
            self._run(provider=fresh, claim_path=self.claim.parent / "other-claim", volume_id=VOLUME_ID)
        self.assertEqual(volume.exception.code, "VOLUME_FORBIDDEN")
        self.assertEqual(fresh.create_calls, 0)
        with self.assertRaises(WorkerStageError) as expired:
            self._run(
                provider=FakeGPUProvider(self.clock.now),
                claim_path=self.claim.parent / "expired-claim",
                now=20_000,
            )
        self.assertEqual(expired.exception.code, "EXPIRED_AUTHORIZATION")
        with self.assertRaises(WorkerStageError) as substituted:
            _speech(revision="0" * 40)
        self.assertEqual(substituted.exception.code, "MODEL_SUBSTITUTION")
        with self.assertRaises(WorkerStageError) as image_path:
            issue_authorization(
                authorization_id="image-auth-speech",
                execution_id="image-run-speech",
                code_sha=SHA,
                allowed_machine_ids=(501,),
                allowed_offer_ids=(1001,),
                selection_policy="exact_offers",
                max_hourly_usd="0.74",
                max_total_usd="1.50",
                max_instances=1,
                max_inference_attempts=1,
                max_image_count=1,
                max_disk_gb=140,
                model=WHISPER,
                revision=WHISPER_REVISION,
                issued_at=1_000,
                expires_at=20_000,
            )
        self.assertEqual(image_path.exception.code, "MODEL_SUBSTITUTION")
        issued = _speech(authorization_id="speech-auth-bind", execution_id="speech-run-bind")
        with self.assertRaises(WorkerStageError) as bound:
            accept_record(issued, now=5_000, code_sha=SHA)
        self.assertEqual(bound.exception.code, "MODEL_SUBSTITUTION")
        with self.assertRaises(WorkerStageError) as grant:
            make_live8_grant(issued, None)
        self.assertEqual(grant.exception.code, "MODEL_SUBSTITUTION")
        with self.assertRaises(WorkerStageError) as missing_image:
            accept_speech_record(issued, now=5_000, code_sha=SHA, recorded_digest=None)
        self.assertEqual(missing_image.exception.code, "SPEECH_IMAGE_NOT_BUILT")
        restored = authorization_from_json(__import__("json").dumps(issued.to_dict()))
        self.assertEqual(restored, issued)
        self.assertEqual(restored.speech_models[1], (TTS, TTS_REVISION))
        self.assertFalse(restored.provider_billing_cap)

    def test_missing_weights_stay_fail_closed_and_the_image_worker_digest_is_unchanged(self) -> None:
        report = verify_tree(self.weights, WHISPER)
        self.assertEqual(report["status"], "ABSENT")
        self.assertIn("model.bin", report["missing"])
        (self.weights / "REVISION").write_text("0" * 40 + "\n", encoding="utf-8")
        self.assertEqual(verify_tree(self.weights, WHISPER)["status"], "REVISION_MISMATCH")
        client = _Client()
        with self.assertRaises(SpeechPrepError) as refused:
            stage_from_r2(client, WHISPER, self.weights, free_disk_bytes=0, clock=_Clock())
        self.assertEqual(refused.exception.code, "WEIGHTS_NOT_APPROVED")
        self.assertEqual(client.calls, 0)
        image = (ROOT / "deploy" / "worker" / "image_ref.json").read_text(encoding="utf-8")
        self.assertIn("sha256:be2c96a09947b15ff1fd8fb1d2aaa4692dcc9c06083ef1486674ac4fbc4a907c", image)
        dockerfile = (ROOT / "deploy" / "worker" / "Dockerfile").read_text(encoding="utf-8")
        self.assertNotIn("faster-whisper", dockerfile)
        self.assertNotIn("speech_probe", dockerfile)
        self.assertFalse((ROOT / "deploy" / "worker" / "speech" / "image_ref.json").read_text(encoding="utf-8").find("sha256:be2c96a09947") >= 0)
        committed = load_image_ref(speech_image_ref_path())
        with self.assertRaises(WorkerStageError) as unverified:
            assess_speech_image_ref(committed)
        self.assertEqual(unverified.exception.code, "IMAGE_UNVERIFIED")
        with self.assertRaises(WorkerStageError) as mutable:
            assess_speech_image_ref({
                "repository": SPEECH_REPOSITORY,
                "tag": "latest",
                "digest": DIGEST,
                "pinned_reference": SPEECH_REPOSITORY + ":latest",
                "cuda_gpu_support": False,
            })
        self.assertEqual(mutable.exception.code, "IMAGE_MUTABLE")
        with self.assertRaises(WorkerStageError) as substituted:
            assess_speech_image_ref({
                "repository": "ghcr.io/mehmetsentc/ai-media-engine-worker",
                "tag": "sha-" + SHA,
                "digest": DIGEST,
                "pinned_reference": "ghcr.io/mehmetsentc/ai-media-engine-worker@" + DIGEST,
                "cuda_gpu_support": False,
            })
        self.assertEqual(substituted.exception.code, "IMAGE_SUBSTITUTION")
        probed = classify_probe({"imports_ok": True, "torch_cuda_is_available": True, "faster_whisper": "1.2.1"})
        self.assertFalse(probed["cuda_gpu_support"])
        self.assertEqual(probed["reason"], "imports_are_not_gpu_proof")
        image_auth = issue_authorization(
            authorization_id="image-auth-kept",
            execution_id="image-run-kept",
            code_sha=SHA,
            allowed_machine_ids=(501,),
            allowed_offer_ids=(1001,),
            selection_policy="exact_offers",
            max_hourly_usd="0.74",
            max_total_usd="1.50",
            max_instances=1,
            max_inference_attempts=1,
            max_image_count=1,
            max_disk_gb=140,
            model=MODEL_ID,
            revision=REVISION,
            issued_at=1_000,
            expires_at=20_000,
        )
        self.assertFalse(image_auth.speech_session)
        self.assertEqual(image_auth.max_model_stages, 1)
        accept_record(image_auth, now=5_000, code_sha=SHA)

    def test_manual_workflow_does_not_retag_the_image_worker(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "build-speech-worker.yml").read_text(encoding="utf-8")
        trigger = workflow.split("jobs:", 1)[0]
        self.assertIn("workflow_dispatch:", trigger)
        self.assertNotIn("push:", trigger)
        self.assertNotIn("pull_request:", trigger)
        self.assertIn("platforms: linux/amd64", workflow)
        self.assertIn("file: deploy/worker/speech/Dockerfile", workflow)
        tags = [line for line in workflow.splitlines() if line.strip().startswith("tags:")]
        self.assertEqual(tags, [
            "          tags: ghcr.io/${{ github.repository_owner }}/ai-media-engine-speech-worker:sha-${{ github.sha }}",
        ])
        self.assertIn("secrets.GITHUB_TOKEN", workflow)
        self.assertNotIn("echo ${{ secrets.GITHUB_TOKEN }}", workflow)
        self.assertIn("speech-image-ref", workflow)
        issued = _speech(authorization_id="speech-auth-unverified", execution_id="speech-run-unverified")
        with self.assertRaises(WorkerStageError) as unverified:
            accept_speech_record(issued, now=5_000, code_sha=SHA, recorded_digest=DIGEST)
        self.assertEqual(unverified.exception.code, "IMAGE_UNVERIFIED")
        with self.assertRaises(WorkerStageError) as mutable_digest:
            _speech(worker_digest="latest")
        self.assertEqual(mutable_digest.exception.code, "IMAGE_DIGEST")
