"""Binding the real local loaders to a controller. No weights, no downloads."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from media_engine.models import local_bind
from media_engine.models.local_bind import (
    Deferred,
    LazyRuntimes,
    attach_local_runtimes,
    capabilities,
    weights_dir_for,
)
from media_engine.models.registry import get_model
from media_engine.models.runner import bound_runtime


class _Controller:
    """Only what the binding touches."""


class _Model:
    def __init__(self) -> None:
        self.calls = 0

    def generate_custom_voice(self, **_kwargs):
        self.calls += 1
        return [[0.0, 0.1]], 24000

    def __call__(self, **_kwargs):
        self.calls += 1
        return b"called"


class WeightsDirTests(unittest.TestCase):
    def test_each_capability_resolves_to_its_primary_model_directory(self) -> None:
        root = Path("/models")
        self.assertEqual(
            weights_dir_for(root, "audio.tts"),
            root / "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        )
        self.assertEqual(
            weights_dir_for(root, "audio.voice_clone"),
            root / "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        )
        # voice_clone and tts are different models, so they cannot share a directory.
        self.assertNotEqual(
            weights_dir_for(root, "audio.tts"), weights_dir_for(root, "audio.voice_clone"),
        )

    def test_every_loader_names_a_model_the_registry_knows(self) -> None:
        from media_engine.models.schema import KINDS
        for model_id in local_bind.MODEL_LOADERS:
            spec = get_model(model_id)
            self.assertIn(spec.capability, KINDS)
        for capability in capabilities():
            self.assertIn(capability, KINDS)

    def test_language_picks_the_model_that_speaks_it(self) -> None:
        root = Path("/models")
        turkish = weights_dir_for(root, "audio.tts", "Turkish")
        english = weights_dir_for(root, "audio.tts", "English")
        self.assertEqual(turkish, root / "freyavoice/Freya-TTS")
        self.assertEqual(english, root / "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
        self.assertNotEqual(turkish, english)


class AttachTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name

    def test_nothing_is_attached_unless_an_operator_asks(self) -> None:
        controller = _Controller()
        self.assertFalse(attach_local_runtimes(controller, {}))
        self.assertFalse(hasattr(controller, "multimodal_runtimes"))
        self.assertFalse(attach_local_runtimes(controller, {
            "MEDIA_ENGINE_LOCAL_MODELS": "0", "MEDIA_ENGINE_WEIGHTS_DIR": self.root,
        }))
        self.assertFalse(hasattr(controller, "multimodal_runtimes"))

    def test_a_missing_or_wrong_weights_directory_is_refused_loudly(self) -> None:
        controller = _Controller()
        with self.assertRaises(ValueError):
            attach_local_runtimes(controller, {"MEDIA_ENGINE_LOCAL_MODELS": "1"})
        with self.assertRaises(ValueError):
            attach_local_runtimes(controller, {
                "MEDIA_ENGINE_LOCAL_MODELS": "1",
                "MEDIA_ENGINE_WEIGHTS_DIR": str(Path(self.root) / "absent"),
            })
        self.assertFalse(hasattr(controller, "multimodal_runtimes"))

    def test_attaching_sets_the_table_and_an_optional_timeout(self) -> None:
        controller = _Controller()
        self.assertTrue(attach_local_runtimes(controller, {
            "MEDIA_ENGINE_LOCAL_MODELS": "yes", "MEDIA_ENGINE_WEIGHTS_DIR": self.root,
        }))
        self.assertIsInstance(controller.multimodal_runtimes, LazyRuntimes)
        self.assertFalse(hasattr(controller, "multimodal_timeout"))
        self.assertTrue(attach_local_runtimes(controller, {
            "MEDIA_ENGINE_LOCAL_MODELS": "1", "MEDIA_ENGINE_WEIGHTS_DIR": self.root,
            "MEDIA_ENGINE_MULTIMODAL_TIMEOUT_SECONDS": "600",
        }))
        self.assertEqual(controller.multimodal_timeout, 600.0)
        for bad in ("0", "-5", "soon"):
            with self.assertRaises(ValueError):
                attach_local_runtimes(_Controller(), {
                    "MEDIA_ENGINE_LOCAL_MODELS": "1", "MEDIA_ENGINE_WEIGHTS_DIR": self.root,
                    "MEDIA_ENGINE_MULTIMODAL_TIMEOUT_SECONDS": bad,
                })


class LazyRuntimesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.table = LazyRuntimes(Path(self.tmp.name))
        self.loads: list[str] = []
        self._real = dict(local_bind.MODEL_LOADERS)

        def loader(model_id):
            def load(directory):
                self.loads.append(model_id)
                return _Model()
            return load

        local_bind.MODEL_LOADERS = {name: loader(name) for name in self._real}
        self.addCleanup(setattr, local_bind, "MODEL_LOADERS", self._real)

    def test_an_empty_table_is_still_a_table(self) -> None:
        """bound_runtime reads the table as `getattr(...) or {}`.

        An ordinary empty dict is falsy and would be thrown away there, so every
        capability would report itself unavailable while a perfectly good lazy
        table was attached.
        """
        self.assertEqual(len(self.table), 0)
        self.assertTrue(self.table)
        controller = _Controller()
        controller.multimodal_runtimes = self.table
        self.assertIsNotNone(bound_runtime(controller, "audio.tts"))

    def test_asking_for_a_capability_does_not_read_the_weights(self) -> None:
        runtime = self.table.get("audio.tts")
        self.assertIsInstance(runtime, Deferred)
        self.assertFalse(runtime.loaded())
        self.assertEqual(self.loads, [])
        # Submission checks availability before charging anyone, so the
        # gigabytes must wait for a job that is actually going to run.
        runtime.generate_custom_voice(text="hi")
        self.assertTrue(runtime.loaded())
        self.assertEqual(self.loads, ["Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"])

    def test_a_capability_without_a_loader_returns_the_default(self) -> None:
        self.assertIsNone(self.table.get("image.generate"))
        self.assertEqual(self.table.get("nonsense", "fallback"), "fallback")

    def test_an_injected_runtime_wins_and_is_returned_untouched(self) -> None:
        injected = _Model()
        self.table["audio.tts"] = injected
        self.assertIs(self.table.get("audio.tts"), injected)
        self.assertEqual(self.loads, [])

    def test_only_one_model_stays_resident(self) -> None:
        first = self.table.get("audio.tts")
        first.generate_custom_voice(text="hi")
        self.assertEqual(self.table.resident, "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
        second = self.table.get("music.generate")
        self.assertEqual(self.table.resident, "ACE-Step/ACE-Step-v1-3.5B")
        self.assertNotIn("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", self.table)
        self.assertIsNot(first, second)
        # Asking again rebuilds it rather than handing back a dropped model.
        third = self.table.get("audio.tts")
        self.assertIsNot(third, first)
        self.assertNotIn("ACE-Step/ACE-Step-v1-3.5B", self.table)

    def test_two_models_of_one_capability_do_not_share_a_runtime(self) -> None:
        """Turkish and English speech are different checkpoints.

        Handing a Turkish job the English runtime would synthesize the wrong
        language with the wrong voice, so the table answers by model.
        """
        english = self.table.for_model("audio.tts", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
        turkish = self.table.for_model("audio.tts", "freyavoice/Freya-TTS")
        self.assertIsNot(english, turkish)
        self.assertEqual(turkish.capability, "freyavoice/Freya-TTS")
        self.assertEqual(self.table.resident, "freyavoice/Freya-TTS")

    def test_an_unknown_model_has_no_runtime(self) -> None:
        self.assertIsNone(self.table.for_model("audio.tts", "someone/else"))

    def test_the_same_capability_is_not_reloaded(self) -> None:
        runtime = self.table.get("audio.tts")
        runtime.generate_custom_voice(text="hi")
        again = self.table.get("audio.tts")
        self.assertIs(again, runtime)
        again.generate_custom_voice(text="again")
        self.assertEqual(self.loads, ["Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"])

    def test_a_runtime_that_is_called_rather_than_addressed_still_resolves(self) -> None:
        runtime = self.table.get("video.text_to_video")
        self.assertFalse(runtime.loaded())
        self.assertEqual(runtime(prompt="a"), b"called")
        self.assertTrue(runtime.loaded())
