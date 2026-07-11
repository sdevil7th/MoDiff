import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.DiffusersAudio.main import Generate  # noqa: E402


class FakeAceStepPipeline:
    device = "cpu"

    def __init__(self):
        self.call_kwargs = None

    def __call__(self, bpm=None, **kwargs):
        self.call_kwargs = {**kwargs, "bpm": bpm}
        return SimpleNamespace(audios=np.zeros((1, 480), dtype=np.float32))


class DiffusersAudioGenerateTests(unittest.TestCase):
    def test_xl_turbo_schema_uses_distilled_defaults(self):
        steps = Generate.params["num_inference_steps"]
        guidance = Generate.params["guidance_scale"]

        self.assertEqual(steps["default"], 8)
        self.assertEqual(guidance["default"], 1.0)
        self.assertIn("8 denoising steps", steps["description"])
        self.assertIn("guidance-distilled", guidance["description"])
        self.assertIn("above 1", guidance["description"])

    def test_execute_forwards_xl_turbo_defaults_when_values_are_omitted(self):
        pipeline = FakeAceStepPipeline()
        node = Generate("ace-defaults-test")
        node.progress = lambda *args, **kwargs: None

        node.execute(pipeline=pipeline, prompt="A short instrumental cue", audio_duration=1, sample_rate=48000)

        self.assertIsNotNone(pipeline.call_kwargs)
        self.assertEqual(pipeline.call_kwargs["num_inference_steps"], 8)
        self.assertEqual(pipeline.call_kwargs["guidance_scale"], 1.0)

    def test_execute_normalizes_string_bpm_for_ace_metadata(self):
        pipeline = FakeAceStepPipeline()
        node = Generate("ace-bpm-test")
        node.progress = lambda *args, **kwargs: None

        node.execute(
            pipeline=pipeline,
            prompt="A short instrumental cue",
            audio_duration=1,
            sample_rate=48000,
            bpm="170",
        )

        self.assertEqual(pipeline.call_kwargs["bpm"], 170)
        self.assertIsInstance(pipeline.call_kwargs["bpm"], int)


if __name__ == "__main__":
    unittest.main()
