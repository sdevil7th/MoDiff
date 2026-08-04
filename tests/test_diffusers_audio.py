import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.DiffusersAudio.main import (  # noqa: E402
    AUDIO_SAMPLE_RATE_OPTIONS,
    FuseAdapters,
    Generate,
    LoadAdapter,
    LoadPipeline,
    SetAdapters,
    audio_to_numpy,
    audio_to_tensor,
    crop_tail,
)
from modiff.server import to_bytes  # noqa: E402


class FakeAceStepPipeline:
    device = "cpu"
    sample_rate = 48000

    def __init__(self):
        self.call_kwargs = None

    def __call__(self, bpm=None, **kwargs):
        self.call_kwargs = {**kwargs, "bpm": bpm}
        return SimpleNamespace(audios=np.zeros((1, 480), dtype=np.float32))


class FakeSourceConditionedAceStepPipeline:
    device = "cpu"
    sample_rate = 48000

    def __init__(self):
        self.call_kwargs = None

    def __call__(
        self,
        prompt=None,
        lyrics=None,
        audio_duration=None,
        task_type=None,
        src_audio=None,
        reference_audio=None,
        audio_cover_strength=None,
        **kwargs,
    ):
        self.call_kwargs = {
            **kwargs,
            "prompt": prompt,
            "lyrics": lyrics,
            "audio_duration": audio_duration,
            "task_type": task_type,
            "src_audio": src_audio,
            "reference_audio": reference_audio,
            "audio_cover_strength": audio_cover_strength,
        }
        return SimpleNamespace(audios=np.zeros((1, 480), dtype=np.float32))


class DiffusersAudioGenerateTests(unittest.TestCase):
    def test_graph_contract_distinguishes_required_and_optional_audio_inputs(self):
        for node_class in (LoadAdapter, SetAdapters, FuseAdapters, Generate):
            with self.subTest(node=node_class.__name__):
                self.assertTrue(node_class.params["pipeline"]["required"])

        self.assertFalse(Generate.params["source_audio"]["required"])
        self.assertFalse(Generate.params["reference_audio"]["required"])
        self.assertTrue(Generate.params["lora_scale"]["hidden"])
        self.assertIn("per-call multiplier", Generate.params["lora_scale"]["description"])
        self.assertIn("ignored by ACE-Step", Generate.params["stable_audio_steps"]["description"])
        self.assertIn("ignored by ACE-Step", Generate.params["stable_audio_guidance"]["description"])
        self.assertIn("ignored by ACE-Step", Generate.params["num_waveforms"]["description"])

    def test_ace_step_lora_load_set_and_fuse_contracts(self):
        events = []

        class Pipeline:
            _modiff_audio_pipeline_class = "AceStepPipeline"

            def unload_lora_weights(self):
                events.append("unload")

            def load_lora_weights(self, path, **kwargs):
                events.append(("load", path, kwargs))

            def set_adapters(self, names, weights):
                events.append(("set", names, weights))

            def fuse_lora(self, **kwargs):
                events.append(("fuse", kwargs))

        pipeline = Pipeline()
        LoadAdapter("audio-lora").execute(
            pipeline=pipeline,
            adapter_path={"source": "local", "value": "/models/audio-style"},
            adapter_name="style",
            scale=0.6,
        )
        SetAdapters("audio-blend").execute(
            pipeline=pipeline,
            adapter_names="style",
            adapter_weights="0.4",
        )
        FuseAdapters("audio-fuse").execute(pipeline=pipeline, enabled=True, safe_fusing=True)

        self.assertEqual(events[0], "unload")
        self.assertEqual(events[1][0], "load")
        self.assertEqual(events[2], ("set", ["style"], [0.6]))
        self.assertEqual(events[3], ("set", ["style"], [0.4]))
        self.assertEqual(events[4], ("fuse", {"safe_fusing": True}))

    def test_audio_lora_rejects_non_ace_pipeline(self):
        with self.assertRaisesRegex(ValueError, "AceStepPipeline"):
            LoadAdapter("wrong-audio-lora").execute(
                pipeline=SimpleNamespace(_modiff_audio_pipeline_class="StableAudioPipeline"),
                adapter_path={"source": "local", "value": "/models/audio-style"},
            )

    def test_stable_audio_uses_native_generation_rate_and_requested_delivery_rate(self):
        class FakeStableAudio:
            _modiff_audio_pipeline_class = "StableAudioPipeline"
            device = "cpu"
            vae = SimpleNamespace(config={"sampling_rate": 44100})

            def __init__(self):
                self.call_kwargs = None

            def __call__(self, **kwargs):
                self.call_kwargs = kwargs
                return SimpleNamespace(audios=np.zeros((2, 2, 44100), dtype=np.float32))

        pipeline = FakeStableAudio()
        result = Generate().execute(
            pipeline=pipeline,
            prompt="Clear wooden impacts in a quiet room",
            negative_prompt="low quality",
            audio_duration=1,
            stable_audio_steps=120,
            stable_audio_guidance=0,
            num_waveforms=2,
            sample_rate=48000,
        )

        self.assertEqual(pipeline.call_kwargs["audio_end_in_s"], 1)
        self.assertEqual(pipeline.call_kwargs["num_inference_steps"], 120)
        self.assertEqual(pipeline.call_kwargs["guidance_scale"], 0)
        self.assertEqual(pipeline.call_kwargs["num_waveforms_per_prompt"], 2)
        self.assertEqual(result["sample_rate_out"], 48000)
        self.assertEqual(result["duration_seconds"], 1)
        self.assertEqual(result["audio"]["samples"].shape[-1], 48000)
        self.assertEqual(len(result["audio_variations"]), 2)
        self.assertTrue(all(item["samples"].shape == (2, 48000) for item in result["audio_variations"]))
        self.assertIs(result["audio"], result["audio_variations"][0])

    def test_explicit_zero_audio_controls_are_forwarded(self):
        pipeline = FakeSourceConditionedAceStepPipeline()
        node = Generate("ace-zero-values-test")
        node.progress = lambda *args, **kwargs: None
        source = {"samples": np.zeros((2, 480), dtype=np.float32), "sample_rate": 48000}

        result = node.execute(
            pipeline=pipeline,
            task_type="cover",
            source_audio=source,
            prompt="A silent control fixture",
            audio_duration=0.01,
            guidance_scale=0,
            shift=0,
            audio_cover_strength=0,
        )

        self.assertEqual(pipeline.call_kwargs["guidance_scale"], 0)
        self.assertEqual(pipeline.call_kwargs["shift"], 0)
        self.assertEqual(pipeline.call_kwargs["audio_cover_strength"], 0)
        self.assertEqual(result["audio_variations"], [result["audio"]])

    def test_unsigned_pcm_midpoint_normalizes_to_zero(self):
        samples, _sample_rate = audio_to_numpy(
            {"samples": np.asarray([0, 128, 255], dtype=np.uint8), "sample_rate": 8000}
        )

        np.testing.assert_allclose(samples[0], [-1.0, 0.0, 127 / 128], atol=1e-7)

    def test_audio_frame_tail_is_cropped_to_requested_duration(self):
        audio = {
            "samples": np.zeros((2, 24000 * 2), dtype=np.float32),
            "sample_rate": 24000,
            "duration_seconds": 2.0,
        }
        cropped = crop_tail(audio, 0.0, 1.0)
        self.assertEqual(cropped["samples"].shape, (2, 24000))
        self.assertEqual(cropped["duration_seconds"], 1.0)

    def test_shared_audio_contract_serializes_to_wav_bytes(self):
        encoded = to_bytes(
            "audio",
            {"samples": np.asarray([[0.0, 0.5, -0.5]], dtype=np.float32), "sample_rate": 24000},
        )
        self.assertEqual(encoded[:4], b"RIFF")
        self.assertEqual(encoded[8:12], b"WAVE")

        with tempfile.NamedTemporaryFile(suffix=".wav") as output:
            output.write(encoded)
            output.flush()
            self.assertEqual(to_bytes("audio", output.name), encoded)

    def test_loader_rejects_unsupported_mode_before_resolving_pipeline(self):
        node = LoadPipeline("ace-mode-test")
        with self.assertRaisesRegex(ValueError, "does not support video_to_video"):
            node.execute(pipeline_class="AceStepPipeline", mode="video_to_video")

    def test_no_offload_audio_pipeline_loads_directly_on_cuda(self):
        loaded = {}

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                loaded.update({"repo": repo, "kwargs": kwargs})
                return cls()

        node = LoadPipeline("ace-direct-load-test")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch("modules.DiffusersAudio.main.pipeline_class_from_name", return_value=FakePipeline),
            patch("modules.DiffusersAudio.main.apply_pipeline_offload"),
        ):
            node.execute(
                model_id="org/ace-step",
                pipeline_class="AceStepPipeline",
                mode="text_to_audio",
                device="cuda:0",
                auto_offload=False,
                offload_mode="none",
            )

        self.assertEqual(loaded["repo"], "org/ace-step")
        self.assertEqual(loaded["kwargs"]["device_map"], "cuda")
        self.assertIsNone(loaded["kwargs"]["revision"])

    def test_curated_audio_pipeline_uses_catalog_revision(self):
        loaded = {}

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                loaded.update({"repo": repo, "kwargs": kwargs})
                return cls()

        node = LoadPipeline("ace-revision-test")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch("modules.DiffusersAudio.main.pipeline_class_from_name", return_value=FakePipeline),
            patch("modules.DiffusersAudio.main.apply_pipeline_offload"),
        ):
            node.execute(
                model_id="ACE-Step/acestep-v15-xl-turbo-diffusers",
                pipeline_class="AceStepPipeline",
                mode="text_to_audio",
                device="cpu",
                auto_offload=False,
                offload_mode="none",
            )

        self.assertEqual(loaded["kwargs"]["revision"], "200ba991ae448051e14b0183157e35c2d27c9fb0")

    def test_xl_turbo_schema_uses_distilled_defaults(self):
        steps = Generate.params["num_inference_steps"]
        guidance = Generate.params["guidance_scale"]

        self.assertEqual(steps["default"], 8)
        self.assertEqual(guidance["default"], 1.0)
        self.assertIn("8 denoising steps", steps["description"])
        self.assertIn("guidance-distilled", guidance["description"])
        self.assertIn("above 1", guidance["description"])

    def test_generate_sample_rate_is_a_four_option_delivery_selector(self):
        self.assertEqual(
            Generate.params["sample_rate"]["options"],
            AUDIO_SAMPLE_RATE_OPTIONS,
        )

    def test_ace_output_is_resampled_to_requested_delivery_rate(self):
        pipeline = FakeAceStepPipeline()
        node = Generate("ace-sample-rate-test")
        node.progress = lambda *args, **kwargs: None

        result = node.execute(
            pipeline=pipeline,
            prompt="A short instrumental cue",
            audio_duration=0.01,
            sample_rate=96000,
        )

        self.assertEqual(result["sample_rate_out"], 96000)
        self.assertEqual(result["audio"]["sample_rate"], 96000)
        self.assertEqual(result["audio"]["samples"].shape[-1], 960)

    def test_execute_forwards_xl_turbo_defaults_when_values_are_omitted(self):
        pipeline = FakeAceStepPipeline()
        node = Generate("ace-defaults-test")
        node.progress = lambda *args, **kwargs: None

        node.execute(pipeline=pipeline, prompt="A short instrumental cue", audio_duration=1, sample_rate=48000)

        self.assertIsNotNone(pipeline.call_kwargs)
        self.assertEqual(pipeline.call_kwargs["num_inference_steps"], 8)
        self.assertEqual(pipeline.call_kwargs["guidance_scale"], 1.0)

    def test_execute_reports_indeterminate_progress_before_ace_pipeline_starts(self):
        events = []

        class ProgressAwarePipeline(FakeAceStepPipeline):
            def __call__(self, bpm=None, **kwargs):
                events.append(("pipeline",))
                return super().__call__(bpm=bpm, **kwargs)

        pipeline = ProgressAwarePipeline()
        node = Generate("ace-initial-progress-test")
        node.progress = lambda value, **metadata: events.append(("progress", value, metadata))

        node.execute(
            pipeline=pipeline,
            prompt="A short instrumental cue",
            audio_duration=1,
            num_inference_steps=8,
            sample_rate=48000,
        )

        self.assertEqual(events[0][0], "progress")
        self.assertEqual(events[0][1], -1)
        self.assertEqual(events[0][2]["phase"], "denoising")
        self.assertEqual(events[0][2]["message"], "Generating audio (text2music)")
        self.assertEqual(events[0][2]["current_step"], 0)
        self.assertEqual(events[0][2]["total_steps"], 8)
        self.assertEqual(events[1], ("pipeline",))

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

    def test_cover_routes_source_track_to_reference_audio_without_optional_audio_code_modules(self):
        pipeline = FakeSourceConditionedAceStepPipeline()
        node = Generate("ace-cover-reference-test")
        node.progress = lambda *args, **kwargs: None
        source = {"samples": np.zeros((2, 480), dtype=np.float32), "sample_rate": 48000}

        node.execute(
            pipeline=pipeline,
            task_type="cover",
            source_audio=source,
            prompt="A restrained acoustic variation",
            audio_duration=0.01,
        )

        self.assertIsNone(pipeline.call_kwargs["src_audio"])
        self.assertIsNotNone(pipeline.call_kwargs["reference_audio"])

    def test_repaint_routes_source_track_to_src_audio_without_implicit_timbre_reference(self):
        pipeline = FakeSourceConditionedAceStepPipeline()
        node = Generate("ace-repaint-source-test")
        node.progress = lambda *args, **kwargs: None
        source = {"samples": np.zeros((2, 480), dtype=np.float32), "sample_rate": 48000}

        node.execute(
            pipeline=pipeline,
            task_type="repaint",
            source_audio=source,
            prompt="Repair the selected interval",
            audio_duration=0.01,
            repainting_start=0,
            repainting_end=0.01,
        )

        self.assertIsNotNone(pipeline.call_kwargs["src_audio"])
        self.assertIsNone(pipeline.call_kwargs["reference_audio"])

    def test_audio_input_is_resampled_to_the_pipeline_native_rate(self):
        source = {"samples": np.zeros((2, 44100), dtype=np.float32), "sample_rate": 44100}

        tensor = audio_to_tensor(source, device="cpu", target_sample_rate=48000)

        self.assertEqual(tuple(tensor.shape), (2, 48000))


if __name__ == "__main__":
    unittest.main()
