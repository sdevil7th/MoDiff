import inspect
import sys
import tempfile
import unittest
from contextlib import chdir
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
from PIL import Image

import modules as module_registry
from modules.DiffusersVideo import (
    BuildShotJobs,
    Generate,
    GenerateLTX2,
    GenerateVideoAudio,
    GenerateSequence,
    GenerateShotJob,
    LoadPipeline,
    PlanLongVideo,
)
from modules.DiffusersVideo.main import (
    ALLEGRO_REPO,
    ALLEGRO_REVISION,
    ANIMATEDIFF_BASE_REPO,
    ANIMATEDIFF_BASE_REVISION,
    ANIMATEDIFF_MOTION_REPO,
    ANIMATEDIFF_MOTION_REVISION,
    ANIMATELCM_LORA_ADAPTER_NAME,
    ANIMATELCM_LORA_SCALE,
    ANIMATELCM_LORA_WEIGHT_NAME,
    ANIMATELCM_MOTION_REPO,
    ANIMATELCM_MOTION_REVISION,
    COGVIDEOX_2B_REPO,
    COGVIDEOX_2B_REVISION,
    FRAMEPACK_BASE_REPO,
    FRAMEPACK_VISION_REPO,
    LATTE_REPO,
    LATTE_REVISION,
    MOCHI_REPO,
    MOCHI_REVISION,
    LTX_DISTILLED_TIMESTEPS,
    STABLE_VIDEO_DIFFUSION_REPO,
    STABLE_VIDEO_DIFFUSION_REVISION,
    VIDEO_PIPELINE_ADAPTERS,
    VIDEO_PIPELINE_EXECUTE_HANDLERS,
    VIDEO_PIPELINE_LOAD_HANDLERS,
    VIDEO_MODE_FIELD_CONTRACTS,
    WAN_VACE_MODE_MEDIA_CONTRACTS,
    _pipeline_adapter,
    _resolve_adapter_model_selection,
    _resolve_loader_revision,
    get_video_pipeline_adapter,
)


class FakeLTXVideoCondition:
    def __init__(self, image=None, video=None, frame_index=0, strength=1.0):
        self.image = image
        self.video = video
        self.frame_index = frame_index
        self.strength = strength


class FakeLTX2VideoCondition:
    def __init__(self, frames=None, index=0, strength=1.0):
        self.frames = frames
        self.index = index
        self.strength = strength


class DiffusersVideoRegistryTests(unittest.TestCase):
    def test_quality_shot_jobs_pair_six_keyframes_with_six_five_second_shots(self):
        images = [object() for _ in range(6)]
        shots = [
            {"title": f"Shot {index + 1}", "prompt": f"Visible story action {index + 1}", "duration_seconds": 5}
            for index in range(6)
        ]

        result = BuildShotJobs().execute(shots=shots, opening_images=images, base_seed=100, fps=16)

        self.assertEqual(result["count"], 6)
        self.assertEqual(result["planned_duration_seconds"], 30.375)
        self.assertEqual([job["num_frames"] for job in result["jobs"]], [81] * 6)
        self.assertEqual([job["seed"] for job in result["jobs"]], list(range(100, 106)))
        self.assertTrue(all(job["conditioning_strength"] == 0.9 for job in result["jobs"]))
        self.assertTrue(all(job["opening_image"] is images[index] for index, job in enumerate(result["jobs"])))

    def test_quality_shot_jobs_allow_a_per_shot_conditioning_strength_override(self):
        result = BuildShotJobs().execute(
            shots=[
                {
                    "prompt": "Preserve the parked car while the station door opens.",
                    "duration_seconds": 5,
                    "conditioning_strength": 0.8,
                },
                {"prompt": "The same car drives away through snow.", "duration_seconds": 5},
            ],
            opening_images=[object(), object()],
            conditioning_strength=0.9,
        )

        self.assertEqual([job["conditioning_strength"] for job in result["jobs"]], [0.8, 0.9])

    def test_quality_shot_jobs_preserve_explicit_zero_controls(self):
        result = BuildShotJobs().execute(
            shots=[{"prompt": "Release every conditioning control.", "duration_seconds": 5}],
            opening_images=[object()],
            guidance_scale=0,
            secondary_guidance_scale=0,
            conditioning_strength=0,
        )

        job = result["jobs"][0]
        self.assertEqual(job["guidance_scale"], 0)
        self.assertEqual(job["secondary_guidance_scale"], 0)
        self.assertEqual(job["conditioning_strength"], 0)

    def test_quality_shot_jobs_reject_short_shots_and_mismatched_keyframes(self):
        with self.assertRaisesRegex(ValueError, "at least 5 seconds"):
            BuildShotJobs().execute(
                shots=[{"prompt": "Too short", "duration_seconds": 4.9}],
                opening_images=[object()],
            )

    def test_quality_text_shot_jobs_do_not_require_keyframes(self):
        result = BuildShotJobs().execute(
            shots=[{"prompt": "A photoreal drone crosses the restored greenhouse.", "duration_seconds": 5}],
            mode="text_to_video",
            fps=24,
            width=1280,
            height=704,
            steps=50,
        )

        self.assertEqual(result["jobs"][0]["mode"], "text_to_video")
        self.assertIsNone(result["jobs"][0]["opening_image"])
        self.assertEqual(result["jobs"][0]["num_frames"], 121)
        with self.assertRaisesRegex(ValueError, "needs 2 opening images"):
            BuildShotJobs().execute(
                shots=[
                    {"prompt": "First", "duration_seconds": 5},
                    {"prompt": "Second", "duration_seconds": 5},
                ],
                opening_images=[object()],
            )

    def test_generate_shot_job_maps_the_normalized_record_to_the_generic_generator(self):
        pipeline = object()
        opening = object()
        ending = object()
        output = {"video_out": ["frame"], "frames_out": 1}
        with patch.object(Generate, "execute", return_value=output) as execute:
            result = GenerateShotJob().execute(
                pipeline=pipeline,
                job={
                    "prompt": "A purposeful camera move reveals the restored seed vault.",
                    "opening_image": opening,
                    "ending_image": ending,
                    "negative_prompt": "flicker",
                    "num_frames": 81,
                    "fps": 16,
                    "steps": 40,
                    "guidance_scale": 3.5,
                    "secondary_guidance_scale": 3.25,
                    "conditioning_strength": 0.7,
                    "seed": 77,
                },
            )

        self.assertEqual(result["fps_out"], 16)
        self.assertEqual(result["width_out"], 832)
        self.assertEqual(result["height_out"], 480)
        self.assertIn("width_out", GenerateShotJob.params)
        self.assertIn("height_out", GenerateShotJob.params)
        self.assertIs(execute.call_args.kwargs["pipeline"], pipeline)
        self.assertEqual(execute.call_args.kwargs["reference_images"], [opening])
        self.assertIs(execute.call_args.kwargs["last_image"], ending)
        self.assertEqual(execute.call_args.kwargs["num_frames"], 81)
        self.assertEqual(execute.call_args.kwargs["secondary_guidance_scale"], 3.25)
        self.assertEqual(execute.call_args.kwargs["strength"], 0.7)

    def test_generate_shot_job_preserves_explicit_zero_controls(self):
        with patch.object(Generate, "execute", return_value={"video_out": [], "frames_out": 0}) as execute:
            GenerateShotJob().execute(
                pipeline=object(),
                job={
                    "prompt": "Unconditioned motion fixture.",
                    "opening_image": object(),
                    "guidance_scale": 0,
                    "secondary_guidance_scale": 0,
                    "conditioning_strength": 0,
                },
            )

        self.assertEqual(execute.call_args.kwargs["guidance_scale"], 0)
        self.assertEqual(execute.call_args.kwargs["secondary_guidance_scale"], 0)
        self.assertEqual(execute.call_args.kwargs["strength"], 0)

    def test_long_video_planner_emits_loop_ready_ltx_chunks_and_one_framepack_job(self):
        ltx = PlanLongVideo().execute(
            prompt="A continuous tracking shot",
            target_seconds=30,
            fps=16,
            strategy="ltx_continuation",
            chunk_seconds=5,
            overlap_seconds=0.25,
            seed=40,
        )
        framepack = PlanLongVideo().execute(
            prompt="A continuous tracking shot",
            target_seconds=30,
            fps=16,
            strategy="framepack_continuous",
            seed=40,
        )

        self.assertGreater(ltx["job_count"], 1)
        self.assertTrue(all((job["num_frames"] - 1) % 8 == 0 for job in ltx["jobs"]))
        self.assertTrue(ltx["jobs"][1]["uses_previous_last_frame"])
        self.assertEqual(framepack["job_count"], 1)
        self.assertEqual(framepack["planned_frames"], 480)

    def test_facade_has_normalized_contract(self):
        self.assertEqual(LoadPipeline.category, "Diffusers Video")
        self.assertEqual(LoadPipeline.params["pipeline"]["type"], "video_diffusion_pipeline")
        self.assertIn("pipeline_class", LoadPipeline.params)
        self.assertEqual(Generate.params["pipeline"]["type"], "video_diffusion_pipeline")
        for name in ("video", "mask", "reference_images", "video_out"):
            self.assertIn(name, Generate.params)

    def test_graph_contract_marks_core_video_inputs_without_requiring_mode_specific_media(self):
        self.assertTrue(Generate.params["pipeline"]["required"])
        for name in ("last_image", "pose_video", "face_video", "background_video"):
            with self.subTest(input=name):
                self.assertFalse(Generate.params[name]["required"])
        self.assertFalse(BuildShotJobs.params["ending_images"]["required"])
        self.assertTrue(GenerateShotJob.params["pipeline"]["required"])
        self.assertTrue(GenerateShotJob.params["job"]["required"])

    def test_unknown_pipeline_is_rejected_before_model_load(self):
        with self.assertRaisesRegex(ValueError, "Unsupported Diffusers video pipeline"):
            get_video_pipeline_adapter("UnknownVideoPipeline")

    def test_missing_null_and_malformed_pipeline_class_fail_closed_before_loading(self):
        node = LoadPipeline("video-loader-identity")
        node._load_wan_vace = MagicMock(side_effect=AssertionError("loader must not run"))

        for values in (
            {},
            {"pipeline_class": None},
            {"pipeline_class": ""},
            {"pipeline_class": " WanVACEPipeline "},
            {"pipeline_class": []},
            {"pipeline_class": {}},
        ):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "registered Diffusers video pipeline class is required"):
                    node(**values)
        node._load_wan_vace.assert_not_called()

    def test_unknown_pipeline_class_survives_node_normalization_for_an_actionable_error(self):
        node = LoadPipeline("video-loader-unknown")
        node._load_wan_vace = MagicMock(side_effect=AssertionError("loader must not run"))

        with self.assertRaisesRegex(ValueError, "Unsupported Diffusers video pipeline class UnknownVideoPipeline"):
            node(pipeline_class="UnknownVideoPipeline")
        node._load_wan_vace.assert_not_called()

    def test_registered_adapter_without_a_dispatch_handler_fails_before_loading_or_execution(self):
        pipeline = SimpleNamespace(_modiff_video_pipeline_class="HunyuanVideoFramepackPipeline")
        loader = LoadPipeline()
        loader._load_framepack = MagicMock(side_effect=AssertionError("FramePack loader must not run"))
        generator = Generate()
        generator._execute_framepack = MagicMock(side_effect=AssertionError("FramePack generator must not run"))

        with patch.dict(VIDEO_PIPELINE_LOAD_HANDLERS, {"HunyuanVideoFramepackPipeline": None}):
            with self.assertRaisesRegex(RuntimeError, "No loader handler.*HunyuanVideoFramepackPipeline"):
                loader.execute(pipeline_class="HunyuanVideoFramepackPipeline")
        with patch.dict(VIDEO_PIPELINE_EXECUTE_HANDLERS, {"HunyuanVideoFramepackPipeline": None}):
            with self.assertRaisesRegex(RuntimeError, "No execution handler.*HunyuanVideoFramepackPipeline"):
                generator.execute(pipeline=pipeline, mode="image_to_video")

        loader._load_framepack.assert_not_called()
        generator._execute_framepack.assert_not_called()

    def test_framepack_has_explicit_load_and_execute_dispatch(self):
        self.assertEqual(set(VIDEO_PIPELINE_LOAD_HANDLERS), set(VIDEO_PIPELINE_ADAPTERS))
        self.assertEqual(set(VIDEO_PIPELINE_EXECUTE_HANDLERS), set(VIDEO_PIPELINE_ADAPTERS))
        self.assertEqual(
            VIDEO_PIPELINE_LOAD_HANDLERS["HunyuanVideoFramepackPipeline"],
            "_load_framepack",
        )
        self.assertEqual(
            VIDEO_PIPELINE_EXECUTE_HANDLERS["HunyuanVideoFramepackPipeline"],
            "_execute_framepack",
        )

        pipeline = SimpleNamespace()
        loader = LoadPipeline()
        with patch.object(loader, "_load_framepack", return_value=pipeline) as load_framepack:
            result = loader.execute(pipeline_class="HunyuanVideoFramepackPipeline")
        load_framepack.assert_called_once()
        self.assertIs(result["pipeline"], pipeline)
        self.assertEqual(pipeline._modiff_video_pipeline_class, "HunyuanVideoFramepackPipeline")
        self.assertEqual(result["resolved_artifact"], "lllyasviel/FramePackI2V_HY")

    def test_real_loader_normalizes_a_stale_managed_repo_before_caching(self):
        pipeline = SimpleNamespace()
        node = LoadPipeline("normalized-video-loader-cache")
        node._load_framepack = MagicMock(return_value=pipeline)
        selected_class = "HunyuanVideoFramepackPipeline"
        stale = {"source": "hub", "value": "Wan-AI/Wan2.1-VACE-1.3B-diffusers"}
        corrected = {"source": "hub", "value": "lllyasviel/FramePackI2V_HY"}
        case_variant = {"source": "HUB", "value": "LLLYASVIEL/FRAMEPACKI2V_HY"}

        with patch("modiff.NodeBase.modelstore.is_hf_cached", return_value=True):
            first = node(pipeline_class=selected_class, model_id=stale)
            second = node(pipeline_class=selected_class, model_id=corrected)
            third = node(pipeline_class=selected_class, model_id=case_variant)

        self.assertIs(first, second)
        self.assertIs(first, third)
        self.assertEqual(node.params["model_id"], corrected)
        self.assertEqual(first["resolved_artifact"], corrected["value"])
        self.assertEqual(first["pipeline"]._modiff_video_revision, "86cef4396041b6002c957852daac4c91aaa47c79")
        node._load_framepack.assert_called_once()

    def test_hub_pipeline_revisions_fail_closed_before_nodebase_or_loader(self):
        custom_revision = "0123456789abcdef0123456789abcdef01234567"
        custom_selection = {"source": "hub", "value": "organization/custom-framepack"}
        invalid_revisions = (
            None,
            "",
            "main",
            custom_revision.upper(),
            f" {custom_revision}",
            custom_revision[:-1],
            123,
            False,
        )
        invalid = LoadPipeline("strict-video-hub-revision")
        invalid._load_framepack = MagicMock(side_effect=AssertionError("loader must not run"))
        for revision in invalid_revisions:
            with self.subTest(revision=revision):
                with self.assertRaisesRegex(ValueError, "immutable lowercase|exact lowercase"):
                    invalid(
                        pipeline_class="HunyuanVideoFramepackPipeline",
                        model_id=custom_selection,
                        revision=revision,
                    )
        invalid._load_framepack.assert_not_called()

        curated_mismatch = LoadPipeline("strict-video-curated-mismatch")
        curated_mismatch._load_framepack = MagicMock(side_effect=AssertionError("loader must not run"))
        with self.assertRaisesRegex(ValueError, "pinned to .* does not match"):
            curated_mismatch(
                pipeline_class="HunyuanVideoFramepackPipeline",
                model_id={"source": "hub", "value": "lllyasviel/FramePackI2V_HY"},
                revision="0000000000000000000000000000000000000000",
            )
        curated_mismatch._load_framepack.assert_not_called()

        custom_pipeline = SimpleNamespace()
        valid_custom = LoadPipeline("strict-video-custom-valid")
        valid_custom._load_framepack = MagicMock(return_value=custom_pipeline)
        with patch("modiff.NodeBase.modelstore.is_hf_cached", return_value=True):
            custom_result = valid_custom(
                pipeline_class="HunyuanVideoFramepackPipeline",
                model_id=custom_selection,
                revision=custom_revision,
            )
        self.assertEqual(custom_result["pipeline"]._modiff_video_revision, custom_revision)
        self.assertEqual(valid_custom._load_framepack.call_args.args[1]["revision"], custom_revision)

        curated_pipeline = SimpleNamespace()
        valid_curated = LoadPipeline("strict-video-curated-valid")
        valid_curated._load_framepack = MagicMock(return_value=curated_pipeline)
        with patch("modiff.NodeBase.modelstore.is_hf_cached", return_value=True):
            curated_result = valid_curated(
                pipeline_class="HunyuanVideoFramepackPipeline",
                model_id={"source": "hub", "value": "lllyasviel/FramePackI2V_HY"},
            )
        self.assertEqual(
            curated_result["pipeline"]._modiff_video_revision,
            "86cef4396041b6002c957852daac4c91aaa47c79",
        )

    def test_tagged_pipeline_recovers_only_its_exact_registered_adapter(self):
        tagged = SimpleNamespace(_modiff_video_pipeline_class="LTXConditionPipeline")
        self.assertEqual(_pipeline_adapter(tagged).pipeline_class, "LTXConditionPipeline")

        tagged._modiff_video_pipeline_class = "UnknownVideoPipeline"
        with self.assertRaisesRegex(ValueError, "Unsupported Diffusers video pipeline"):
            _pipeline_adapter(tagged)

    def test_real_node_rejects_null_or_runtime_inconsistent_pipeline_tags(self):
        null_tagged = type(
            "LTXConditionPipeline",
            (),
            {
                "_modiff_video_pipeline_class": None,
                "_modiff_video_repo": "Lightricks/LTX-Video-0.9.8-13B-distilled",
            },
        )()
        inconsistent = type(
            "WanVACEPipeline",
            (),
            {
                "_modiff_video_pipeline_class": "HunyuanVideoFramepackPipeline",
                "_modiff_video_repo": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
            },
        )()
        managed_repo_inconsistent = type(
            "WanPipeline",
            (),
            {
                "_modiff_video_pipeline_class": "WanTI2VPipeline",
                "_modiff_video_repo": "WAN-AI/WAN2.1-T2V-1.3B-DIFFUSERS",
            },
        )()

        for pipeline, message in (
            (null_tagged, "registered Diffusers video pipeline class is required"),
            (inconsistent, "identity is inconsistent.*tagged as HunyuanVideoFramepackPipeline"),
            (managed_repo_inconsistent, "identity is inconsistent.*tagged as WanTI2VPipeline"),
        ):
            with self.subTest(message=message):
                node = Generate("strict-video-tag")
                node._execute_ltx = MagicMock(side_effect=AssertionError("LTX handler must not run"))
                node._execute_framepack = MagicMock(side_effect=AssertionError("FramePack handler must not run"))
                node._execute_wan_vace = MagicMock(side_effect=AssertionError("VACE handler must not run"))
                node._execute_wan_text_to_video = MagicMock(
                    side_effect=AssertionError("Wan text handler must not run")
                )
                with self.assertRaisesRegex(RuntimeError, message):
                    node(
                        pipeline=pipeline,
                        mode="image_to_video" if pipeline is inconsistent else "text_to_video",
                        reference_images=[Image.new("RGB", (4, 4))] if pipeline is inconsistent else None,
                    )
                node._execute_ltx.assert_not_called()
                node._execute_framepack.assert_not_called()
                node._execute_wan_vace.assert_not_called()
                node._execute_wan_text_to_video.assert_not_called()

    def test_known_shared_runtime_accepts_a_consistent_tag_and_custom_repo(self):
        pipeline = type(
            "WanPipeline",
            (),
            {
                "_modiff_video_pipeline_class": "WanTI2VPipeline",
                "_modiff_video_repo": "organization/custom-wan-ti2v",
            },
        )()
        node = Generate("consistent-video-tag")
        node._execute_wan_text_to_video = MagicMock(
            return_value={"video_out": [], "width_out": 1, "height_out": 1, "frames_out": 0}
        )

        result = node(pipeline=pipeline, mode="text_to_video")

        self.assertEqual(result["frames_out"], 0)
        node._execute_wan_text_to_video.assert_called_once()

    def test_untagged_unique_runtime_class_requires_an_exact_reviewed_repo(self):
        pipeline = type("LTXConditionPipeline", (), {})()
        with self.assertRaisesRegex(ValueError, "without an exact reviewed repository"):
            _pipeline_adapter(pipeline)

        pipeline._modiff_video_repo = "organization/custom-ltx"
        with self.assertRaisesRegex(ValueError, "unreviewed repository"):
            _pipeline_adapter(pipeline)

        pipeline._modiff_video_repo = "Lightricks/LTX-Video-0.9.8-13B-distilled"
        self.assertEqual(_pipeline_adapter(pipeline).pipeline_class, "LTXConditionPipeline")

    def test_tagged_loader_output_keeps_custom_repository_support(self):
        pipeline = SimpleNamespace(
            _modiff_video_pipeline_class="LTXConditionPipeline",
            _modiff_video_repo="organization/custom-ltx",
        )
        self.assertEqual(_pipeline_adapter(pipeline).pipeline_class, "LTXConditionPipeline")

    def test_untagged_unknown_runtime_class_fails_closed(self):
        pipeline = type("UnreviewedVideoPipeline", (), {})()
        with self.assertRaisesRegex(ValueError, "Cannot recover.*UnreviewedVideoPipeline"):
            _pipeline_adapter(pipeline)

    def test_untagged_shared_wan_runtime_is_ambiguous_without_an_exact_reviewed_repo(self):
        pipeline = type("WanPipeline", (), {})()
        with self.assertRaisesRegex(ValueError, "without an exact reviewed repository"):
            _pipeline_adapter(pipeline)

        pipeline._modiff_video_repo = "organization/custom-wan"
        with self.assertRaisesRegex(ValueError, "unreviewed repository"):
            _pipeline_adapter(pipeline)

    def test_exact_reviewed_repo_disambiguates_a_shared_wan_runtime(self):
        pipeline = type(
            "WanPipeline",
            (),
            {"_modiff_video_repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers"},
        )()
        self.assertEqual(_pipeline_adapter(pipeline).pipeline_class, "WanTI2VPipeline")

    def test_pipeline_signal_publishes_backend_owned_exact_adapter_modes(self):
        loader = LoadPipeline("loader")
        loader.set_field_value = MagicMock()
        loader.set_field_params = MagicMock()
        loader.select_adapter(
            {
                "pipeline_class": "HunyuanVideoFramepackPipeline",
                "model_id": {"source": "hub", "value": "Wan-AI/Wan2.1-VACE-1.3B-diffusers"},
            },
            None,
        )

        loader.set_field_value.assert_called_once_with(
            {
                "model_id": {"source": "hub", "value": "lllyasviel/FramePackI2V_HY"},
                "revision": "86cef4396041b6002c957852daac4c91aaa47c79",
            }
        )
        signal = loader.set_field_params.call_args.args[1]["signal"]
        self.assertEqual(
            signal["value"],
            {
                "schemaVersion": 1,
                "library": "diffusers",
                "mediaKind": "video",
                "pipelineClass": "HunyuanVideoFramepackPipeline",
                "modes": ["image_to_video"],
            },
        )

        generator = Generate("generator")
        generator.set_field_params = MagicMock()
        generator.update_adapter_modes(
            {
                "mode": "text_to_video",
                "video_contract": {
                    "schemaVersion": 1,
                    "library": "diffusers",
                    "mediaKind": "video",
                    "pipelineClass": "HunyuanVideoFramepackPipeline",
                    "modes": ["image_to_video"],
                },
            },
            None,
        )
        updates = {call.args[0]: call.args[1] for call in generator.set_field_params.call_args_list}
        self.assertEqual(
            updates["mode"],
            {"options": ["image_to_video"], "default": "image_to_video", "value": "image_to_video"},
        )
        self.assertEqual(updates["reference_images"], {"hidden": False, "required": True})
        self.assertEqual(updates["video"], {"hidden": True, "required": False})
        self.assertEqual(updates["framepack_sampling"], {"hidden": False})
        self.assertEqual(
            updates["strength"]["fieldOptions"]["studioBinding"],
            {
                "schemaVersion": 1,
                "group": "video-strength",
                "formFields": ["strength"],
                "transform": "identity",
            },
        )
        self.assertEqual(LoadPipeline.params["pipeline_class"]["onChange"], "select_adapter")
        self.assertEqual(LoadPipeline.params["model_id"]["onChange"], "select_adapter")
        self.assertEqual(Generate.params["mode"]["onChange"], "update_adapter_modes")
        self.assertEqual(
            Generate.params["pipeline"]["onSignal"],
            [
                {"action": "value", "target": "video_contract"},
                {"action": "exec", "data": "update_adapter_modes"},
            ],
        )
        self.assertEqual(
            LoadPipeline.params["pipeline"]["signal"]["value"],
            {
                "schemaVersion": 1,
                "library": "diffusers",
                "mediaKind": "video",
                "pipelineClass": "WanVACEPipeline",
                "modes": list(VIDEO_PIPELINE_ADAPTERS["WanVACEPipeline"].modes),
            },
        )

    def test_video_model_action_couples_repository_and_revision_before_real_execution(self):
        stale_revision = "0" * 40
        replacement_revision = "1234567890abcdef1234567890abcdef12345678"
        replacement = {"source": "hub", "value": "organization/replacement-video"}
        loader = LoadPipeline("video-model-identity-action")
        loader._sid = "video-browser-session"
        messages = []
        current_server = SimpleNamespace(
            _current_dynamic_message_identity_payload=lambda: {},
            queue_message=lambda message, sid=None: messages.append((message, sid)),
        )

        with patch("modiff.NodeBase._server", return_value=current_server):
            loader.select_adapter(
                {
                    "pipeline_class": "LTXConditionPipeline",
                    "model_id": replacement,
                    "revision": stale_revision,
                },
                {"key": "model_id"},
            )

        value_message = next(message for message, _sid in messages if message["type"] == "set_field_value")
        self.assertEqual(value_message["fields"], {"revision": ""})
        self.assertEqual(
            next(sid for message, sid in messages if message["type"] == "set_field_value"),
            "video-browser-session",
        )

        preserving = LoadPipeline("video-custom-pin-class-action")
        preserving.set_field_params = MagicMock()
        preserving.set_field_value = MagicMock()
        preserving.select_adapter(
            {
                "pipeline_class": "LTXConditionPipeline",
                "model_id": replacement,
                "revision": replacement_revision,
            },
            {"key": "pipeline_class"},
        )
        preserving.set_field_value.assert_not_called()

        upstream_calls = []

        class FakePipelineClass:
            @classmethod
            def from_pretrained(cls, repository, **kwargs):
                upstream_calls.append((repository, kwargs["revision"]))
                return SimpleNamespace()

        executing = LoadPipeline("video-replacement-execution")
        executing.mm_add = MagicMock()
        with (
            patch("diffusers.LTXConditionPipeline", FakePipelineClass),
            patch("modules.DiffusersVideo.main.local_files_only", return_value=True),
            patch("modules.DiffusersVideo.main.apply_pipeline_offload"),
            patch("modules.DiffusersRuntime.main.apply_execution_recipe_to_pipeline"),
        ):
            result = executing.execute(
                pipeline_class="LTXConditionPipeline",
                model_id=replacement,
                revision=replacement_revision,
            )

        self.assertEqual(upstream_calls, [(replacement["value"], replacement_revision)])
        self.assertEqual(result["pipeline"]._modiff_video_repo, replacement["value"])
        self.assertEqual(result["pipeline"]._modiff_video_revision, replacement_revision)

    def test_video_model_action_publishes_catalog_pin_and_clears_local_revision(self):
        cataloged = LoadPipeline("video-catalog-pin-action")
        cataloged.set_field_params = MagicMock()
        cataloged.set_field_value = MagicMock()
        cataloged.select_adapter(
            {
                "pipeline_class": "LTXConditionPipeline",
                "model_id": {
                    "source": "hub",
                    "value": "Lightricks/LTX-Video-0.9.8-13B-distilled",
                },
                "revision": "0" * 40,
            },
            {"key": "model_id"},
        )
        self.assertEqual(
            cataloged.set_field_value.call_args.args[0]["revision"],
            "7c64400e1861cc0d7b98d570a1926d5408ec60cd",
        )

        with tempfile.TemporaryDirectory() as temporary:
            local_model = Path(temporary) / "local-video"
            local_model.mkdir()
            local = LoadPipeline("video-local-revision-action")
            local.set_field_params = MagicMock()
            local.set_field_value = MagicMock()
            local.select_adapter(
                {
                    "pipeline_class": "LTXConditionPipeline",
                    "model_id": {"source": "local", "value": str(local_model)},
                    "revision": "0" * 40,
                },
                {"key": "model_id"},
            )
        self.assertEqual(local.set_field_value.call_args.args[0]["revision"], "")

    def test_mode_action_rejects_a_stale_or_unknown_video_contract(self):
        generator = Generate("generator")
        generator.set_field_params = MagicMock()
        with self.assertRaisesRegex(ValueError, "valid adapter contract"):
            generator.update_adapter_modes({}, None)
        with self.assertRaisesRegex(ValueError, "Unsupported Diffusers video pipeline"):
            generator.update_adapter_modes(
                {
                    "video_contract": {
                        "pipelineClass": "UnknownVideoPipeline",
                        "modes": ["text_to_video"],
                    }
                },
                None,
            )
        with self.assertRaisesRegex(ValueError, "stale or mismatched"):
            generator.update_adapter_modes(
                {
                    "video_contract": {
                        "pipelineClass": "HunyuanVideoFramepackPipeline",
                        "modes": ["text_to_video"],
                    }
                },
                None,
            )
        with self.assertRaisesRegex(ValueError, "stale or mismatched"):
            generator.update_adapter_modes(
                {
                    "video_contract": {
                        "schemaVersion": 999,
                        "library": "unreviewed",
                        "mediaKind": "audio",
                        "pipelineClass": "HunyuanVideoFramepackPipeline",
                        "modes": ["image_to_video"],
                    }
                },
                None,
            )
        generator.set_field_params.assert_not_called()

    def test_video_field_contracts_cover_every_adapter_mode_and_update_selected_fields(self):
        self.assertEqual(set(VIDEO_MODE_FIELD_CONTRACTS), set(VIDEO_PIPELINE_ADAPTERS))
        for pipeline_class, adapter in VIDEO_PIPELINE_ADAPTERS.items():
            with self.subTest(pipeline_class=pipeline_class):
                self.assertEqual(tuple(VIDEO_MODE_FIELD_CONTRACTS[pipeline_class]), adapter.modes)

        cases = (
            (
                "WanVACEPipeline",
                "video_inpaint",
                {
                    "video": {"hidden": False, "required": True},
                    "mask": {"hidden": False, "required": True},
                    "reference_images": {"hidden": False, "required": False},
                    "framepack_sampling": {"hidden": True},
                },
                "strength",
            ),
            (
                "LTXConditionPipeline",
                "video_to_video",
                {
                    "video": {"hidden": False, "required": True},
                    "reference_images": {"hidden": True, "required": False},
                    "strength": {"hidden": False},
                    "denoise_strength": {"hidden": False},
                },
                "conditioningScale",
            ),
            (
                "WanAnimatePipeline",
                "character_replace",
                {
                    "reference_images": {"hidden": False, "required": True},
                    "pose_video": {"hidden": False, "required": True},
                    "face_video": {"hidden": False, "required": True},
                    "background_video": {"hidden": False, "required": True},
                    "mask": {"hidden": False, "required": True},
                    "strength": {"hidden": True},
                },
                "strength",
            ),
        )
        for pipeline_class, mode, expected_fields, strength_form_field in cases:
            with self.subTest(pipeline_class=pipeline_class, mode=mode):
                adapter = VIDEO_PIPELINE_ADAPTERS[pipeline_class]
                node = Generate(f"field-contract-{pipeline_class}-{mode}")
                node.set_field_params = MagicMock()
                node.update_adapter_modes(
                    {
                        "mode": mode,
                        "video_contract": {
                            "schemaVersion": 1,
                            "library": "diffusers",
                            "mediaKind": "video",
                            "pipelineClass": pipeline_class,
                            "modes": list(adapter.modes),
                        },
                    },
                    {"key": "mode"},
                )
                updates = {call.args[0]: call.args[1] for call in node.set_field_params.call_args_list}
                for field, expected in expected_fields.items():
                    self.assertEqual(
                        {key: updates[field][key] for key in expected},
                        expected,
                    )
                strength_binding = updates["strength"]["fieldOptions"]["studioBinding"]
                self.assertEqual(strength_binding["formFields"], [strength_form_field])
                self.assertEqual(
                    strength_binding["group"],
                    "video-conditioning-scale" if strength_form_field == "conditioningScale" else "video-strength",
                )

    def test_wan_vace_declares_the_reviewed_mode_media_matrix(self):
        adapter = VIDEO_PIPELINE_ADAPTERS["WanVACEPipeline"]
        expected = {
            "text_to_video": ("forbidden", "forbidden", "forbidden"),
            "video_to_video": ("required", "forbidden", "optional"),
            "video_inpaint": ("required", "required", "optional"),
            "video_outpaint": ("required", "required", "optional"),
            "reference_to_video": ("forbidden", "forbidden", "required"),
            "control_to_video": ("required", "forbidden", "optional"),
            "video_color_edit": ("required", "forbidden", "optional"),
        }

        self.assertEqual(tuple(WAN_VACE_MODE_MEDIA_CONTRACTS), adapter.modes)
        self.assertEqual(set(WAN_VACE_MODE_MEDIA_CONTRACTS), set(expected))
        self.assertNotIn("image_to_video", adapter.modes)
        for mode, requirements in expected.items():
            with self.subTest(mode=mode):
                contract = WAN_VACE_MODE_MEDIA_CONTRACTS[mode]
                self.assertEqual((contract.video, contract.mask, contract.reference_images), requirements)

    def test_wan_vace_every_required_and_forbidden_media_rule_preflights_before_torch(self):
        pipeline = SimpleNamespace(
            _modiff_video_pipeline_class="WanVACEPipeline",
            vae_scale_factor_temporal=1,
        )
        media = {
            "video": [np.zeros((4, 4, 3), dtype=np.uint8)],
            "mask": [np.zeros((4, 4), dtype=np.uint8)],
            "reference_images": [Image.new("RGB", (4, 4))],
        }

        with patch.dict(sys.modules, {"torch": None}):
            for mode, contract in WAN_VACE_MODE_MEDIA_CONTRACTS.items():
                requirements = {
                    "video": contract.video,
                    "mask": contract.mask,
                    "reference_images": contract.reference_images,
                }
                valid = {
                    field: media[field] for field, requirement in requirements.items() if requirement == "required"
                }
                for field, requirement in requirements.items():
                    if requirement == "required":
                        values = {key: value for key, value in valid.items() if key != field}
                        node = Generate()
                        node._execute_wan_vace = MagicMock(side_effect=AssertionError("VACE handler must not run"))
                        with self.subTest(mode=mode, missing=field):
                            with self.assertRaisesRegex(ValueError, "requires"):
                                node.execute(pipeline=pipeline, mode=mode, num_frames=1, **values)
                            node._execute_wan_vace.assert_not_called()
                    elif requirement == "forbidden":
                        node = Generate()
                        node._execute_wan_vace = MagicMock(side_effect=AssertionError("VACE handler must not run"))
                        with self.subTest(mode=mode, forbidden=field):
                            with self.assertRaisesRegex(ValueError, "does not accept"):
                                node.execute(
                                    pipeline=pipeline,
                                    mode=mode,
                                    num_frames=1,
                                    **valid,
                                    **{field: media[field]},
                                )
                            node._execute_wan_vace.assert_not_called()

    def test_wan_vace_optional_references_and_empty_media_are_normalized_before_dispatch(self):
        pipeline = SimpleNamespace(
            _modiff_video_pipeline_class="WanVACEPipeline",
            vae_scale_factor_temporal=1,
        )
        video = [np.zeros((4, 4, 3), dtype=np.uint8)]
        mask = [np.zeros((4, 4), dtype=np.uint8)]
        reference = Image.new("RGB", (6, 5))
        output = {"video_out": [], "width_out": 4, "height_out": 4, "frames_out": 0}

        node = Generate()
        node._execute_wan_vace = MagicMock(return_value=output)
        for mode, contract in WAN_VACE_MODE_MEDIA_CONTRACTS.items():
            values = {"video": [], "mask": [], "reference_images": [], "num_frames": 1}
            if contract.video == "required":
                values["video"] = video
            if contract.mask == "required":
                values["mask"] = mask
            if contract.reference_images in {"required", "optional"}:
                values["reference_images"] = [reference]

            with self.subTest(mode=mode):
                node.execute(pipeline=pipeline, mode=mode, **values)
                dispatched = node._execute_wan_vace.call_args.args[3]
                if contract.video == "forbidden":
                    self.assertIsNone(dispatched["video"])
                else:
                    self.assertEqual(len(dispatched["video"]), 1)
                    self.assertIs(dispatched["video"][0], video[0])
                if contract.mask == "forbidden":
                    self.assertIsNone(dispatched["mask"])
                else:
                    self.assertEqual(len(dispatched["mask"]), 1)
                    self.assertIs(dispatched["mask"][0], mask[0])
                if contract.reference_images == "forbidden":
                    self.assertIsNone(dispatched["reference_images"])
                else:
                    self.assertEqual(dispatched["reference_images"], [reference])
                self.assertEqual(dispatched["num_frames"], 1)
                node._execute_wan_vace.reset_mock()

    def test_real_video_node_preserves_unknown_mode_for_an_actionable_failure(self):
        pipeline = SimpleNamespace(_modiff_video_pipeline_class="WanVACEPipeline")
        node = Generate("strict-video-mode")
        node._execute_wan_vace = MagicMock(side_effect=AssertionError("VACE handler must not run"))

        self.assertTrue(Generate.params["mode"]["fieldOptions"]["noValidation"])
        for value in (None, [], {}):
            with self.subTest(mode=value):
                with self.assertRaisesRegex(ValueError, "exact non-empty Diffusers video mode"):
                    node(pipeline=pipeline, mode=value)
        with self.assertRaisesRegex(ValueError, "exact non-empty Diffusers video mode"):
            node(pipeline=pipeline)
        with self.assertRaisesRegex(RuntimeError, "does not support video mode future_video_mode"):
            node(pipeline=pipeline, mode="future_video_mode")
        with self.assertRaisesRegex(RuntimeError, "does not support video mode  text_to_video "):
            node(pipeline=pipeline, mode=" text_to_video ")
        with self.assertRaisesRegex(RuntimeError, "does not support video mode image_to_video"):
            node(pipeline=pipeline, mode="image_to_video")
        node._execute_wan_vace.assert_not_called()

    def test_real_video_node_passes_normalized_media_to_the_wan_handler(self):
        pipeline = SimpleNamespace(
            _modiff_video_pipeline_class="WanVACEPipeline",
            vae_scale_factor_temporal=1,
        )
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        node = Generate("normalized-wan-media")
        node._execute_wan_vace = MagicMock(
            return_value={"video_out": [], "width_out": 4, "height_out": 4, "frames_out": 0}
        )

        result = node(
            pipeline=pipeline,
            mode="video_to_video",
            video=(frame,),
            mask=[],
            reference_images=[],
            num_frames=1,
        )

        self.assertEqual(result["frames_out"], 0)
        dispatched = node._execute_wan_vace.call_args.args[3]
        self.assertEqual(len(dispatched["video"]), 1)
        np.testing.assert_array_equal(dispatched["video"][0], frame)
        self.assertIsNone(dispatched["mask"])
        self.assertIsNone(dispatched["reference_images"])
        self.assertEqual(dispatched["num_frames"], 1)

    def test_wan_vace_media_types_shapes_counts_and_requested_length_preflight_before_torch(self):
        class TorchLikeFrame:
            __module__ = "torch"
            shape = (3, 4, 4)
            dtype = "float32"
            device = "cpu"

            def detach(self):
                return self

        class HWCTorchLikeFrame:
            __module__ = "torch"
            shape = (8, 8, 3)
            dtype = "float32"
            device = "cpu"

            def detach(self):
                return self

        pipeline = SimpleNamespace(
            _modiff_video_pipeline_class="WanVACEPipeline",
            vae_scale_factor_temporal=1,
        )
        output = {"video_out": [], "width_out": 4, "height_out": 4, "frames_out": 0}
        valid_frames = (
            Image.new("RGB", (4, 4)),
            np.zeros((4, 4, 3), dtype=np.uint8),
            TorchLikeFrame(),
        )

        with patch.dict(sys.modules, {"torch": None}):
            for frame in valid_frames:
                with self.subTest(valid_type=type(frame).__name__):
                    node = Generate()
                    node._execute_wan_vace = MagicMock(return_value=output)
                    node.execute(
                        pipeline=pipeline,
                        mode="video_to_video",
                        video=[frame],
                        num_frames=1,
                    )
                    node._execute_wan_vace.assert_called_once()

            reference = Image.new("RGB", (6, 5))
            for selection in (reference, [reference], [[reference]], ((reference,),)):
                with self.subTest(reference_shape=type(selection).__name__, selection=selection):
                    node = Generate()
                    node._execute_wan_vace = MagicMock(return_value=output)
                    node.execute(
                        pipeline=pipeline,
                        mode="reference_to_video",
                        reference_images=selection,
                        num_frames=1,
                    )
                    dispatched_references = node._execute_wan_vace.call_args.args[3]["reference_images"]
                    self.assertEqual(dispatched_references, [reference])

            frame4 = np.zeros((4, 4, 3), dtype=np.uint8)
            frame5 = np.zeros((5, 4, 3), dtype=np.uint8)
            mask4 = np.zeros((4, 4), dtype=np.uint8)
            mask5 = np.zeros((5, 4), dtype=np.uint8)
            cases = (
                (
                    {"mode": "video_to_video", "video": [object()], "num_frames": 1},
                    "must be a PIL image, NumPy array, or Torch tensor-like image",
                ),
                (
                    {
                        "mode": "video_to_video",
                        "video": [frame4],
                        "reference_images": [object()],
                        "num_frames": 1,
                    },
                    "reference images must be actual PIL images",
                ),
                (
                    {
                        "mode": "reference_to_video",
                        "reference_images": [np.zeros((4, 4, 3), dtype=np.uint8)],
                        "num_frames": 1,
                    },
                    "reference images must be actual PIL images",
                ),
                (
                    {
                        "mode": "reference_to_video",
                        "reference_images": [[Image.new("RGB", (4, 4))], [Image.new("RGB", (4, 4))]],
                        "num_frames": 1,
                    },
                    "one flat list or one nested batch",
                ),
                (
                    {"mode": "video_inpaint", "video": [frame4], "mask": [object()], "num_frames": 1},
                    "mask video frame 1 must be a PIL image",
                ),
                (
                    {
                        "mode": "video_inpaint",
                        "video": [frame4, frame4],
                        "mask": [mask4],
                        "num_frames": 2,
                    },
                    "video/mask frame count mismatch: 2 vs 1",
                ),
                (
                    {"mode": "video_inpaint", "video": [frame4], "mask": [mask5], "num_frames": 1},
                    "must have matching spatial dimensions",
                ),
                (
                    {
                        "mode": "video_inpaint",
                        "video": [frame4],
                        "mask": [Image.new("L", (4, 4))],
                        "num_frames": 1,
                    },
                    "must use the same container family",
                ),
                (
                    {
                        "mode": "video_inpaint",
                        "video": [Image.new("RGB", (4, 4))],
                        "mask": [mask4],
                        "num_frames": 1,
                    },
                    "must use the same container family",
                ),
                (
                    {
                        "mode": "video_inpaint",
                        "video": [np.zeros((8, 8), dtype=np.uint8)],
                        "mask": [np.zeros((8, 8, 1), dtype=np.uint8)],
                        "num_frames": 1,
                    },
                    "2D source/control video frames require 2D mask frames",
                ),
                (
                    {"mode": "video_to_video", "video": [frame4, frame5], "num_frames": 2},
                    "frames must all have the same spatial dimensions",
                ),
                (
                    {
                        "mode": "video_to_video",
                        "video": [Image.new("RGB", (4, 4)), frame4],
                        "num_frames": 2,
                    },
                    "source/control video frames must use one container family",
                ),
                (
                    {
                        "mode": "video_inpaint",
                        "video": [Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4))],
                        "mask": [Image.new("L", (4, 4)), mask4],
                        "num_frames": 2,
                    },
                    "mask video frames must use one container family",
                ),
                (
                    {
                        "mode": "reference_to_video",
                        "reference_images": [Image.new("RGB", (4, 4)) for _ in range(9)],
                        "num_frames": 1,
                    },
                    "accepts at most 8 reference images",
                ),
                (
                    {
                        "mode": "reference_to_video",
                        "reference_images": [Image.new("1", (4097, 4097))],
                        "num_frames": 1,
                    },
                    "16777216-pixel cumulative input limit",
                ),
                (
                    {
                        "mode": "video_to_video",
                        "video": [frame4] * 82,
                        "num_frames": 82,
                        "output_type": "np",
                    },
                    "conditioned video currently requires output_type=pil",
                ),
                (
                    {
                        "mode": "video_to_video",
                        "video": [frame4] * 82,
                        "num_frames": 82,
                        "output_type": "pt",
                    },
                    "conditioned video currently requires output_type=pil",
                ),
                (
                    {
                        "mode": "video_to_video",
                        "video": [np.zeros((3, 8, 8), dtype=np.uint8)],
                        "num_frames": 1,
                    },
                    "NumPy images must use HWC layout",
                ),
                (
                    {
                        "mode": "video_to_video",
                        "video": [HWCTorchLikeFrame()],
                        "num_frames": 1,
                    },
                    "Torch tensor-like images must use CHW layout",
                ),
                (
                    {"mode": "video_to_video", "video": [frame4], "num_frames": 2},
                    "received 1 conditioned video frames, but normalized num_frames is 2",
                ),
            )
            for values, message in cases:
                with self.subTest(message=message):
                    node = Generate()
                    node._execute_wan_vace = MagicMock(side_effect=AssertionError("VACE handler must not run"))
                    with self.assertRaisesRegex(ValueError, message):
                        node.execute(pipeline=pipeline, **values)
                    node._execute_wan_vace.assert_not_called()

    def test_wan_vace_numeric_resource_contract_preflights_before_torch_and_dispatch(self):
        pipeline = SimpleNamespace(
            _modiff_video_pipeline_class="WanVACEPipeline",
            vae_scale_factor_temporal=1,
            vae_scale_factor_spatial=8,
            transformer=SimpleNamespace(config=SimpleNamespace(patch_size=(1, 2, 2))),
        )
        invalid = (
            ("width", False, "width.*finite integer"),
            ("width", 15, "width.*16 through 2048"),
            ("width", 2049, "width.*16 through 2048"),
            ("width", 16.5, "width.*finite integer"),
            ("height", 15, "height.*16 through 2048"),
            ("height", 2049, "height.*16 through 2048"),
            ("height", 16.5, "height.*finite integer"),
            ("height", float("nan"), "height.*finite integer"),
            ("num_inference_steps", 0, "inference steps.*1 through 100"),
            ("num_inference_steps", 101, "inference steps.*1 through 100"),
            ("num_inference_steps", 1.5, "inference steps.*finite integer"),
            ("guidance_scale", -0.1, "guidance scale.*0 through 20"),
            ("guidance_scale", float("nan"), "guidance scale.*finite"),
            ("guidance_scale_2", 20.1, "secondary guidance scale.*0 through 20"),
            ("guidance_scale_2", float("inf"), "secondary guidance scale.*finite"),
            ("conditioning_scale", -0.1, "conditioning scale.*0 through 2"),
            ("conditioning_scale", float("nan"), "conditioning scale.*finite"),
            ("seed", -1, "seed.*0 through 4294967295"),
            ("seed", 4294967296, "seed.*0 through 4294967295"),
            ("seed", 1.5, "seed.*finite integer"),
            ("num_videos_per_prompt", 0, "videos per prompt.*1 through 1"),
            ("num_videos_per_prompt", 2, "videos per prompt.*1 through 1"),
            ("num_videos_per_prompt", True, "videos per prompt.*finite integer"),
            ("output_type", "tensor", "output_type must be exactly"),
            ("output_type", " pt ", "output_type must be exactly"),
            ("max_sequence_length", 0, "max sequence length.*1 through 512"),
            ("max_sequence_length", 513, "max sequence length.*1 through 512"),
            ("max_sequence_length", 1.5, "max sequence length.*finite integer"),
        )

        with patch.dict(sys.modules, {"torch": None}):
            for index, (field, value, message) in enumerate(invalid):
                node = Generate(f"strict-wan-scalar-{index}")
                node._execute_wan_vace = MagicMock(side_effect=AssertionError("VACE handler must not run"))
                values = {
                    "pipeline": pipeline,
                    "mode": "text_to_video",
                    "num_frames": 1,
                    "width": 16,
                    "height": 16,
                    field: value,
                }
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, message):
                        node(**values)
                node._execute_wan_vace.assert_not_called()

            misaligned = Generate("strict-wan-alignment")
            misaligned._execute_wan_vace = MagicMock(side_effect=AssertionError("VACE handler must not run"))
            with self.assertRaisesRegex(ValueError, "size must be divisible by 16x16"):
                misaligned(
                    pipeline=pipeline,
                    mode="text_to_video",
                    width=24,
                    height=16,
                    num_frames=1,
                )
            misaligned._execute_wan_vace.assert_not_called()

            boundary = Generate("strict-wan-scalar-boundaries")
            boundary._execute_wan_vace = MagicMock(
                return_value={"video_out": [], "width_out": 16, "height_out": 16, "frames_out": 0}
            )
            result = boundary(
                pipeline=pipeline,
                mode="text_to_video",
                width=16,
                height=16,
                num_frames=1,
                num_inference_steps=1,
                guidance_scale=0,
                guidance_scale_2=0,
                conditioning_scale=0,
                seed=4294967295,
                num_videos_per_prompt=1,
                output_type="pt",
                max_sequence_length=512,
            )

        self.assertEqual(result["frames_out"], 0)
        dispatched = boundary._execute_wan_vace.call_args.args[3]
        self.assertEqual(dispatched["width"], 16)
        self.assertEqual(dispatched["height"], 16)
        self.assertEqual(dispatched["num_inference_steps"], 1)
        self.assertEqual(dispatched["guidance_scale"], 0)
        self.assertEqual(dispatched["guidance_scale_2"], 0)
        self.assertEqual(dispatched["conditioning_scale"], 0)
        self.assertEqual(dispatched["seed"], 4294967295)
        self.assertEqual(dispatched["num_videos_per_prompt"], 1)
        self.assertEqual(dispatched["output_type"], "pt")
        self.assertEqual(dispatched["max_sequence_length"], 512)

    def test_real_wan_vace_node_rejects_invalid_reference_pairing_and_frame_bounds_before_dispatch(self):
        pipeline = SimpleNamespace(
            _modiff_video_pipeline_class="WanVACEPipeline",
            vae_scale_factor_temporal=4,
        )
        invalid_media = (
            {
                "mode": "reference_to_video",
                "reference_images": [np.zeros((4, 4, 3), dtype=np.uint8)],
                "num_frames": 1,
            },
            {
                "mode": "video_inpaint",
                "video": [np.zeros((4, 4, 3), dtype=np.uint8)],
                "mask": [Image.new("L", (4, 4))],
                "num_frames": 1,
            },
        )
        with patch.dict(sys.modules, {"torch": None}):
            for index, values in enumerate(invalid_media):
                with self.subTest(values=values):
                    node = Generate(f"strict-wan-media-{index}")
                    node._execute_wan_vace = MagicMock(side_effect=AssertionError("VACE handler must not run"))
                    with self.assertRaisesRegex(RuntimeError, "actual PIL images|same container family"):
                        node(pipeline=pipeline, **values)
                    node._execute_wan_vace.assert_not_called()

            invalid_frame_counts = (False, 0, -1, 1.5, "1.5", 242, float("inf"), float("nan"), object())
            for index, num_frames in enumerate(invalid_frame_counts):
                with self.subTest(num_frames=num_frames):
                    node = Generate(f"strict-wan-frames-{index}")
                    node._execute_wan_vace = MagicMock(side_effect=AssertionError("VACE handler must not run"))
                    with self.assertRaisesRegex(ValueError, "finite integer from 1 through 241"):
                        node(pipeline=pipeline, mode="text_to_video", num_frames=num_frames)
                    node._execute_wan_vace.assert_not_called()

            node = Generate("strict-wan-frames-boundary")
            node._execute_wan_vace = MagicMock(
                return_value={
                    "video_out": [],
                    "width_out": 4,
                    "height_out": 4,
                    "frames_out": 0,
                }
            )
            result = node(pipeline=pipeline, mode="text_to_video", num_frames=241.0)
            self.assertEqual(result["frames_out"], 0)
            self.assertEqual(node._execute_wan_vace.call_args.args[3]["num_frames"], 241)

    def test_invalid_non_vace_media_contract_fails_before_torch_import(self):
        pipeline = SimpleNamespace(_modiff_video_pipeline_class="WanPipeline")
        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaisesRegex(ValueError, "text_to_video does not accept a source video"):
                Generate().execute(pipeline=pipeline, mode="text_to_video", video=[object()])

    def test_sequence_generator_reuses_one_generic_pipeline_for_multiple_shots(self):
        class FakePipeline:
            _modiff_video_pipeline_class = "LTXConditionPipeline"

        outputs = [
            {"video_out": ["a1", "a2"], "frames_out": 2},
            {"video_out": ["b1", "b2"], "frames_out": 2},
        ]
        with patch.object(Generate, "execute", side_effect=outputs) as execute:
            result = GenerateSequence().execute(
                pipeline=FakePipeline(),
                prompts_json='["First moving shot", "Second moving shot"]',
                seed=100,
            )
        self.assertEqual(result["clips"], [["a1", "a2"], ["b1", "b2"]])
        self.assertEqual(result["video_out"], ["a1", "a2", "b1", "b2"])
        self.assertEqual(result["frames_out"], 4)
        self.assertEqual(result["total_frames"], 4)
        self.assertEqual(execute.call_args_list[0].kwargs["seed"], 100)
        self.assertEqual(execute.call_args_list[1].kwargs["seed"], 101)

    def test_sequence_generator_accepts_explicit_model_neutral_shot_seeds(self):
        class FakePipeline:
            _modiff_video_pipeline_class = "LTXConditionPipeline"

        outputs = [
            {"video_out": ["b1"], "frames_out": 1},
            {"video_out": ["a1"], "frames_out": 1},
        ]
        with patch.object(Generate, "execute", side_effect=outputs) as execute:
            result = GenerateSequence().execute(
                pipeline=FakePipeline(),
                prompts_json='[{"prompt":"Second visual first","seed":101},{"prompt":"First visual second","seed":100}]',
                seed=900,
            )
        self.assertEqual(result["video_out"], ["b1", "a1"])
        self.assertEqual(execute.call_args_list[0].kwargs["seed"], 101)
        self.assertEqual(execute.call_args_list[1].kwargs["seed"], 100)

    def test_wan_vace_legacy_adapter_covers_all_studio_modes(self):
        adapter = get_video_pipeline_adapter("WanVACEPipeline")
        self.assertEqual(adapter.default_repo, "Wan-AI/Wan2.1-VACE-1.3B-diffusers")
        self.assertIn("video_inpaint", adapter.modes)
        self.assertIn("reference_to_video", adapter.modes)

    def test_wan_22_image_to_video_uses_quality_first_five_second_contract(self):
        adapter = get_video_pipeline_adapter("WanImageToVideoPipeline")
        self.assertEqual(adapter.default_repo, "Wan-AI/Wan2.2-I2V-A14B-Diffusers")
        self.assertEqual(adapter.modes, ("image_to_video",))

        class Output:
            frames = [["frame-a", "frame-b"]]

        class FakePipeline:
            _modiff_video_pipeline_class = "WanImageToVideoPipeline"
            _execution_device = "cpu"
            vae_scale_factor_spatial = 8
            transformer = SimpleNamespace(config=SimpleNamespace(patch_size=(1, 2, 2)))

            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return Output()

        pipeline = FakePipeline()
        opening = object()
        ending = object()
        result = Generate().execute(
            pipeline=pipeline,
            mode="image_to_video",
            reference_images=[opening],
            last_image=ending,
            prompt="A cinematic realistic rescue unfolds in one controlled shot.",
            negative_prompt="low quality, deformed anatomy, flicker",
            width=832,
            height=480,
            num_frames=81,
            num_inference_steps=40,
            guidance_scale=3.5,
            secondary_guidance_scale=3.25,
            seed=12,
        )

        call = pipeline.calls[0]
        self.assertEqual(result["frames_out"], 2)
        self.assertIs(call["image"], opening)
        self.assertIs(call["last_image"], ending)
        self.assertEqual(call["num_frames"], 81)
        self.assertEqual(call["num_inference_steps"], 40)
        self.assertEqual(call["guidance_scale"], 3.5)
        self.assertEqual(call["guidance_scale_2"], 3.25)

    def test_wan_22_ti2v_5b_uses_official_five_second_defaults(self):
        adapter = get_video_pipeline_adapter("WanTI2VPipeline")
        self.assertEqual(adapter.default_repo, "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
        self.assertEqual(adapter.modes, ("text_to_video",))

        class Output:
            frames = [["frame"]]

        class FakePipeline:
            _modiff_video_pipeline_class = "WanTI2VPipeline"
            _execution_device = "cpu"

            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return Output()

        pipeline = FakePipeline()
        result = Generate().execute(pipeline=pipeline, mode="text_to_video", prompt="A cinematic seed-vault rescue.")

        call = pipeline.calls[0]
        self.assertEqual(result["frames_out"], 1)
        self.assertEqual(call["width"], 1280)
        self.assertEqual(call["height"], 704)
        self.assertEqual(call["num_frames"], 121)
        self.assertEqual(call["num_inference_steps"], 50)

    def test_generic_video_seed_supports_published_large_seed_examples(self):
        self.assertGreaterEqual(Generate.params["seed"]["max"], 898471028164125)

    def test_wan_22_ti2v_5b_loader_preserves_native_flow_shift(self):
        adapter = get_video_pipeline_adapter("WanTI2VPipeline")
        vae = object()
        scheduler = SimpleNamespace(config={"flow_shift": 5.0})
        pipeline = SimpleNamespace(scheduler=scheduler)
        node = LoadPipeline()

        with (
            patch("diffusers.AutoencoderKLWan.from_pretrained", return_value=vae),
            patch("diffusers.WanPipeline.from_pretrained", return_value=pipeline),
            patch(
                "diffusers.schedulers.scheduling_unipc_multistep.UniPCMultistepScheduler.from_config"
            ) as replace_scheduler,
            patch("modules.DiffusersRuntime.main.apply_execution_recipe_to_pipeline"),
            patch.object(node, "progress"),
            patch.object(node, "mm_add"),
            patch("modules.DiffusersVideo.main.apply_pipeline_offload"),
        ):
            loaded = node._load_wan_text_to_video(
                adapter,
                {
                    "model_id": {"source": "hub", "value": adapter.default_repo},
                    "dtype": "bfloat16",
                    "device": "cpu",
                    "execution_recipe": {"offload_mode": "none", "device": "cpu"},
                },
            )

        self.assertIs(loaded, pipeline)
        self.assertIs(loaded.scheduler, scheduler)
        self.assertEqual(loaded.scheduler.config["flow_shift"], 5.0)
        replace_scheduler.assert_not_called()

    def test_wan_text_generation_can_override_flow_shift_for_a_locked_recipe(self):
        original_scheduler = SimpleNamespace(config={"flow_shift": 5.0})
        replacement_scheduler = SimpleNamespace(config={"flow_shift": 8.0})

        class Output:
            frames = [["frame"]]

        class FakePipeline:
            _modiff_video_pipeline_class = "WanTI2VPipeline"
            _execution_device = "cpu"

            def __init__(self):
                self.scheduler = original_scheduler

            def __call__(self, **_kwargs):
                self.scheduler_during_call = self.scheduler
                return Output()

        pipeline = FakePipeline()
        with patch(
            "diffusers.UniPCMultistepScheduler.from_config", return_value=replacement_scheduler
        ) as replace_scheduler:
            Generate().execute(
                pipeline=pipeline,
                mode="text_to_video",
                prompt="A street musician in a subway station.",
                scheduler_flow_shift=8,
            )

        replace_scheduler.assert_called_once_with(original_scheduler.config, flow_shift=8.0)
        self.assertIs(pipeline.scheduler_during_call, replacement_scheduler)
        self.assertIs(pipeline.scheduler, original_scheduler)

    def test_wan_21_t2v_13b_loader_uses_upstream_quality_flow_shift(self):
        adapter = get_video_pipeline_adapter("WanPipeline")
        vae = object()
        scheduler = SimpleNamespace(config={"flow_shift": 3.0})
        replacement_scheduler = SimpleNamespace(config={"flow_shift": 8.0})
        pipeline = SimpleNamespace(scheduler=scheduler)
        node = LoadPipeline()

        with (
            patch("diffusers.AutoencoderKLWan.from_pretrained", return_value=vae),
            patch("diffusers.WanPipeline.from_pretrained", return_value=pipeline),
            patch(
                "diffusers.schedulers.scheduling_unipc_multistep.UniPCMultistepScheduler.from_config",
                return_value=replacement_scheduler,
            ) as replace_scheduler,
            patch("modules.DiffusersRuntime.main.apply_execution_recipe_to_pipeline"),
            patch.object(node, "progress"),
            patch.object(node, "mm_add"),
            patch("modules.DiffusersVideo.main.apply_pipeline_offload"),
        ):
            loaded = node._load_wan_text_to_video(
                adapter,
                {
                    "model_id": {"source": "hub", "value": adapter.default_repo},
                    "dtype": "bfloat16",
                    "device": "cpu",
                    "execution_recipe": {"offload_mode": "none", "device": "cpu"},
                },
            )

        self.assertIs(loaded.scheduler, replacement_scheduler)
        replace_scheduler.assert_called_once_with(scheduler.config, flow_shift=8.0)

    def test_wan_22_quality_contract_rejects_shorter_than_five_second_clips(self):
        class FakePipeline:
            _modiff_video_pipeline_class = "WanImageToVideoPipeline"
            _execution_device = "cpu"
            vae_scale_factor_spatial = 8
            transformer = SimpleNamespace(config=SimpleNamespace(patch_size=(1, 2, 2)))

            def __call__(self, **_kwargs):
                raise AssertionError("short quality request must not execute")

        with self.assertRaisesRegex(ValueError, "at least 81 frames"):
            Generate().execute(
                pipeline=FakePipeline(),
                mode="image_to_video",
                reference_images=[object()],
                prompt="A realistic shot.",
                width=832,
                height=480,
                num_frames=65,
            )

    def test_wan_22_image_to_video_loader_preserves_fp32_vae_and_recipe(self):
        adapter = get_video_pipeline_adapter("WanImageToVideoPipeline")
        vae = object()
        pipeline = SimpleNamespace()
        node = LoadPipeline()

        with (
            patch("diffusers.AutoencoderKLWan.from_pretrained", return_value=vae) as load_vae,
            patch("diffusers.WanImageToVideoPipeline.from_pretrained", return_value=pipeline) as load_pipeline,
            patch("modules.DiffusersRuntime.main.apply_execution_recipe_to_pipeline") as apply_recipe,
            patch.object(node, "progress"),
            patch.object(node, "mm_add"),
            patch("modules.DiffusersVideo.main.apply_pipeline_offload"),
        ):
            loaded = node._load_wan_image_to_video(
                adapter,
                {
                    "model_id": {"source": "hub", "value": adapter.default_repo},
                    "dtype": "bfloat16",
                    "device": "cpu",
                    "execution_recipe": {
                        "offload_mode": "model_cpu",
                        "device": "cpu",
                    },
                },
            )

        self.assertIs(loaded, pipeline)
        self.assertEqual(load_vae.call_args.args[0], adapter.default_repo)
        self.assertEqual(str(load_vae.call_args.kwargs["torch_dtype"]), "torch.float32")
        self.assertIs(load_pipeline.call_args.kwargs["vae"], vae)
        self.assertNotIn("quantization_config", load_pipeline.call_args.kwargs)
        apply_recipe.assert_called_once()

    def test_ltx_adapter_exposes_only_the_conditioning_modes_supported_by_diffusers(self):
        adapter = get_video_pipeline_adapter("LTXConditionPipeline")
        self.assertEqual(adapter.default_repo, "Lightricks/LTX-Video-0.9.8-13B-distilled")
        self.assertEqual(
            adapter.modes,
            ("text_to_video", "image_to_video", "video_to_video", "reference_to_video"),
        )
        self.assertEqual(adapter.max_prompt_tokens, 128)
        self.assertIn("LTXConditionPipeline", LoadPipeline.params["pipeline_class"]["options"])

    def test_no_offload_ltx_pipeline_loads_directly_on_cuda(self):
        adapter = get_video_pipeline_adapter("LTXConditionPipeline")
        pipeline = SimpleNamespace()
        node = LoadPipeline("ltx-direct-load-test")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch("diffusers.LTXConditionPipeline.from_pretrained", return_value=pipeline) as load_pipeline,
            patch("modules.DiffusersRuntime.main.apply_execution_recipe_to_pipeline"),
            patch("modules.DiffusersVideo.main.apply_pipeline_offload"),
        ):
            loaded = node._load_ltx(
                adapter,
                {
                    "model_id": adapter.default_repo,
                    "device": "cuda:0",
                    "auto_offload": False,
                    "offload_mode": "none",
                },
            )

        self.assertIs(loaded, pipeline)
        self.assertEqual(load_pipeline.call_args.kwargs["device_map"], "cuda")

    def test_ltx_long_adapter_uses_native_sliding_windows_instead_of_independent_clip_chaining(self):
        class Output:
            frames = [["frame-a", "frame-b"]]

        class FakePipeline:
            _modiff_video_pipeline_class = "LTXI2VLongMultiPromptPipeline"
            _modiff_video_repo = "Lightricks/LTX-Video-0.9.8-13B-distilled"

            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return Output()

        pipeline = FakePipeline()
        opening = object()
        result = Generate().execute(
            pipeline=pipeline,
            mode="image_to_video",
            reference_images=[opening],
            prompt="A calm animated lighthouse shot with coherent geometry.",
            width=832,
            height=480,
            num_frames=750,
            frame_rate=25,
            num_inference_steps=8,
            guidance_scale=1,
            temporal_tile_size=80,
            temporal_overlap=24,
            seed=42,
        )

        call = pipeline.calls[0]
        self.assertEqual(result["frames_out"], 2)
        self.assertIs(call["cond_image"], opening)
        self.assertEqual(call["num_frames"], 753)
        self.assertEqual(call["temporal_tile_size"], 80)
        self.assertEqual(call["temporal_overlap"], 24)
        self.assertEqual(call["decode_timestep"], 0.05)
        self.assertEqual(call["decode_noise_scale"], 0.025)
        self.assertEqual(call["callback_on_step_end_tensor_inputs"], [])

    def test_ltx_long_rejects_invalid_overlap_before_execution(self):
        class FakePipeline:
            _modiff_video_pipeline_class = "LTXI2VLongMultiPromptPipeline"
            _modiff_video_repo = "Lightricks/LTX-Video-0.9.8-13B-distilled"

            def __call__(self, **_kwargs):
                raise AssertionError("pipeline must not run")

        with self.assertRaisesRegex(ValueError, "overlap must be smaller"):
            Generate().execute(
                pipeline=FakePipeline(),
                mode="image_to_video",
                reference_images=[object()],
                prompt="A stable animated shot.",
                width=832,
                height=480,
                temporal_tile_size=80,
                temporal_overlap=80,
                num_inference_steps=8,
                guidance_scale=1,
            )

    def test_capable_pipeline_video_audio_contract_returns_both_modalities(self):
        class Output:
            frames = [["frame-a", "frame-b"]]
            audios = np.zeros((1, 2, 240), dtype=np.float32)

        class FakePipeline:
            _modiff_video_pipeline_class = "LTX2ConditionPipeline"
            _execution_device = "cpu"
            vocoder = SimpleNamespace(config={"output_sampling_rate": 24000})

            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return Output()

        pipeline = FakePipeline()
        condition_module = SimpleNamespace(LTX2VideoCondition=FakeLTX2VideoCondition)
        with patch.dict(
            sys.modules,
            {"diffusers.pipelines.ltx2.pipeline_ltx2_condition": condition_module},
        ):
            result = GenerateVideoAudio().execute(
                pipeline=pipeline,
                mode="image_to_video",
                reference_images=[object()],
                prompt="A continuous walking shot with natural synchronized ambience",
                width=768,
                height=512,
                num_frames=121,
                frame_rate=24,
            )

        self.assertEqual(result["frames_out"], 2)
        self.assertEqual(result["sample_rate_out"], 24000)
        self.assertEqual(result["audio"]["channels"], 2)
        self.assertEqual(len(pipeline.calls[0]["conditions"]), 1)

    def test_video_audio_node_rejects_video_only_pipeline_before_generation(self):
        class FakePipeline:
            _modiff_video_pipeline_class = "WanPipeline"

            def __call__(self, **_kwargs):
                raise AssertionError("video-only pipeline must not run")

        with self.assertRaisesRegex(ValueError, "does not produce synchronized audio"):
            GenerateVideoAudio().execute(pipeline=FakePipeline(), mode="text_to_video", prompt="A quiet street.")

    def test_legacy_ltx2_action_uses_generic_video_audio_contract_but_is_hidden(self):
        self.assertTrue(issubclass(GenerateLTX2, GenerateVideoAudio))
        self.assertTrue(module_registry.MODULE_MAP["modules.DiffusersVideo"]["GenerateLTX2"]["hidden"])
        self.assertIn("GenerateVideoAudio", module_registry.MODULE_MAP["modules.DiffusersVideo"])

    def test_wan_animate_requires_aligned_pose_and_face_controls(self):
        class Output:
            frames = [["animated-a", "animated-b"]]

        class FakePipeline:
            _modiff_video_pipeline_class = "WanAnimatePipeline"
            _execution_device = "cpu"

            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return Output()

        pipeline = FakePipeline()
        result = Generate().execute(
            pipeline=pipeline,
            mode="character_animate",
            reference_images=[object()],
            pose_video=["pose-a", "pose-b"],
            face_video=["face-a", "face-b"],
            width=1280,
            height=720,
        )

        self.assertEqual(result["frames_out"], 2)
        self.assertEqual(pipeline.calls[0]["mode"], "animate")
        with self.assertRaisesRegex(ValueError, "same number of frames"):
            Generate().execute(
                pipeline=pipeline,
                mode="character_animate",
                reference_images=[object()],
                pose_video=["pose-a"],
                face_video=["face-a", "face-b"],
            )

    def test_framepack_adapter_exposes_continuous_image_to_video_contract(self):
        adapter = get_video_pipeline_adapter("HunyuanVideoFramepackPipeline")
        self.assertEqual(adapter.default_repo, "lllyasviel/FramePackI2V_HY")
        self.assertEqual(adapter.modes, ("image_to_video",))
        self.assertIn("last_image", Generate.params)

        class Output:
            frames = [["frame-a", "frame-b"]]

        class FakePipeline:
            _modiff_video_pipeline_class = "HunyuanVideoFramepackPipeline"
            _execution_device = "cpu"

            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return Output()

        pipeline = FakePipeline()
        first = object()
        last = object()
        result = Generate().execute(
            pipeline=pipeline,
            mode="image_to_video",
            reference_images=[first],
            last_image=last,
            prompt="The camera follows the subject through a continuous action.",
            width=512,
            height=320,
            num_frames=481,
            num_inference_steps=4,
            framepack_sampling="inverted_anti_drifting",
            latent_window_size=9,
            seed=11,
        )

        self.assertEqual(result["frames_out"], 2)
        self.assertIs(pipeline.calls[0]["image"], first)
        self.assertIs(pipeline.calls[0]["last_image"], last)
        self.assertEqual(pipeline.calls[0]["num_frames"], 481)
        self.assertEqual(pipeline.calls[0]["sampling_type"], "inverted_anti_drifting")

    def test_adapter_replaces_any_managed_default_but_preserves_custom_hub_and_local_selections(self):
        adapter = get_video_pipeline_adapter("HunyuanVideoFramepackPipeline")
        inherited = {"source": "hub", "value": "Wan-AI/Wan2.1-VACE-1.3B-diffusers"}
        self.assertEqual(
            _resolve_adapter_model_selection(adapter, inherited),
            {"source": "hub", "value": "lllyasviel/FramePackI2V_HY"},
        )

        other_managed = {"source": "hub", "value": "Lightricks/LTX-2"}
        self.assertEqual(
            _resolve_adapter_model_selection(adapter, other_managed),
            {"source": "hub", "value": "lllyasviel/FramePackI2V_HY"},
        )

        same_managed_with_noncanonical_case = {"source": "HUB", "value": "LLLYASVIEL/FRAMEPACKI2V_HY"}
        self.assertEqual(
            _resolve_adapter_model_selection(adapter, same_managed_with_noncanonical_case),
            {"source": "hub", "value": "lllyasviel/FramePackI2V_HY"},
        )

        custom_hub = {"source": "Hub", "value": "organization/custom-framepack"}
        self.assertEqual(
            _resolve_adapter_model_selection(adapter, custom_hub),
            {"source": "hub", "value": "organization/custom-framepack"},
        )

        with tempfile.TemporaryDirectory() as temporary:
            explicit_path = Path(temporary) / "models" / "custom-framepack"
            explicit_path.mkdir(parents=True)
            managed_path = Path(temporary) / "Lightricks" / "LTX-2"
            managed_path.mkdir(parents=True)
            explicit = {"source": "LOCAL", "value": str(explicit_path)}
            self.assertEqual(
                _resolve_adapter_model_selection(adapter, explicit),
                {"source": "local", "value": str(explicit_path.resolve())},
            )

            with chdir(temporary):
                local_managed_name = {"source": "Local", "value": "Lightricks/LTX-2"}
                self.assertEqual(
                    _resolve_adapter_model_selection(adapter, local_managed_name),
                    {"source": "local", "value": str(managed_path.resolve())},
                )

                for invalid in ("organization/not-a-local-model", str(Path(temporary) / "missing")):
                    with self.subTest(invalid=invalid):
                        with self.assertRaisesRegex(ValueError, "directory does not exist"):
                            _resolve_adapter_model_selection(
                                adapter,
                                {"source": "local", "value": invalid},
                            )

                        node = LoadPipeline("missing-local-video-boundary")
                        node._load_framepack = MagicMock(side_effect=AssertionError("upstream must not run"))
                        with self.assertRaisesRegex(ValueError, "directory does not exist"):
                            node(
                                pipeline_class="HunyuanVideoFramepackPipeline",
                                model_id={"source": "local", "value": invalid},
                                revision="main",
                            )
                        node._load_framepack.assert_not_called()
        self.assertEqual(
            _resolve_adapter_model_selection(adapter, "organization/legacy-framepack"),
            {"source": "hub", "value": "organization/legacy-framepack"},
        )

    def test_video_model_selection_defaults_and_source_boundary_fail_before_revision_or_loader(self):
        adapter = get_video_pipeline_adapter("HunyuanVideoFramepackPipeline")
        expected_default = {"source": "hub", "value": adapter.default_repo}
        for selection in (None, "", "   ", {"source": "Hub", "value": ""}):
            with self.subTest(default_selection=selection):
                self.assertEqual(_resolve_adapter_model_selection(adapter, selection), expected_default)

        invalid_selections = (
            {"value": "organization/model"},
            {"source": None, "value": "organization/model"},
            {"source": "", "value": "organization/model"},
            {"source": " hub ", "value": "organization/model"},
            {"source": "remote", "value": "organization/model"},
            {"source": [], "value": "organization/model"},
            {"source": {}, "value": "organization/model"},
            {"source": "hub"},
            {"source": "hub", "value": []},
            {"source": "local", "value": ""},
            {"source": "local", "value": "   "},
            [],
            (),
        )
        node = LoadPipeline("strict-video-model-selection")
        node._load_framepack = MagicMock(side_effect=AssertionError("FramePack loader must not run"))
        with patch("modules.DiffusersVideo.main.catalog_revision") as resolve_revision:
            for selection in invalid_selections:
                with self.subTest(invalid_selection=selection):
                    with self.assertRaisesRegex(
                        ValueError,
                        "source must be exactly hub or local|value must be a repository ID|local Diffusers video "
                        "model path is required|model selection must be a repository ID",
                    ):
                        node(
                            pipeline_class="HunyuanVideoFramepackPipeline",
                            model_id=selection,
                        )
            resolve_revision.assert_not_called()
        node._load_framepack.assert_not_called()

    def test_video_hub_source_cannot_resolve_as_a_local_directory(self):
        revision = "0123456789abcdef0123456789abcdef01234567"
        node = LoadPipeline("video-hub-local-path-boundary")
        node._load_ltx = MagicMock(side_effect=AssertionError("upstream must not run"))

        with tempfile.TemporaryDirectory() as temporary:
            local_repo = Path(temporary) / "organization" / "local-video"
            local_repo.mkdir(parents=True)
            with chdir(temporary):
                invalid_hub_values = (
                    "organization/local-video",
                    str(local_repo),
                    local_repo.as_uri(),
                    "../local-video",
                    "single-component",
                )
                for value in invalid_hub_values:
                    with self.subTest(value=value):
                        with self.assertRaisesRegex(ValueError, "namespace/repository|local filesystem"):
                            node(
                                pipeline_class="LTXConditionPipeline",
                                model_id={"source": "hub", "value": value},
                                revision=revision,
                            )

        node._load_ltx.assert_not_called()

    def test_default_video_hub_model_cannot_resolve_as_a_local_directory(self):
        adapter = get_video_pipeline_adapter("LTXConditionPipeline")
        node = LoadPipeline("video-default-hub-local-path-boundary")
        node._load_ltx = MagicMock(side_effect=AssertionError("upstream must not run"))

        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / adapter.default_repo).mkdir(parents=True)
            with chdir(temporary):
                for selection in (None, "", {"source": "hub", "value": ""}):
                    with self.subTest(selection=selection):
                        with self.assertRaisesRegex(ValueError, "local filesystem"):
                            node(
                                pipeline_class="LTXConditionPipeline",
                                model_id=selection,
                            )

        node._load_ltx.assert_not_called()

    def test_local_video_model_is_canonical_before_cache_and_drops_hub_revision(self):
        pipeline = SimpleNamespace()
        node = LoadPipeline("canonical-local-video-cache")
        node._load_ltx = MagicMock(return_value=pipeline)

        with tempfile.TemporaryDirectory() as temporary:
            local_model = Path(temporary) / "models" / "local-video"
            local_model.mkdir(parents=True)
            with chdir(temporary), patch("modiff.NodeBase.modelstore.is_local_cached", return_value=True):
                first = node(
                    pipeline_class="LTXConditionPipeline",
                    model_id={"source": "local", "value": "models/local-video"},
                    revision="main",
                )
                second = node(
                    pipeline_class="LTXConditionPipeline",
                    model_id={"source": "local", "value": str(local_model)},
                    revision=None,
                )

        self.assertIs(first, second)
        node._load_ltx.assert_called_once()
        dispatched = node._load_ltx.call_args.args[1]
        self.assertEqual(
            dispatched["model_id"],
            {"source": "local", "value": str(local_model.resolve())},
        )
        self.assertIsNone(dispatched["revision"])
        self.assertIsNone(pipeline._modiff_video_revision)

    def test_curated_video_hub_default_keeps_catalog_revision_while_local_selection_does_not(self):
        adapter = get_video_pipeline_adapter("HunyuanVideoFramepackPipeline")
        default_selection = _resolve_adapter_model_selection(
            adapter,
            {"source": "Hub", "value": ""},
        )
        self.assertEqual(default_selection, {"source": "hub", "value": adapter.default_repo})
        self.assertEqual(
            _resolve_loader_revision(default_selection, adapter.default_repo, None),
            "86cef4396041b6002c957852daac4c91aaa47c79",
        )

        with tempfile.TemporaryDirectory() as temporary:
            local_model = Path(temporary) / adapter.default_repo
            local_model.mkdir(parents=True)
            with chdir(temporary):
                local_selection = _resolve_adapter_model_selection(
                    adapter,
                    {"source": "LOCAL", "value": adapter.default_repo},
                )
        self.assertEqual(local_selection, {"source": "local", "value": str(local_model.resolve())})
        self.assertIsNone(_resolve_loader_revision(local_selection, local_selection["value"], None))
        self.assertIsNone(_resolve_loader_revision(local_selection, local_selection["value"], "main"))

    def test_framepack_loader_composes_the_official_transformer_base_and_vision_repositories(self):
        adapter = get_video_pipeline_adapter("HunyuanVideoFramepackPipeline")
        try:
            import transformers  # noqa: F401
        except ModuleNotFoundError:
            pass
        else:
            import diffusers

            # Resolve Diffusers' lazy FramePack exports before replacing the
            # optional dependency with the bounded loader fixture below.
            diffusers.HunyuanVideoFramepackPipeline
            diffusers.HunyuanVideoFramepackTransformer3DModel
        transformer = object()
        feature_extractor = object()
        image_encoder = object()
        pipeline = object()
        node = LoadPipeline()
        load_processor = MagicMock(return_value=feature_extractor)
        load_encoder = MagicMock(return_value=image_encoder)
        transformers_module = ModuleType("transformers")
        transformers_module.SiglipImageProcessor = SimpleNamespace(from_pretrained=load_processor)
        transformers_module.SiglipVisionModel = SimpleNamespace(from_pretrained=load_encoder)

        with (
            patch.dict(sys.modules, {"transformers": transformers_module}),
            patch(
                "diffusers.HunyuanVideoFramepackTransformer3DModel.from_pretrained", return_value=transformer
            ) as load_transformer,
            patch("diffusers.HunyuanVideoFramepackPipeline.from_pretrained", return_value=pipeline) as load_pipeline,
            patch.object(node, "progress"),
            patch.object(node, "mm_add"),
            patch("modules.DiffusersVideo.main.apply_pipeline_offload"),
        ):
            loaded = node._load_framepack(
                adapter,
                {
                    "model_id": {"source": "hub", "value": adapter.default_repo},
                    "dtype": "bfloat16",
                    "device": "cpu",
                    "auto_offload": False,
                    "offload_mode": "none",
                },
            )

        self.assertIs(loaded, pipeline)
        self.assertEqual(load_transformer.call_args.args[0], adapter.default_repo)
        self.assertEqual(
            load_transformer.call_args.kwargs["revision"],
            "86cef4396041b6002c957852daac4c91aaa47c79",
        )
        self.assertEqual(load_processor.call_args.args[0], FRAMEPACK_VISION_REPO)
        self.assertEqual(
            load_processor.call_args.kwargs["revision"],
            "45b801affc54ff2af4e5daf1b282e0921901db87",
        )
        self.assertEqual(load_encoder.call_args.args[0], FRAMEPACK_VISION_REPO)
        self.assertEqual(
            load_encoder.call_args.kwargs["revision"],
            "45b801affc54ff2af4e5daf1b282e0921901db87",
        )
        self.assertEqual(load_pipeline.call_args.args[0], FRAMEPACK_BASE_REPO)
        self.assertEqual(
            load_pipeline.call_args.kwargs["revision"],
            "e8c2aaa66fe3742a32c11a6766aecbf07c56e773",
        )
        self.assertIs(load_pipeline.call_args.kwargs["transformer"], transformer)
        self.assertIs(load_pipeline.call_args.kwargs["feature_extractor"], feature_extractor)
        self.assertIs(load_pipeline.call_args.kwargs["image_encoder"], image_encoder)

    def test_framepack_rejects_last_image_for_vanilla_sampling(self):
        class FakePipeline:
            _modiff_video_pipeline_class = "HunyuanVideoFramepackPipeline"
            _execution_device = "cpu"

            def __call__(self, **_kwargs):
                raise AssertionError("invalid FramePack request must not run")

        with self.assertRaisesRegex(ValueError, "requires inverted_anti_drifting"):
            Generate().execute(
                pipeline=FakePipeline(),
                mode="image_to_video",
                reference_images=[object()],
                last_image=object(),
                framepack_sampling="vanilla",
            )

    def test_ltx_rejects_overlong_prompts_before_pipeline_execution(self):
        class FakeTokenizer:
            def __call__(self, _text, **_kwargs):
                return {"input_ids": list(range(129))}

        class FakePipeline:
            _modiff_video_pipeline_class = "LTXConditionPipeline"
            _execution_device = "cpu"
            tokenizer = FakeTokenizer()

            def __call__(self, **_kwargs):
                raise AssertionError("pipeline must not run for an overlong prompt")

        with self.assertRaisesRegex(ValueError, "uses 129 tokens.*at most 128"):
            Generate().execute(
                pipeline=FakePipeline(),
                mode="text_to_video",
                prompt="overlong",
                width=704,
                height=480,
            )

    def test_wan_video_to_video_adapter_uses_the_strength_capable_pipeline(self):
        adapter = get_video_pipeline_adapter("WanVideoToVideoPipeline")
        self.assertEqual(adapter.default_repo, "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        self.assertEqual(adapter.modes, ("video_to_video", "video_color_edit"))

        class Output:
            frames = [["frame-a", "frame-b"]]

        class FakePipeline:
            _modiff_video_pipeline_class = "WanVideoToVideoPipeline"
            _execution_device = "cpu"

            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return Output()

        pipeline = FakePipeline()
        output = Generate().execute(
            pipeline=pipeline,
            mode="video_to_video",
            video=["source-a", "source-b"],
            prompt="Preserve geometry and apply a restrained winter grade.",
            width=832,
            height=480,
            num_inference_steps=4,
            guidance_scale=3,
            strength=0.35,
            seed=9,
        )

        self.assertEqual(output["frames_out"], 2)
        self.assertEqual(pipeline.calls[0]["strength"], 0.35)
        self.assertEqual(pipeline.calls[0]["video"], ["source-a", "source-b"])

    def test_base_wan_text_to_video_uses_the_same_generic_node_contract(self):
        adapter = get_video_pipeline_adapter("WanPipeline")
        self.assertEqual(adapter.default_repo, "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        self.assertEqual(adapter.modes, ("text_to_video",))

        class Output:
            frames = [["frame-a", "frame-b"]]

        class FakePipeline:
            _modiff_video_pipeline_class = "WanPipeline"
            _execution_device = "cpu"
            vae_scale_factor_temporal = 4

            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return Output()

        pipeline = FakePipeline()
        output = Generate().execute(
            pipeline=pipeline,
            mode="text_to_video",
            prompt="A low camera tracks rapidly through windblown coastal grass.",
            width=832,
            height=480,
            num_frames=81,
            num_inference_steps=4,
            guidance_scale=5,
            seed=12,
        )

        self.assertEqual(output["frames_out"], 2)
        self.assertEqual(pipeline.calls[0]["num_frames"], 81)
        self.assertNotIn("video", pipeline.calls[0])

    def test_wan_video_to_video_rejects_vace_only_inputs_before_execution(self):
        class FakePipeline:
            _modiff_video_pipeline_class = "WanVideoToVideoPipeline"
            _execution_device = "cpu"

            def __call__(self, **_kwargs):
                raise AssertionError("pipeline must not run for an invalid contract")

        with self.assertRaisesRegex(ValueError, "does not accept a mask"):
            Generate().execute(
                pipeline=FakePipeline(),
                mode="video_to_video",
                video=["source"],
                mask=["mask"],
            )

    def test_ltx_rejects_incompatible_inputs_before_pipeline_execution(self):
        class FakePipeline:
            _modiff_video_pipeline_class = "LTXConditionPipeline"
            _execution_device = "cpu"

            def __call__(self, **_kwargs):
                raise AssertionError("pipeline must not run for an invalid contract")

        node = Generate()
        with self.assertRaisesRegex(ValueError, "requires at least one reference image"):
            node.execute(pipeline=FakePipeline(), mode="image_to_video", width=704, height=480)
        with self.assertRaisesRegex(ValueError, "does not support the generic mask input"):
            node.execute(
                pipeline=FakePipeline(),
                mode="text_to_video",
                mask=[object()],
                width=704,
                height=480,
            )

    def test_ltx_image_condition_uses_the_qualified_one_frame_video_contract(self):
        class Output:
            frames = [["frame-a"]]

        class FakePipeline:
            _modiff_video_pipeline_class = "LTXConditionPipeline"
            _execution_device = "cpu"

            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return Output()

        source = object()
        pipeline = FakePipeline()
        condition_module = SimpleNamespace(LTXVideoCondition=FakeLTXVideoCondition)
        with patch.dict(
            sys.modules,
            {"diffusers.pipelines.ltx.pipeline_ltx_condition": condition_module},
        ):
            Generate().execute(
                pipeline=pipeline,
                mode="image_to_video",
                reference_images=[source],
                prompt="Water pours while the camera moves laterally.",
                width=704,
                height=480,
                num_frames=81,
                num_inference_steps=4,
                strength=0.85,
            )

        condition = pipeline.calls[0]["conditions"][0]
        self.assertEqual(condition.video, [source])
        self.assertIsNone(condition.image)
        self.assertEqual(condition.frame_index, 0)
        self.assertEqual(condition.strength, 0.85)
        self.assertNotIn("image", pipeline.calls[0])

    def test_ltx_video_input_is_one_condition_not_one_condition_per_frame(self):
        class Output:
            frames = [["frame-a"]]

        class FakePipeline:
            _modiff_video_pipeline_class = "LTXConditionPipeline"
            _execution_device = "cpu"

            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return Output()

        source_frames = [object(), object(), object()]
        pipeline = FakePipeline()
        condition_module = SimpleNamespace(LTXVideoCondition=FakeLTXVideoCondition)
        with patch.dict(
            sys.modules,
            {"diffusers.pipelines.ltx.pipeline_ltx_condition": condition_module},
        ):
            Generate().execute(
                pipeline=pipeline,
                mode="video_to_video",
                video=source_frames,
                prompt="Preserve motion and restyle the season.",
                width=704,
                height=480,
                num_frames=81,
                num_inference_steps=4,
                strength=0.55,
            )

        conditions = pipeline.calls[0]["conditions"]
        self.assertEqual(len(conditions), 1)
        self.assertEqual(conditions[0].video, source_frames)

    def test_ltx_distilled_uses_its_eight_step_schedule_without_cfg(self):
        class Output:
            frames = [["frame-a"]]

        class FakePipeline:
            _modiff_video_pipeline_class = "LTXConditionPipeline"
            _modiff_video_repo = "Lightricks/LTX-Video-0.9.8-13B-distilled"
            _execution_device = "cpu"

            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return Output()

        pipeline = FakePipeline()
        Generate().execute(
            pipeline=pipeline,
            mode="text_to_video",
            prompt="A documentary camera flies beside a waterfall.",
            negative_prompt="jitter",
            width=704,
            height=480,
            num_frames=81,
            num_inference_steps=8,
            guidance_scale=1,
        )

        call = pipeline.calls[0]
        self.assertEqual(call["timesteps"], LTX_DISTILLED_TIMESTEPS)
        self.assertIsNone(call["negative_prompt"])

    def test_ltx_distilled_rejects_dev_schedule_parameters(self):
        class FakePipeline:
            _modiff_video_pipeline_class = "LTXConditionPipeline"
            _modiff_video_repo = "Lightricks/LTX-Video-0.9.8-13B-distilled"
            _execution_device = "cpu"

            def __call__(self, **_kwargs):
                raise AssertionError("pipeline must not run with an incompatible distilled schedule")

        with self.assertRaisesRegex(ValueError, "exactly 8 inference steps"):
            Generate().execute(
                pipeline=FakePipeline(),
                mode="text_to_video",
                prompt="A documentary shot.",
                width=704,
                height=480,
                num_inference_steps=30,
                guidance_scale=1,
            )
        with self.assertRaisesRegex(ValueError, "guidance scale 1"):
            Generate().execute(
                pipeline=FakePipeline(),
                mode="text_to_video",
                prompt="A documentary shot.",
                width=704,
                height=480,
                num_inference_steps=8,
                guidance_scale=3,
            )

    def test_same_ltx_graph_accepts_different_repositories_via_the_adapter_contract(self):
        class Output:
            frames = [["frame-a", "frame-b"]]

        class FakePipeline:
            _modiff_video_pipeline_class = "LTXConditionPipeline"
            _execution_device = "cpu"

            def __init__(self, repo):
                self.repo = repo
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return Output()

        node = Generate()
        for repo in ("Lightricks/LTX-Video-0.9.8-13B-distilled", "local/qualified-ltx-repo"):
            pipeline = FakePipeline(repo)
            output = node.execute(
                pipeline=pipeline,
                mode="text_to_video",
                prompt="A fixed-camera documentary shot.",
                width=704,
                height=480,
                num_frames=96,
                num_inference_steps=4,
                guidance_scale=3,
                seed=7,
            )
            self.assertEqual(output["frames_out"], 2)
            self.assertEqual(pipeline.calls[0]["num_frames"], 97)
            self.assertNotIn("image", pipeline.calls[0])

    def test_ltx_condition_injects_resolution_dependent_dynamic_scheduler_shift(self):
        class Output:
            frames = [["frame-a"]]

        class Scheduler:
            config = {
                "use_dynamic_shifting": True,
                "base_image_seq_len": 1024,
                "max_image_seq_len": 4096,
                "base_shift": 0.95,
                "max_shift": 2.05,
            }

            def __init__(self):
                self.calls = []

            def set_timesteps(
                self,
                num_inference_steps=None,
                device=None,
                sigmas=None,
                mu=None,
                timesteps=None,
            ):
                self.calls.append(
                    (
                        (num_inference_steps,),
                        {"device": device, "sigmas": sigmas, "mu": mu, "timesteps": timesteps},
                    )
                )

        class FakePipeline:
            _modiff_video_pipeline_class = "LTXConditionPipeline"
            _execution_device = "cpu"
            vae_temporal_compression_ratio = 8
            vae_spatial_compression_ratio = 32

            def __init__(self):
                self.scheduler = Scheduler()

            def __call__(self, **_kwargs):
                self.scheduler_signature = tuple(inspect.signature(self.scheduler.set_timesteps).parameters)
                self.scheduler.set_timesteps(4)
                return Output()

        pipeline = FakePipeline()
        original_set_timesteps = pipeline.scheduler.set_timesteps
        Generate().execute(
            pipeline=pipeline,
            mode="text_to_video",
            prompt="A fixed-camera documentary shot.",
            width=704,
            height=480,
            num_frames=49,
            num_inference_steps=4,
            guidance_scale=3,
            seed=7,
        )

        expected_sequence_length = 7 * 15 * 22
        expected_mu = 0.95 + (expected_sequence_length - 1024) * (2.05 - 0.95) / (4096 - 1024)
        self.assertAlmostEqual(pipeline.scheduler.calls[0][1]["mu"], expected_mu)
        self.assertIn("timesteps", pipeline.scheduler_signature)
        self.assertEqual(pipeline.scheduler.set_timesteps, original_set_timesteps)

    def test_stable_video_loader_uses_only_the_reviewed_safe_chunked_recipe(self):
        pipeline = SimpleNamespace(
            unet=SimpleNamespace(enable_forward_chunking=MagicMock()),
        )
        node = LoadPipeline("stable-video-loader")
        with (
            patch(
                "diffusers.StableVideoDiffusionPipeline.from_pretrained",
                return_value=pipeline,
            ) as from_pretrained,
            patch("modules.DiffusersVideo.main.apply_pipeline_offload") as apply_offload,
            patch("modules.DiffusersRuntime.main.apply_execution_recipe_to_pipeline") as apply_recipe,
            patch.object(node, "mm_add") as mm_add,
        ):
            result = node.execute(
                pipeline_class="StableVideoDiffusionPipeline",
                model_id={"source": "hub", "value": STABLE_VIDEO_DIFFUSION_REPO},
                dtype="float16",
                device="cpu",
                offload_mode="model_cpu",
            )

        self.assertIs(result["pipeline"], pipeline)
        self.assertEqual(result["resolved_artifact"], STABLE_VIDEO_DIFFUSION_REPO)
        self.assertEqual(pipeline._modiff_video_pipeline_class, "StableVideoDiffusionPipeline")
        self.assertEqual(pipeline._modiff_video_repo, STABLE_VIDEO_DIFFUSION_REPO)
        self.assertEqual(pipeline._modiff_video_revision, STABLE_VIDEO_DIFFUSION_REVISION)
        load_args, load_kwargs = from_pretrained.call_args
        self.assertEqual(load_args, (STABLE_VIDEO_DIFFUSION_REPO,))
        self.assertEqual(load_kwargs["revision"], STABLE_VIDEO_DIFFUSION_REVISION)
        self.assertEqual(load_kwargs["variant"], "fp16")
        self.assertIs(load_kwargs["use_safetensors"], True)
        self.assertNotIn("trust_remote_code", load_kwargs)
        self.assertNotIn("quantization_config", load_kwargs)
        self.assertNotIn("device_map", load_kwargs)
        pipeline.unet.enable_forward_chunking.assert_called_once_with()
        apply_recipe.assert_called_once_with(pipeline, {})
        apply_offload.assert_called_once_with(
            pipeline,
            mode="none",
            device="cpu",
            node_id="stable-video-loader",
            scope="stable-video-diffusion",
        )
        mm_add.assert_called_once_with(pipeline, priority=2)

    def test_animatediff_loader_pins_safe_base_motion_adapter_and_ddim_scheduler(self):
        motion_adapter = object()
        scheduler = object()
        pipeline = SimpleNamespace(vae=SimpleNamespace(enable_slicing=MagicMock()))
        node = LoadPipeline("animatediff-loader")
        with (
            patch("diffusers.MotionAdapter.from_pretrained", return_value=motion_adapter) as load_motion,
            patch("diffusers.DDIMScheduler.from_pretrained", return_value=scheduler) as load_scheduler,
            patch("diffusers.AnimateDiffPipeline.from_pretrained", return_value=pipeline) as load_pipeline,
            patch("modules.DiffusersVideo.main.apply_pipeline_offload") as apply_offload,
            patch("modules.DiffusersRuntime.main.apply_execution_recipe_to_pipeline") as apply_recipe,
            patch.object(node, "mm_add") as mm_add,
        ):
            result = node.execute(
                pipeline_class="AnimateDiffPipeline",
                model_id={"source": "hub", "value": ANIMATEDIFF_BASE_REPO},
                revision=ANIMATEDIFF_BASE_REVISION,
                motion_adapter_id={"source": "hub", "value": ANIMATEDIFF_MOTION_REPO},
                motion_adapter_revision=ANIMATEDIFF_MOTION_REVISION,
                dtype="float16",
                device="cpu",
                offload_mode="model_cpu",
            )

        self.assertIs(result["pipeline"], pipeline)
        self.assertEqual(pipeline._modiff_video_pipeline_class, "AnimateDiffPipeline")
        self.assertEqual(pipeline._modiff_video_repo, ANIMATEDIFF_BASE_REPO)
        self.assertEqual(pipeline._modiff_video_revision, ANIMATEDIFF_BASE_REVISION)
        self.assertEqual(pipeline._modiff_video_motion_adapter_repo, ANIMATEDIFF_MOTION_REPO)
        self.assertEqual(pipeline._modiff_video_motion_adapter_revision, ANIMATEDIFF_MOTION_REVISION)
        motion_args, motion_kwargs = load_motion.call_args
        self.assertEqual(motion_args, (ANIMATEDIFF_MOTION_REPO,))
        self.assertEqual(motion_kwargs["revision"], ANIMATEDIFF_MOTION_REVISION)
        self.assertEqual(motion_kwargs["variant"], "fp16")
        self.assertIs(motion_kwargs["use_safetensors"], True)
        scheduler_args, scheduler_kwargs = load_scheduler.call_args
        self.assertEqual(scheduler_args, (ANIMATEDIFF_BASE_REPO,))
        self.assertEqual(scheduler_kwargs["subfolder"], "scheduler")
        self.assertEqual(scheduler_kwargs["revision"], ANIMATEDIFF_BASE_REVISION)
        self.assertIs(scheduler_kwargs["clip_sample"], False)
        self.assertEqual(scheduler_kwargs["timestep_spacing"], "linspace")
        self.assertEqual(scheduler_kwargs["beta_schedule"], "linear")
        self.assertEqual(scheduler_kwargs["steps_offset"], 1)
        base_args, base_kwargs = load_pipeline.call_args
        self.assertEqual(base_args, (ANIMATEDIFF_BASE_REPO,))
        self.assertIs(base_kwargs["motion_adapter"], motion_adapter)
        self.assertIs(base_kwargs["scheduler"], scheduler)
        self.assertEqual(base_kwargs["revision"], ANIMATEDIFF_BASE_REVISION)
        self.assertEqual(base_kwargs["variant"], "fp16")
        self.assertIs(base_kwargs["use_safetensors"], True)
        pipeline.vae.enable_slicing.assert_called_once_with()
        apply_recipe.assert_called_once_with(pipeline, {})
        apply_offload.assert_called_once_with(
            pipeline,
            mode="none",
            device="cpu",
            node_id="animatediff-loader",
            scope="animatediff",
        )
        mm_add.assert_called_once_with(pipeline, priority=2)

    def test_animatelcm_loader_pins_linear_scheduler_and_named_safe_lora(self):
        replacement_scheduler = object()
        pipeline = SimpleNamespace(
            scheduler=SimpleNamespace(config={"beta_schedule": "scaled_linear"}),
            vae=SimpleNamespace(enable_slicing=MagicMock()),
            load_lora_weights=MagicMock(),
            set_adapters=MagicMock(),
        )
        node = LoadPipeline("animatelcm-loader")
        with (
            patch("diffusers.MotionAdapter.from_pretrained", return_value=object()) as load_motion,
            patch("diffusers.AnimateDiffPipeline.from_pretrained", return_value=pipeline),
            patch("diffusers.LCMScheduler.from_config", return_value=replacement_scheduler) as lcm_scheduler,
            patch("modules.DiffusersVideo.main.apply_pipeline_offload"),
            patch("modules.DiffusersRuntime.main.apply_execution_recipe_to_pipeline"),
            patch.object(node, "mm_add"),
        ):
            node.execute(
                pipeline_class="AnimateLCMPipeline",
                model_id={"source": "hub", "value": ANIMATEDIFF_BASE_REPO},
                revision=ANIMATEDIFF_BASE_REVISION,
                motion_adapter_id={"source": "hub", "value": ANIMATELCM_MOTION_REPO},
                motion_adapter_revision=ANIMATELCM_MOTION_REVISION,
                dtype="float16",
                device="cpu",
            )

        self.assertEqual(load_motion.call_args.args, (ANIMATELCM_MOTION_REPO,))
        self.assertEqual(load_motion.call_args.kwargs["revision"], ANIMATELCM_MOTION_REVISION)
        lcm_scheduler.assert_called_once_with({"beta_schedule": "scaled_linear"}, beta_schedule="linear")
        self.assertIs(pipeline.scheduler, replacement_scheduler)
        pipeline.load_lora_weights.assert_called_once_with(
            ANIMATELCM_MOTION_REPO,
            weight_name=ANIMATELCM_LORA_WEIGHT_NAME,
            adapter_name=ANIMATELCM_LORA_ADAPTER_NAME,
            revision=ANIMATELCM_MOTION_REVISION,
            local_files_only=True,
            use_safetensors=True,
        )
        pipeline.set_adapters.assert_called_once_with(
            [ANIMATELCM_LORA_ADAPTER_NAME],
            [ANIMATELCM_LORA_SCALE],
        )

    def test_animatediff_loader_rejects_unreviewed_components_before_weight_loading(self):
        node = LoadPipeline("strict-animatediff-loader")
        with patch("diffusers.MotionAdapter.from_pretrained") as load_motion:
            with self.assertRaisesRegex(ValueError, "requires MotionAdapter"):
                node.execute(
                    pipeline_class="AnimateDiffPipeline",
                    model_id={"source": "hub", "value": ANIMATEDIFF_BASE_REPO},
                    revision=ANIMATEDIFF_BASE_REVISION,
                    motion_adapter_id={"source": "hub", "value": ANIMATELCM_MOTION_REPO},
                    motion_adapter_revision=ANIMATELCM_MOTION_REVISION,
                    dtype="float16",
                )
        load_motion.assert_not_called()

    def test_animatediff_generate_seals_the_short_text_to_video_contract(self):
        class Output:
            frames = [[f"frame-{index}" for index in range(8)]]

        class FakePipeline:
            _modiff_video_pipeline_class = "AnimateLCMPipeline"
            _modiff_video_repo = ANIMATEDIFF_BASE_REPO
            _modiff_video_revision = ANIMATEDIFF_BASE_REVISION
            _modiff_video_motion_adapter_repo = ANIMATELCM_MOTION_REPO
            _modiff_video_motion_adapter_revision = ANIMATELCM_MOTION_REVISION
            _execution_device = "cpu"

            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return Output()

        pipeline = FakePipeline()
        result = Generate().execute(
            pipeline=pipeline,
            mode="text_to_video",
            prompt="A paper kite circles above a quiet field.",
            negative_prompt="flicker",
            width=512,
            height=512,
            num_frames=8,
            num_inference_steps=6,
            guidance_scale=1.5,
            seed=23,
            output_type="pil",
        )

        self.assertEqual(result["frames_out"], 8)
        call = pipeline.calls[0]
        self.assertEqual(call["num_frames"], 8)
        self.assertEqual(call["num_inference_steps"], 6)
        self.assertEqual(call["guidance_scale"], 1.5)
        self.assertEqual(call["decode_chunk_size"], 16)
        self.assertEqual(call["num_videos_per_prompt"], 1)
        self.assertNotIn("image", call)
        with self.assertRaisesRegex(ValueError, "step count must be an integer from 1 through 8"):
            Generate().execute(
                pipeline=pipeline,
                mode="text_to_video",
                prompt="A paper kite circles above a quiet field.",
                width=512,
                height=512,
                num_frames=8,
                num_inference_steps=9,
            )

    def test_cogvideox_loader_pins_safe_weights_and_documented_tiling(self):
        pipeline = SimpleNamespace(vae=SimpleNamespace(enable_tiling=MagicMock()))
        node = LoadPipeline("cogvideox-loader")
        ordered_calls = MagicMock()
        with (
            patch("diffusers.CogVideoXPipeline.from_pretrained", return_value=pipeline) as from_pretrained,
            patch("modules.DiffusersVideo.main.apply_pipeline_offload") as apply_offload,
            patch("modules.DiffusersRuntime.main.apply_execution_recipe_to_pipeline") as apply_recipe,
            patch.object(node, "mm_add") as mm_add,
        ):
            ordered_calls.attach_mock(apply_recipe, "apply_recipe")
            ordered_calls.attach_mock(pipeline.vae.enable_tiling, "enable_tiling")
            result = node.execute(
                pipeline_class="CogVideoXPipeline",
                model_id={"source": "hub", "value": COGVIDEOX_2B_REPO},
                revision=COGVIDEOX_2B_REVISION,
                dtype="float16",
                device="cpu",
                offload_mode="model_cpu",
            )

        self.assertIs(result["pipeline"], pipeline)
        self.assertEqual(result["resolved_artifact"], COGVIDEOX_2B_REPO)
        self.assertEqual(pipeline._modiff_video_pipeline_class, "CogVideoXPipeline")
        self.assertEqual(pipeline._modiff_video_repo, COGVIDEOX_2B_REPO)
        self.assertEqual(pipeline._modiff_video_revision, COGVIDEOX_2B_REVISION)
        load_args, load_kwargs = from_pretrained.call_args
        self.assertEqual(load_args, (COGVIDEOX_2B_REPO,))
        self.assertEqual(load_kwargs["revision"], COGVIDEOX_2B_REVISION)
        self.assertIs(load_kwargs["use_safetensors"], True)
        self.assertNotIn("trust_remote_code", load_kwargs)
        self.assertNotIn("quantization_config", load_kwargs)
        self.assertNotIn("device_map", load_kwargs)
        pipeline.vae.enable_tiling.assert_called_once_with()
        apply_recipe.assert_called_once_with(pipeline, {})
        self.assertEqual(
            ordered_calls.method_calls[:2],
            [
                call.apply_recipe(pipeline, {}),
                call.enable_tiling(),
            ],
        )
        apply_offload.assert_called_once_with(
            pipeline,
            mode="none",
            device="cpu",
            node_id="cogvideox-loader",
            scope="cogvideox-2b",
        )
        mm_add.assert_called_once_with(pipeline, priority=2)

    def test_cogvideox_loader_rejects_unreviewed_artifacts_before_diffusers(self):
        node = LoadPipeline("strict-cogvideox-loader")
        with patch("diffusers.CogVideoXPipeline.from_pretrained") as from_pretrained:
            with self.assertRaisesRegex(ValueError, "exact reviewed Hub artifact"):
                node.execute(
                    pipeline_class="CogVideoXPipeline",
                    model_id={"source": "hub", "value": "organization/custom-cogvideo"},
                    revision="0123456789abcdef0123456789abcdef01234567",
                    dtype="float16",
                )
        from_pretrained.assert_not_called()

    def test_cogvideox_generate_seals_native_short_text_to_video_contract(self):
        class Output:
            frames = [[f"frame-{index}" for index in range(25)]]

        class FakePipeline:
            _modiff_video_pipeline_class = "CogVideoXPipeline"
            _modiff_video_repo = COGVIDEOX_2B_REPO
            _modiff_video_revision = COGVIDEOX_2B_REVISION
            _execution_device = "cpu"

            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return Output()

        pipeline = FakePipeline()
        result = Generate().execute(
            pipeline=pipeline,
            mode="text_to_video",
            prompt="A red kite crosses a quiet winter sky.",
            negative_prompt="camera shake",
            width=720,
            height=480,
            num_frames=25,
            num_inference_steps=25,
            guidance_scale=6,
            seed=29,
            output_type="pil",
            max_sequence_length=226,
        )

        self.assertEqual(result, {"video_out": Output.frames[0], "width_out": 720, "height_out": 480, "frames_out": 25})
        call = pipeline.calls[0]
        self.assertEqual(call["num_frames"], 25)
        self.assertEqual(call["num_inference_steps"], 25)
        self.assertEqual(call["guidance_scale"], 6)
        self.assertIs(call["use_dynamic_cfg"], False)
        self.assertEqual(call["num_videos_per_prompt"], 1)
        self.assertEqual(call["max_sequence_length"], 226)
        self.assertNotIn("image", call)

    def test_cogvideox_invalid_short_contracts_fail_before_torch_or_execution(self):
        pipeline = SimpleNamespace(
            _modiff_video_pipeline_class="CogVideoXPipeline",
            _modiff_video_repo=COGVIDEOX_2B_REPO,
            _modiff_video_revision=COGVIDEOX_2B_REVISION,
        )
        invalid = (
            ({"num_frames": 8}, "integer from 9 through 25"),
            ({"num_frames": 24}, r"4k\+1"),
            ({"width": 712}, "integer from 720 through 720"),
            ({"reference_images": [Image.new("RGB", (720, 480))]}, "does not accept image conditioning"),
            ({"output_type": "np"}, "requires output_type=pil"),
            ({"max_sequence_length": 227}, "integer from 1 through 226"),
        )
        with patch.dict(sys.modules, {"torch": None}):
            for update, message in invalid:
                with self.subTest(update=update):
                    with self.assertRaisesRegex(ValueError, message):
                        Generate().execute(
                            pipeline=pipeline,
                            mode="text_to_video",
                            prompt="A red kite crosses a quiet winter sky.",
                            **update,
                        )

    def test_allegro_loader_pins_safe_weights_fp32_vae_and_documented_tiling(self):
        vae = SimpleNamespace(enable_tiling=MagicMock())
        pipeline = SimpleNamespace(vae=vae)
        node = LoadPipeline("allegro-loader")
        ordered_calls = MagicMock()
        with (
            patch("diffusers.AutoencoderKLAllegro.from_pretrained", return_value=vae) as load_vae,
            patch("diffusers.AllegroPipeline.from_pretrained", return_value=pipeline) as load_pipeline,
            patch("modules.DiffusersVideo.main.apply_pipeline_offload") as apply_offload,
            patch("modules.DiffusersRuntime.main.apply_execution_recipe_to_pipeline") as apply_recipe,
            patch.object(node, "mm_add") as mm_add,
        ):
            ordered_calls.attach_mock(apply_recipe, "apply_recipe")
            ordered_calls.attach_mock(vae.enable_tiling, "enable_tiling")
            result = node.execute(
                pipeline_class="AllegroPipeline",
                model_id={"source": "hub", "value": ALLEGRO_REPO},
                revision=ALLEGRO_REVISION,
                dtype="bfloat16",
                device="cpu",
                offload_mode="sequential_cpu",
            )

        self.assertIs(result["pipeline"], pipeline)
        self.assertEqual(result["resolved_artifact"], ALLEGRO_REPO)
        self.assertEqual(pipeline._modiff_video_pipeline_class, "AllegroPipeline")
        self.assertEqual(pipeline._modiff_video_repo, ALLEGRO_REPO)
        self.assertEqual(pipeline._modiff_video_revision, ALLEGRO_REVISION)
        self.assertEqual(load_vae.call_args.args, (ALLEGRO_REPO,))
        self.assertEqual(load_vae.call_args.kwargs["subfolder"], "vae")
        self.assertEqual(str(load_vae.call_args.kwargs["torch_dtype"]), "torch.float32")
        self.assertIs(load_vae.call_args.kwargs["use_safetensors"], True)
        self.assertEqual(load_pipeline.call_args.args, (ALLEGRO_REPO,))
        self.assertIs(load_pipeline.call_args.kwargs["vae"], vae)
        self.assertEqual(str(load_pipeline.call_args.kwargs["torch_dtype"]), "torch.bfloat16")
        self.assertEqual(load_pipeline.call_args.kwargs["revision"], ALLEGRO_REVISION)
        self.assertIs(load_pipeline.call_args.kwargs["use_safetensors"], True)
        self.assertNotIn("trust_remote_code", load_pipeline.call_args.kwargs)
        self.assertNotIn("quantization_config", load_pipeline.call_args.kwargs)
        self.assertNotIn("device_map", load_pipeline.call_args.kwargs)
        self.assertEqual(
            ordered_calls.method_calls[:2],
            [call.apply_recipe(pipeline, {}), call.enable_tiling()],
        )
        apply_offload.assert_called_once_with(
            pipeline,
            mode="none",
            device="cpu",
            node_id="allegro-loader",
            scope="allegro",
        )
        mm_add.assert_called_once_with(pipeline, priority=2)

    def test_allegro_loader_rejects_unreviewed_artifacts_before_diffusers(self):
        node = LoadPipeline("strict-allegro-loader")
        with patch("diffusers.AllegroPipeline.from_pretrained") as from_pretrained:
            with self.assertRaisesRegex(ValueError, "exact reviewed Hub artifact"):
                node.execute(
                    pipeline_class="AllegroPipeline",
                    model_id={"source": "hub", "value": "organization/custom-allegro"},
                    revision="0123456789abcdef0123456789abcdef01234567",
                    dtype="bfloat16",
                )
        from_pretrained.assert_not_called()

    def test_allegro_generate_seals_native_text_to_video_contract(self):
        class Output:
            frames = [[f"frame-{index}" for index in range(88)]]

        class FakePipeline:
            _modiff_video_pipeline_class = "AllegroPipeline"
            _modiff_video_repo = ALLEGRO_REPO
            _modiff_video_revision = ALLEGRO_REVISION
            _execution_device = "cpu"

            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return Output()

        pipeline = FakePipeline()
        result = Generate().execute(
            pipeline=pipeline,
            mode="text_to_video",
            prompt="A sailboat crosses a sunlit bay.",
            negative_prompt="flicker",
            width=1280,
            height=720,
            num_frames=88,
            num_inference_steps=100,
            guidance_scale=7.5,
            seed=31,
            output_type="pil",
            max_sequence_length=512,
        )

        self.assertEqual(
            result,
            {"video_out": Output.frames[0], "width_out": 1280, "height_out": 720, "frames_out": 88},
        )
        call_kwargs = pipeline.calls[0]
        self.assertEqual(call_kwargs["num_frames"], 88)
        self.assertEqual(call_kwargs["num_inference_steps"], 100)
        self.assertEqual(call_kwargs["guidance_scale"], 7.5)
        self.assertEqual(call_kwargs["num_videos_per_prompt"], 1)
        self.assertEqual(call_kwargs["max_sequence_length"], 512)
        self.assertIs(call_kwargs["clean_caption"], False)
        self.assertNotIn("image", call_kwargs)
        self.assertNotIn("attention_kwargs", call_kwargs)

    def test_allegro_invalid_native_contracts_fail_before_torch_or_execution(self):
        pipeline = SimpleNamespace(
            _modiff_video_pipeline_class="AllegroPipeline",
            _modiff_video_repo=ALLEGRO_REPO,
            _modiff_video_revision=ALLEGRO_REVISION,
        )
        invalid = (
            ({"num_frames": 87}, "integer from 88 through 88"),
            ({"width": 1272}, "integer from 1280 through 1280"),
            ({"height": 712}, "integer from 720 through 720"),
            ({"num_inference_steps": 101}, "integer from 1 through 100"),
            ({"reference_images": [Image.new("RGB", (1280, 720))]}, "does not accept image conditioning"),
            ({"output_type": "np"}, "requires output_type=pil"),
            ({"max_sequence_length": 513}, "integer from 1 through 512"),
        )
        with patch.dict(sys.modules, {"torch": None}):
            for update, message in invalid:
                with self.subTest(update=update):
                    with self.assertRaisesRegex(ValueError, message):
                        Generate().execute(
                            pipeline=pipeline,
                            mode="text_to_video",
                            prompt="A sailboat crosses a sunlit bay.",
                            **update,
                        )

    def test_latte_loader_pins_safe_float16_weights_and_sequential_offload(self):
        pipeline = SimpleNamespace()
        node = LoadPipeline("latte-loader")
        with (
            patch("diffusers.LattePipeline.from_pretrained", return_value=pipeline) as load_pipeline,
            patch("modules.DiffusersVideo.main.apply_pipeline_offload") as apply_offload,
            patch("modules.DiffusersRuntime.main.apply_execution_recipe_to_pipeline") as apply_recipe,
            patch.object(node, "mm_add") as mm_add,
        ):
            result = node.execute(
                pipeline_class="LattePipeline",
                model_id={"source": "hub", "value": LATTE_REPO},
                revision=LATTE_REVISION,
                dtype="float16",
                device="cpu",
                offload_mode="sequential_cpu",
            )

        self.assertIs(result["pipeline"], pipeline)
        self.assertEqual(result["resolved_artifact"], LATTE_REPO)
        self.assertEqual(pipeline._modiff_video_pipeline_class, "LattePipeline")
        self.assertEqual(pipeline._modiff_video_repo, LATTE_REPO)
        self.assertEqual(pipeline._modiff_video_revision, LATTE_REVISION)
        self.assertEqual(load_pipeline.call_args.args, (LATTE_REPO,))
        self.assertEqual(str(load_pipeline.call_args.kwargs["torch_dtype"]), "torch.float16")
        self.assertEqual(load_pipeline.call_args.kwargs["revision"], LATTE_REVISION)
        self.assertIs(load_pipeline.call_args.kwargs["use_safetensors"], True)
        self.assertNotIn("trust_remote_code", load_pipeline.call_args.kwargs)
        self.assertNotIn("quantization_config", load_pipeline.call_args.kwargs)
        self.assertNotIn("device_map", load_pipeline.call_args.kwargs)
        apply_recipe.assert_called_once_with(pipeline, {})
        apply_offload.assert_called_once_with(
            pipeline,
            mode="none",
            device="cpu",
            node_id="latte-loader",
            scope="latte",
        )
        mm_add.assert_called_once_with(pipeline, priority=2)

    def test_latte_loader_rejects_unreviewed_artifacts_before_diffusers(self):
        node = LoadPipeline("strict-latte-loader")
        with patch("diffusers.LattePipeline.from_pretrained") as from_pretrained:
            with self.assertRaisesRegex(ValueError, "exact reviewed Hub artifact"):
                node.execute(
                    pipeline_class="LattePipeline",
                    model_id={"source": "hub", "value": "organization/custom-latte"},
                    revision="0123456789abcdef0123456789abcdef01234567",
                    dtype="float16",
                )
        from_pretrained.assert_not_called()

    def test_latte_generate_seals_native_text_to_video_contract(self):
        class Output:
            frames = [[f"frame-{index}" for index in range(16)]]

        class FakePipeline:
            _modiff_video_pipeline_class = "LattePipeline"
            _modiff_video_repo = LATTE_REPO
            _modiff_video_revision = LATTE_REVISION
            _execution_device = "cpu"

            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return Output()

        pipeline = FakePipeline()
        result = Generate().execute(
            pipeline=pipeline,
            mode="text_to_video",
            prompt="A coffee cup steams beside a rain-streaked window.",
            negative_prompt="flicker",
            width=512,
            height=512,
            num_frames=16,
            num_inference_steps=50,
            guidance_scale=7.5,
            seed=37,
            output_type="pil",
        )

        self.assertEqual(
            result,
            {"video_out": Output.frames[0], "width_out": 512, "height_out": 512, "frames_out": 16},
        )
        call_kwargs = pipeline.calls[0]
        self.assertEqual(call_kwargs["video_length"], 16)
        self.assertEqual(call_kwargs["num_inference_steps"], 50)
        self.assertEqual(call_kwargs["guidance_scale"], 7.5)
        self.assertEqual(call_kwargs["num_images_per_prompt"], 1)
        self.assertEqual(call_kwargs["decode_chunk_size"], 14)
        self.assertIs(call_kwargs["clean_caption"], False)
        self.assertIs(call_kwargs["mask_feature"], True)
        self.assertIs(call_kwargs["enable_temporal_attentions"], True)
        self.assertNotIn("image", call_kwargs)

    def test_latte_invalid_native_contracts_fail_before_torch_or_execution(self):
        pipeline = SimpleNamespace(
            _modiff_video_pipeline_class="LattePipeline",
            _modiff_video_repo=LATTE_REPO,
            _modiff_video_revision=LATTE_REVISION,
        )
        invalid = (
            ({"num_frames": 15}, "integer from 16 through 16"),
            ({"width": 504}, "integer from 512 through 512"),
            ({"height": 520}, "integer from 512 through 512"),
            ({"num_inference_steps": 51}, "integer from 1 through 50"),
            ({"reference_images": [Image.new("RGB", (512, 512))]}, "does not accept image conditioning"),
            ({"output_type": "np"}, "requires output_type=pil"),
            ({"guidance_scale": 12.1}, "finite and from 1 through 12"),
        )
        with patch.dict(sys.modules, {"torch": None}):
            for update, message in invalid:
                with self.subTest(update=update):
                    with self.assertRaisesRegex(ValueError, message):
                        Generate().execute(
                            pipeline=pipeline,
                            mode="text_to_video",
                            prompt="A coffee cup steams beside a rain-streaked window.",
                            **update,
                        )

    def test_mochi_loader_pins_indexed_t5_and_bfloat16_variant(self):
        pipeline = SimpleNamespace(enable_vae_tiling=MagicMock())
        text_encoder = object()
        node = LoadPipeline("mochi-loader")
        with (
            patch("transformers.T5EncoderModel.from_pretrained", return_value=text_encoder) as load_text_encoder,
            patch("diffusers.MochiPipeline.from_pretrained", return_value=pipeline) as load_pipeline,
            patch("modules.DiffusersVideo.main.apply_pipeline_offload") as apply_offload,
            patch("modules.DiffusersRuntime.main.apply_execution_recipe_to_pipeline") as apply_recipe,
            patch.object(node, "mm_add") as mm_add,
        ):
            result = node.execute(
                pipeline_class="MochiPipeline",
                model_id={"source": "hub", "value": MOCHI_REPO},
                revision=MOCHI_REVISION,
                dtype="bfloat16",
                device="cpu",
                offload_mode="sequential_cpu",
            )

        self.assertIs(result["pipeline"], pipeline)
        self.assertEqual(result["resolved_artifact"], MOCHI_REPO)
        self.assertEqual(pipeline._modiff_video_pipeline_class, "MochiPipeline")
        self.assertEqual(pipeline._modiff_video_repo, MOCHI_REPO)
        self.assertEqual(pipeline._modiff_video_revision, MOCHI_REVISION)
        self.assertEqual(load_text_encoder.call_args.args, (MOCHI_REPO,))
        self.assertEqual(load_text_encoder.call_args.kwargs["subfolder"], "text_encoder")
        self.assertEqual(str(load_text_encoder.call_args.kwargs["torch_dtype"]), "torch.bfloat16")
        self.assertIs(load_text_encoder.call_args.kwargs["use_safetensors"], True)
        self.assertEqual(load_pipeline.call_args.args, (MOCHI_REPO,))
        self.assertIs(load_pipeline.call_args.kwargs["text_encoder"], text_encoder)
        self.assertEqual(load_pipeline.call_args.kwargs["variant"], "bf16")
        self.assertEqual(str(load_pipeline.call_args.kwargs["torch_dtype"]), "torch.bfloat16")
        self.assertEqual(load_pipeline.call_args.kwargs["revision"], MOCHI_REVISION)
        self.assertIs(load_pipeline.call_args.kwargs["use_safetensors"], True)
        self.assertNotIn("trust_remote_code", load_pipeline.call_args.kwargs)
        self.assertNotIn("quantization_config", load_pipeline.call_args.kwargs)
        self.assertNotIn("device_map", load_pipeline.call_args.kwargs)
        apply_recipe.assert_called_once_with(pipeline, {})
        pipeline.enable_vae_tiling.assert_called_once_with()
        apply_offload.assert_called_once_with(
            pipeline,
            mode="none",
            device="cpu",
            node_id="mochi-loader",
            scope="mochi",
        )
        mm_add.assert_called_once_with(pipeline, priority=2)

    def test_mochi_loader_rejects_unreviewed_artifacts_before_diffusers(self):
        node = LoadPipeline("strict-mochi-loader")
        with patch("diffusers.MochiPipeline.from_pretrained") as from_pretrained:
            with self.assertRaisesRegex(ValueError, "exact reviewed Hub artifact"):
                node.execute(
                    pipeline_class="MochiPipeline",
                    model_id={"source": "hub", "value": "organization/custom-mochi"},
                    revision="0123456789abcdef0123456789abcdef01234567",
                    dtype="bfloat16",
                )
        from_pretrained.assert_not_called()

    def test_mochi_generate_seals_native_text_to_video_contract(self):
        class Output:
            frames = [[f"frame-{index}" for index in range(31)]]

        class FakePipeline:
            _modiff_video_pipeline_class = "MochiPipeline"
            _modiff_video_repo = MOCHI_REPO
            _modiff_video_revision = MOCHI_REVISION
            _execution_device = "cpu"

            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return Output()

        pipeline = FakePipeline()
        result = Generate().execute(
            pipeline=pipeline,
            mode="text_to_video",
            prompt="A paper lantern drifts over a moonlit lake.",
            negative_prompt="flicker",
            width=848,
            height=480,
            num_frames=31,
            num_inference_steps=64,
            guidance_scale=4.5,
            max_sequence_length=256,
            seed=41,
            output_type="pil",
        )

        self.assertEqual(
            result,
            {"video_out": Output.frames[0], "width_out": 848, "height_out": 480, "frames_out": 31},
        )
        call_kwargs = pipeline.calls[0]
        self.assertEqual(call_kwargs["num_frames"], 31)
        self.assertEqual(call_kwargs["num_inference_steps"], 64)
        self.assertEqual(call_kwargs["guidance_scale"], 4.5)
        self.assertEqual(call_kwargs["num_videos_per_prompt"], 1)
        self.assertEqual(call_kwargs["max_sequence_length"], 256)
        self.assertNotIn("image", call_kwargs)
        self.assertNotIn("video", call_kwargs)

    def test_mochi_invalid_native_contracts_fail_before_torch_or_execution(self):
        pipeline = SimpleNamespace(
            _modiff_video_pipeline_class="MochiPipeline",
            _modiff_video_repo=MOCHI_REPO,
            _modiff_video_revision=MOCHI_REVISION,
        )
        invalid = (
            ({"num_frames": 30}, "integer from 31 through 31"),
            ({"width": 840}, "integer from 848 through 848"),
            ({"height": 488}, "integer from 480 through 480"),
            ({"num_inference_steps": 65}, "integer from 1 through 64"),
            ({"reference_images": [Image.new("RGB", (848, 480))]}, "does not accept image conditioning"),
            ({"output_type": "np"}, "requires output_type=pil"),
            ({"max_sequence_length": 257}, "integer from 1 through 256"),
            ({"guidance_scale": 12.1}, "finite and from 1 through 12"),
        )
        with patch.dict(sys.modules, {"torch": None}):
            for update, message in invalid:
                with self.subTest(update=update):
                    with self.assertRaisesRegex(ValueError, message):
                        Generate().execute(
                            pipeline=pipeline,
                            mode="text_to_video",
                            prompt="A paper lantern drifts over a moonlit lake.",
                            **update,
                        )

    def test_stable_video_loader_rejects_unreviewed_artifacts_before_diffusers(self):
        node = LoadPipeline("strict-stable-video-loader")
        custom_revision = "0123456789abcdef0123456789abcdef01234567"
        with patch("diffusers.StableVideoDiffusionPipeline.from_pretrained") as from_pretrained:
            with self.assertRaisesRegex(ValueError, "exact reviewed gated Hub artifact"):
                node.execute(
                    pipeline_class="StableVideoDiffusionPipeline",
                    model_id={"source": "hub", "value": "organization/custom-svd"},
                    revision=custom_revision,
                )
        from_pretrained.assert_not_called()

    def test_stable_video_generate_seals_image_only_short_video_arguments(self):
        class Output:
            frames = [[f"frame-{index}" for index in range(8)]]

        class FakePipeline:
            _modiff_video_pipeline_class = "StableVideoDiffusionPipeline"
            _modiff_video_repo = STABLE_VIDEO_DIFFUSION_REPO
            _modiff_video_revision = STABLE_VIDEO_DIFFUSION_REVISION
            _execution_device = "cpu"

            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return Output()

        reference = Image.new("RGB", (1024, 576))
        pipeline = FakePipeline()
        result = Generate().execute(
            pipeline=pipeline,
            mode="image_to_video",
            reference_images=[reference],
            width=1024,
            height=576,
            num_frames=8,
            num_inference_steps=12,
            guidance_scale=3,
            frame_rate=7,
            seed=17,
            output_type="pil",
        )

        self.assertEqual(result, {"video_out": Output.frames[0], "width_out": 1024, "height_out": 576, "frames_out": 8})
        call = pipeline.calls[0]
        self.assertIs(call["image"], reference)
        self.assertEqual(call["num_frames"], 8)
        self.assertEqual(call["num_inference_steps"], 12)
        self.assertEqual(call["min_guidance_scale"], 1.0)
        self.assertEqual(call["max_guidance_scale"], 3.0)
        self.assertEqual(call["motion_bucket_id"], 127)
        self.assertEqual(call["noise_aug_strength"], 0.02)
        self.assertEqual(call["decode_chunk_size"], 2)
        self.assertEqual(call["num_videos_per_prompt"], 1)
        self.assertEqual(call["output_type"], "pil")

    def test_stable_video_invalid_contracts_fail_before_torch_or_pipeline_execution(self):
        pipeline = SimpleNamespace(
            _modiff_video_pipeline_class="StableVideoDiffusionPipeline",
            _modiff_video_repo=STABLE_VIDEO_DIFFUSION_REPO,
            _modiff_video_revision=STABLE_VIDEO_DIFFUSION_REVISION,
        )
        reference = Image.new("RGB", (1024, 576))
        invalid = (
            ({"prompt": "ignored text"}, "does not accept prompt text"),
            ({"num_frames": 26}, "frame count must be an integer from 8 through 25"),
            ({"width": 1023}, "must be divisible by 8"),
            ({"output_type": "np"}, "requires output_type=pil"),
            ({"reference_images": [reference, reference]}, "exactly one opening reference image"),
            ({"last_image": reference}, "does not accept last-image conditioning"),
        )
        with patch.dict(sys.modules, {"torch": None}):
            for update, message in invalid:
                with self.subTest(update=update):
                    values = {
                        "pipeline": pipeline,
                        "mode": "image_to_video",
                        "reference_images": [reference],
                        "width": 1024,
                        "height": 576,
                        "num_frames": 8,
                        **update,
                    }
                    with self.assertRaisesRegex(ValueError, message):
                        Generate().execute(**values)

        unreviewed = SimpleNamespace(
            _modiff_video_pipeline_class="StableVideoDiffusionPipeline",
            _modiff_video_repo="organization/custom-svd",
            _modiff_video_revision=STABLE_VIDEO_DIFFUSION_REVISION,
        )
        with self.assertRaisesRegex(ValueError, "does not match the reviewed artifact"):
            Generate().execute(
                pipeline=unreviewed,
                mode="image_to_video",
                reference_images=[reference],
            )

    def test_only_generic_video_module_key_is_registered(self):
        self.assertNotIn("modules.WanVACE", module_registry.MODULE_MAP)
        self.assertIn("modules.DiffusersVideo", module_registry.MODULE_MAP)


if __name__ == "__main__":
    unittest.main()
