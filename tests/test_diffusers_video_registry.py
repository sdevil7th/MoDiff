import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

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
    FRAMEPACK_BASE_REPO,
    FRAMEPACK_VISION_REPO,
    LTX_DISTILLED_TIMESTEPS,
    _resolve_adapter_model_selection,
    get_video_pipeline_adapter,
)


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
                {"prompt": "Preserve the parked car while the station door opens.", "duration_seconds": 5, "conditioning_strength": 0.8},
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

    def test_non_wan_adapter_replaces_only_the_inherited_legacy_model_default(self):
        adapter = get_video_pipeline_adapter("HunyuanVideoFramepackPipeline")
        inherited = {"source": "hub", "value": "Wan-AI/Wan2.1-VACE-1.3B-diffusers"}
        self.assertEqual(
            _resolve_adapter_model_selection(adapter, inherited),
            {"source": "hub", "value": "lllyasviel/FramePackI2V_HY"},
        )

        explicit = {"source": "local", "value": "/models/custom-framepack"}
        self.assertIs(_resolve_adapter_model_selection(adapter, explicit), explicit)

    def test_framepack_loader_composes_the_official_transformer_base_and_vision_repositories(self):
        adapter = get_video_pipeline_adapter("HunyuanVideoFramepackPipeline")
        transformer = object()
        feature_extractor = object()
        image_encoder = object()
        pipeline = object()
        node = LoadPipeline()

        with (
            patch(
                "diffusers.HunyuanVideoFramepackTransformer3DModel.from_pretrained", return_value=transformer
            ) as load_transformer,
            patch(
                "transformers.SiglipImageProcessor.from_pretrained", return_value=feature_extractor
            ) as load_processor,
            patch("transformers.SiglipVisionModel.from_pretrained", return_value=image_encoder) as load_encoder,
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

    def test_only_generic_video_module_key_is_registered(self):
        self.assertNotIn("modules.WanVACE", module_registry.MODULE_MAP)
        self.assertIn("modules.DiffusersVideo", module_registry.MODULE_MAP)


if __name__ == "__main__":
    unittest.main()
