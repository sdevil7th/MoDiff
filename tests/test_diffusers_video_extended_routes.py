from source_contract_helpers import source_sha256

import ast
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

from modiff.modular_workflow_contracts import PINNED_DIFFUSERS_REVISION
from modules.DiffusersVideo import Generate, LoadPipeline
from modules.DiffusersVideo.main import (
    ANIMATEDIFF_BASE_REPO,
    ANIMATEDIFF_BASE_REVISION,
    ANIMATEDIFF_CONTROLNET_REPO,
    ANIMATEDIFF_CONTROLNET_REVISION,
    ANIMATEDIFF_MOTION_REPO,
    ANIMATEDIFF_MOTION_REVISION,
    COGVIDEOX_2B_REPO,
    COGVIDEOX_2B_REVISION,
    VIDEO_MODE_FIELD_CONTRACTS,
    VIDEO_PIPELINE_ADAPTERS,
)


ANIMATEDIFF_EXTENDED_ROUTES = (
    ("AnimateDiffPAGPipeline", "text_to_video", False),
    ("AnimateDiffVideoToVideoPipeline", "video_to_video", False),
    ("AnimateDiffControlNetPipeline", "control_to_video", True),
    ("AnimateDiffVideoToVideoControlNetPipeline", "control_video_to_video", True),
)
UPSTREAM_PIPELINE_SOURCES = (
    (
        "pipelines/pag/pipeline_pag_sd_animatediff.py",
        "AnimateDiffPAGPipeline",
        "d9605b43ba0f33046289d08b679a085074152e87c06f8e49f7e2c8f0528462fe",
        {"prompt", "num_frames", "pag_scale", "pag_adaptive_scale"},
    ),
    (
        "pipelines/animatediff/pipeline_animatediff_video2video.py",
        "AnimateDiffVideoToVideoPipeline",
        "8863ad938101c7bc3b7cb282f38b624ce3335ca80eb7cc70e58763eb758fb462",
        {"video", "prompt", "strength", "enforce_inference_steps"},
    ),
    (
        "pipelines/animatediff/pipeline_animatediff_controlnet.py",
        "AnimateDiffControlNetPipeline",
        "1259f90db01892213c538af3774aca423981aeb0e531b8bf05fe776403f2641c",
        {
            "prompt",
            "num_frames",
            "conditioning_frames",
            "controlnet_conditioning_scale",
        },
    ),
    (
        "pipelines/animatediff/pipeline_animatediff_video2video_controlnet.py",
        "AnimateDiffVideoToVideoControlNetPipeline",
        "7e02f650cf690561e3ddc4c0148837ece0dc793e0c1137c3cdf6ed941d131cbb",
        {
            "video",
            "prompt",
            "strength",
            "enforce_inference_steps",
            "conditioning_frames",
            "controlnet_conditioning_scale",
        },
    ),
    (
        "pipelines/cogvideo/pipeline_cogvideox_video2video.py",
        "CogVideoXVideoToVideoPipeline",
        "7c211c34fe2816a92fffc99a6aa1ec8f5899c77524d9774f52a538e1285f87fe",
        {"video", "prompt", "strength", "max_sequence_length"},
    ),
)


class RecordingVideoPipeline:
    def __init__(self, pipeline_class, frame_count, *, controlnet=False):
        self._modiff_video_pipeline_class = pipeline_class
        self._modiff_video_repo = ANIMATEDIFF_BASE_REPO
        self._modiff_video_revision = ANIMATEDIFF_BASE_REVISION
        self._modiff_video_motion_adapter_repo = ANIMATEDIFF_MOTION_REPO
        self._modiff_video_motion_adapter_revision = ANIMATEDIFF_MOTION_REVISION
        if controlnet:
            self._modiff_video_conditioning_component_class = "ControlNetModel"
            self._modiff_video_conditioning_repo = ANIMATEDIFF_CONTROLNET_REPO
            self._modiff_video_conditioning_revision = ANIMATEDIFF_CONTROLNET_REVISION
        self._execution_device = "cpu"
        self.frame_count = frame_count
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(frames=[[f"frame-{index}" for index in range(self.frame_count)]])


class DiffusersVideoExtendedRouteTests(unittest.TestCase):
    def test_exact_pinned_upstream_sources_and_call_signatures_are_preserved(self):
        import diffusers

        self.assertEqual(
            PINNED_DIFFUSERS_REVISION,
            "2f7e0154a9db246e95c9ede43edba7db5b130805",
        )
        diffusers_root = Path(diffusers.__file__).resolve().parent
        for relative_path, class_name, expected_digest, required_parameters in (
            UPSTREAM_PIPELINE_SOURCES
        ):
            with self.subTest(pipeline=class_name):
                source_path = diffusers_root / relative_path
                self.assertTrue(source_path.is_file())
                self.assertEqual(
                    source_sha256(source_path),
                    expected_digest,
                )
                tree = ast.parse(source_path.read_text(encoding="utf-8"))
                pipeline_node = next(
                    node
                    for node in tree.body
                    if isinstance(node, ast.ClassDef) and node.name == class_name
                )
                call_node = next(
                    node
                    for node in pipeline_node.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "__call__"
                )
                parameters = {
                    argument.arg
                    for argument in (
                        *call_node.args.posonlyargs,
                        *call_node.args.args,
                        *call_node.args.kwonlyargs,
                    )
                }
                self.assertTrue(required_parameters.issubset(parameters))

    def test_extended_routes_are_additive_exact_class_adapters_with_generic_modes(self):
        expected = {
            "AnimateDiffPAGPipeline": ("animatediff-pag", ("text_to_video",)),
            "AnimateDiffVideoToVideoPipeline": (
                "animatediff-video-to-video",
                ("video_to_video",),
            ),
            "AnimateDiffControlNetPipeline": (
                "animatediff-controlnet",
                ("control_to_video",),
            ),
            "AnimateDiffVideoToVideoControlNetPipeline": (
                "animatediff-video-to-video-controlnet",
                ("control_video_to_video",),
            ),
            "CogVideoXVideoToVideoPipeline": (
                "cogvideox-2b-video-to-video",
                ("video_to_video",),
            ),
        }

        for pipeline_class, (adapter_id, modes) in expected.items():
            with self.subTest(pipeline_class=pipeline_class):
                adapter = VIDEO_PIPELINE_ADAPTERS[pipeline_class]
                self.assertEqual(adapter.id, adapter_id)
                self.assertEqual(adapter.diffusers_class, pipeline_class)
                self.assertEqual(adapter.modes, modes)
                self.assertEqual(tuple(VIDEO_MODE_FIELD_CONTRACTS[pipeline_class]), modes)

        self.assertEqual(VIDEO_PIPELINE_ADAPTERS["AnimateDiffPipeline"].modes, ("text_to_video",))
        self.assertEqual(VIDEO_PIPELINE_ADAPTERS["CogVideoXPipeline"].modes, ("text_to_video",))

    def test_control_video_field_contract_is_model_neutral_and_required_only_by_control_routes(self):
        self.assertIn("control_video", Generate.params)
        self.assertFalse(Generate.params["control_video"]["required"])
        control = VIDEO_MODE_FIELD_CONTRACTS["AnimateDiffControlNetPipeline"]["control_to_video"]
        combined = VIDEO_MODE_FIELD_CONTRACTS["AnimateDiffVideoToVideoControlNetPipeline"]["control_video_to_video"]
        self.assertEqual(control.required_fields, ("control_video",))
        self.assertEqual(combined.required_fields, ("video", "control_video"))
        self.assertIn("conditioning_scale", control.visible_fields)
        self.assertIn("strength", combined.visible_fields)

    def test_animatediff_extended_loaders_use_exact_classes_and_only_reviewed_artifacts(self):
        for pipeline_class, _mode, uses_controlnet in ANIMATEDIFF_EXTENDED_ROUTES:
            with self.subTest(pipeline_class=pipeline_class):
                pipeline = SimpleNamespace(vae=SimpleNamespace(enable_slicing=MagicMock()))
                motion_adapter = object()
                scheduler = object()
                controlnet = object()
                node = LoadPipeline(f"load-{pipeline_class}")
                with (
                    patch("diffusers.MotionAdapter.from_pretrained", return_value=motion_adapter) as load_motion,
                    patch("diffusers.DDIMScheduler.from_pretrained", return_value=scheduler) as load_scheduler,
                    patch("diffusers.ControlNetModel.from_pretrained", return_value=controlnet) as load_controlnet,
                    patch(
                        f"diffusers.{pipeline_class}.from_pretrained",
                        return_value=pipeline,
                    ) as load_pipeline,
                    patch("modules.DiffusersVideo.main.apply_pipeline_offload"),
                    patch("modules.DiffusersRuntime.main.apply_execution_recipe_to_pipeline"),
                    patch.object(node, "mm_add"),
                ):
                    result = node.execute(
                        pipeline_class=pipeline_class,
                        model_id={"source": "hub", "value": ANIMATEDIFF_BASE_REPO},
                        revision=ANIMATEDIFF_BASE_REVISION,
                        motion_adapter_id={"source": "hub", "value": ANIMATEDIFF_MOTION_REPO},
                        motion_adapter_revision=ANIMATEDIFF_MOTION_REVISION,
                        dtype="float16",
                        device="cpu",
                        offload_mode="model_cpu",
                    )

                self.assertIs(result["pipeline"], pipeline)
                self.assertEqual(pipeline._modiff_video_pipeline_class, pipeline_class)
                load_motion.assert_called_once()
                load_scheduler.assert_called_once()
                load_args, load_kwargs = load_pipeline.call_args
                self.assertEqual(load_args, (ANIMATEDIFF_BASE_REPO,))
                self.assertIs(load_kwargs["motion_adapter"], motion_adapter)
                self.assertIs(load_kwargs["scheduler"], scheduler)
                self.assertIs(load_kwargs["use_safetensors"], True)
                self.assertNotIn("trust_remote_code", load_kwargs)
                if uses_controlnet:
                    control_args, control_kwargs = load_controlnet.call_args
                    self.assertEqual(control_args, (ANIMATEDIFF_CONTROLNET_REPO,))
                    self.assertEqual(control_kwargs["revision"], ANIMATEDIFF_CONTROLNET_REVISION)
                    self.assertIs(control_kwargs["use_safetensors"], True)
                    self.assertIs(load_kwargs["controlnet"], controlnet)
                    self.assertEqual(pipeline._modiff_video_conditioning_repo, ANIMATEDIFF_CONTROLNET_REPO)
                    self.assertEqual(
                        pipeline._modiff_video_conditioning_revision,
                        ANIMATEDIFF_CONTROLNET_REVISION,
                    )
                else:
                    load_controlnet.assert_not_called()
                    self.assertNotIn("controlnet", load_kwargs)

    def test_animatediff_extended_execution_translates_generic_media_to_exact_signatures(self):
        source = [Image.new("RGB", (64, 64), "red") for _ in range(8)]
        control = [Image.new("RGB", (64, 64), "black") for _ in range(8)]

        for pipeline_class, mode, uses_controlnet in ANIMATEDIFF_EXTENDED_ROUTES:
            with self.subTest(pipeline_class=pipeline_class):
                pipeline = RecordingVideoPipeline(pipeline_class, 8, controlnet=uses_controlnet)
                values = {
                    "pipeline": pipeline,
                    "mode": mode,
                    "prompt": "A paper kite follows a smooth circular path.",
                    "negative_prompt": "flicker",
                    "width": 512,
                    "height": 512,
                    "num_frames": 8,
                    "num_inference_steps": 4,
                    "guidance_scale": 7.5,
                    "seed": 31,
                    "output_type": "pil",
                }
                if mode in {"video_to_video", "control_video_to_video"}:
                    values.update(video=source, strength=0.65)
                if mode in {"control_to_video", "control_video_to_video"}:
                    values.update(control_video=control, conditioning_scale=0.75)
                if pipeline_class == "AnimateDiffPAGPipeline":
                    values.update(pag_scale=2.5, pag_adaptive_scale=0.25)

                result = Generate().execute(**values)

                self.assertEqual(result["frames_out"], 8)
                received = pipeline.calls[0]
                if mode in {"video_to_video", "control_video_to_video"}:
                    self.assertEqual(received["video"], source)
                    self.assertEqual(received["strength"], 0.65)
                    self.assertIs(received["enforce_inference_steps"], False)
                    self.assertNotIn("num_frames", received)
                else:
                    self.assertEqual(received["num_frames"], 8)
                    self.assertNotIn("video", received)
                if uses_controlnet:
                    self.assertEqual(received["conditioning_frames"], control)
                    self.assertEqual(received["controlnet_conditioning_scale"], 0.75)
                else:
                    self.assertNotIn("conditioning_frames", received)
                if pipeline_class == "AnimateDiffPAGPipeline":
                    self.assertEqual(received["pag_scale"], 2.5)
                    self.assertEqual(received["pag_adaptive_scale"], 0.25)
                else:
                    self.assertNotIn("pag_scale", received)

    def test_animatediff_extended_media_and_artifact_failures_preflight_before_torch(self):
        source = [Image.new("RGB", (64, 64), "red") for _ in range(8)]
        control = [Image.new("RGB", (64, 64), "black") for _ in range(8)]
        invalid = (
            (
                RecordingVideoPipeline("AnimateDiffVideoToVideoPipeline", 8),
                {"mode": "video_to_video"},
                "requires a source video",
            ),
            (
                RecordingVideoPipeline("AnimateDiffControlNetPipeline", 8, controlnet=True),
                {"mode": "control_to_video"},
                "requires a control video",
            ),
            (
                RecordingVideoPipeline(
                    "AnimateDiffVideoToVideoControlNetPipeline",
                    8,
                    controlnet=True,
                ),
                {
                    "mode": "control_video_to_video",
                    "video": source,
                    "control_video": control[:-1],
                    "num_frames": 8,
                },
                "control video contains 7 frames; expected 8",
            ),
            (
                RecordingVideoPipeline(
                    "AnimateDiffVideoToVideoControlNetPipeline",
                    8,
                    controlnet=True,
                ),
                {
                    "mode": "control_video_to_video",
                    "video": source,
                    "control_video": [Image.new("RGB", (32, 64), "black") for _ in range(8)],
                    "num_frames": 8,
                },
                "matching spatial dimensions",
            ),
            (
                RecordingVideoPipeline("AnimateDiffPAGPipeline", 8),
                {"mode": "text_to_video", "pag_scale": 21},
                "PAG scale must be finite and from 0 through 20",
            ),
        )

        with patch.dict(sys.modules, {"torch": None}):
            for pipeline, values, message in invalid:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        Generate().execute(
                            pipeline=pipeline,
                            prompt="A fixed-camera documentary shot.",
                            **values,
                        )
                    self.assertEqual(pipeline.calls, [])

        wrong_controlnet = RecordingVideoPipeline("AnimateDiffControlNetPipeline", 8, controlnet=True)
        wrong_controlnet._modiff_video_conditioning_revision = "0" * 40
        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaisesRegex(ValueError, "reviewed ControlNet assembly"):
                Generate().execute(
                    pipeline=wrong_controlnet,
                    mode="control_to_video",
                    prompt="A fixed-camera documentary shot.",
                    control_video=control,
                    num_frames=8,
                )
        self.assertEqual(wrong_controlnet.calls, [])

    def test_cogvideox_video_to_video_loads_the_exact_reused_pipeline_artifact(self):
        pipeline = SimpleNamespace(vae=SimpleNamespace(enable_tiling=MagicMock()))
        node = LoadPipeline("cogvideox-v2v-loader")
        with (
            patch(
                "diffusers.CogVideoXVideoToVideoPipeline.from_pretrained",
                return_value=pipeline,
            ) as from_pretrained,
            patch("modules.DiffusersVideo.main.apply_pipeline_offload"),
            patch("modules.DiffusersRuntime.main.apply_execution_recipe_to_pipeline"),
            patch.object(node, "mm_add"),
        ):
            result = node.execute(
                pipeline_class="CogVideoXVideoToVideoPipeline",
                model_id={"source": "hub", "value": COGVIDEOX_2B_REPO},
                revision=COGVIDEOX_2B_REVISION,
                dtype="float16",
                device="cpu",
                offload_mode="model_cpu",
            )

        self.assertIs(result["pipeline"], pipeline)
        self.assertEqual(pipeline._modiff_video_pipeline_class, "CogVideoXVideoToVideoPipeline")
        load_args, load_kwargs = from_pretrained.call_args
        self.assertEqual(load_args, (COGVIDEOX_2B_REPO,))
        self.assertEqual(load_kwargs["revision"], COGVIDEOX_2B_REVISION)
        self.assertIs(load_kwargs["use_safetensors"], True)
        self.assertNotIn("trust_remote_code", load_kwargs)

    def test_cogvideox_video_to_video_translates_source_video_without_inventing_num_frames(self):
        source = [Image.new("RGB", (72, 48), "red") for _ in range(9)]

        class Pipeline:
            _modiff_video_pipeline_class = "CogVideoXVideoToVideoPipeline"
            _modiff_video_repo = COGVIDEOX_2B_REPO
            _modiff_video_revision = COGVIDEOX_2B_REVISION
            _execution_device = "cpu"

            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(frames=[[f"frame-{index}" for index in range(9)]])

        pipeline = Pipeline()
        result = Generate().execute(
            pipeline=pipeline,
            mode="video_to_video",
            prompt="Restyle the same camera move as charcoal animation.",
            video=source,
            width=720,
            height=480,
            num_frames=9,
            num_inference_steps=20,
            guidance_scale=6,
            strength=0.7,
            max_sequence_length=226,
            seed=41,
            output_type="pil",
        )

        self.assertEqual(result["frames_out"], 9)
        received = pipeline.calls[0]
        self.assertEqual(received["video"], source)
        self.assertEqual(received["strength"], 0.7)
        self.assertNotIn("num_frames", received)
        self.assertEqual(received["max_sequence_length"], 226)

        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaisesRegex(ValueError, "contains 8 frames; expected 9"):
                Generate().execute(
                    pipeline=pipeline,
                    mode="video_to_video",
                    prompt="Restyle the same camera move as charcoal animation.",
                    video=source[:-1],
                    width=720,
                    height=480,
                    num_frames=9,
                )


if __name__ == "__main__":
    unittest.main()
