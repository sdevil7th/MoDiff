import ast
import hashlib
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

import modules as module_registry
from modiff.model_artifact_catalog import catalog_repository_pin, catalog_revision
from modiff.modular_workflow_contracts import PINNED_DIFFUSERS_REVISION
from modiff.optional_runtimes import (
    OPTIONAL_RUNTIME_PROFILES,
    TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID,
)
from modiff.studio_execution_specs import (
    CHROMA1_HD_DIFFUSERS_FILES,
    studio_capability_definitions,
    studio_execution_spec_for_pair,
    validate_studio_execution_specs,
)
from modules.DiffusersImage.main import (
    CHROMA1_HD_REPO,
    IMAGE_PIPELINE_ADAPTERS,
    Edit,
    Inpaint,
    LoadPipeline as LoadImagePipeline,
    _tag_image_pipeline,
    image_pipeline_contract,
)
from modules.DiffusersVideo.main import (
    VIDEO_PIPELINE_ADAPTERS,
    GenerateVideoAudio,
    LoadPipeline as LoadVideoPipeline,
    get_video_mode_field_contract,
)


CHROMA_IMG2IMG_SOURCE_SHA256 = "4dfa9751d317efb5cbae8faadc4a77b53312a98a9920675afb44bab01aae0a09"
CHROMA_INPAINT_SOURCE_SHA256 = "b149fa04c9ae0b4165d761aa380b484e7a8d57f62589c7139fd41040f41e78ae"
LTX2_SOURCE_SHA256 = "603460f8e85819e8f5884667af8c26e873710fdf6c310a35a322b84de7d55c09"
CHROMA_REVISION = "0e0c60ece1e82b17cb7f77342d765ba5024c40c0"
LTX2_REPO = "Lightricks/LTX-2"
LTX2_REVISION = "47da56e2ad66ce4125a9922b4a8826bf407f9d0a"
_UNSET = object()


def _pinned_call_node(source_path: Path, class_name: str) -> ast.FunctionDef:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    pipeline_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(
        node
        for node in pipeline_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__call__"
    )


def _parameter_names(call_node: ast.FunctionDef) -> set[str]:
    return {
        argument.arg
        for argument in (
            *call_node.args.posonlyargs,
            *call_node.args.args,
            *call_node.args.kwonlyargs,
        )
    }


def _tag_chroma(pipeline, pipeline_class: str, mode: str):
    adapter = IMAGE_PIPELINE_ADAPTERS[pipeline_class]
    _tag_image_pipeline(
        pipeline,
        adapter,
        mode,
        CHROMA1_HD_REPO,
        "hub",
        catalog_revision(CHROMA1_HD_REPO),
    )
    return pipeline


class ChromaImg2ImgPipeline:
    _execution_device = "cpu"

    def __init__(self):
        self.calls = []

    def __call__(
        self,
        *,
        prompt,
        negative_prompt,
        image,
        height,
        width,
        num_inference_steps,
        guidance_scale,
        strength,
        generator,
        output_type,
        return_dict,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        max_sequence_length,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": image,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "strength": strength,
                "generator": generator,
                "output_type": output_type,
                "return_dict": return_dict,
                "callback_on_step_end": callback_on_step_end,
                "callback_on_step_end_tensor_inputs": callback_on_step_end_tensor_inputs,
                "max_sequence_length": max_sequence_length,
            }
        )
        return SimpleNamespace(images=[Image.new("RGB", (width, height), "green")])


class ChromaInpaintPipeline:
    _execution_device = "cpu"

    def __init__(self):
        self.calls = []

    def __call__(
        self,
        *,
        prompt,
        negative_prompt,
        true_cfg_scale=_UNSET,
        image,
        mask_image,
        height,
        width,
        padding_mask_crop=None,
        strength,
        num_inference_steps,
        guidance_scale,
        generator,
        output_type,
        return_dict,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        max_sequence_length,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "true_cfg_scale": true_cfg_scale,
                "image": image,
                "mask_image": mask_image,
                "height": height,
                "width": width,
                "padding_mask_crop": padding_mask_crop,
                "strength": strength,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
                "output_type": output_type,
                "return_dict": return_dict,
                "callback_on_step_end": callback_on_step_end,
                "callback_on_step_end_tensor_inputs": callback_on_step_end_tensor_inputs,
                "max_sequence_length": max_sequence_length,
            }
        )
        return SimpleNamespace(images=[Image.new("RGB", (width, height), "green")])


class LTX2Pipeline:
    _execution_device = "cpu"
    _modiff_video_pipeline_class = "LTX2Pipeline"
    _modiff_video_repo = LTX2_REPO
    _modiff_video_revision = LTX2_REVISION
    vocoder = SimpleNamespace(config={"output_sampling_rate": 24000})

    def __init__(self):
        self.calls = []

    def __call__(
        self,
        *,
        prompt,
        negative_prompt,
        height,
        width,
        num_frames,
        frame_rate,
        num_inference_steps,
        guidance_scale,
        generator,
        output_type,
        return_dict,
        attention_kwargs,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        max_sequence_length,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "frame_rate": frame_rate,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
                "output_type": output_type,
                "return_dict": return_dict,
                "attention_kwargs": attention_kwargs,
                "callback_on_step_end": callback_on_step_end,
                "callback_on_step_end_tensor_inputs": callback_on_step_end_tensor_inputs,
                "max_sequence_length": max_sequence_length,
            }
        )
        return SimpleNamespace(
            frames=[["frame-a", "frame-b"]],
            audio=np.zeros((2, 240), dtype=np.float32),
        )


class ChromaLTX2ExactClosureTests(unittest.TestCase):
    def test_exact_pinned_sources_and_call_surfaces_are_preserved(self):
        import diffusers

        self.assertEqual(PINNED_DIFFUSERS_REVISION, "90b4e34e79a86ec5e7f2437634fe95ecd2108796")
        root = Path(diffusers.__file__).resolve().parent / "pipelines"
        cases = (
            (
                root / "chroma" / "pipeline_chroma_img2img.py",
                "ChromaImg2ImgPipeline",
                CHROMA_IMG2IMG_SOURCE_SHA256,
                {
                    "prompt",
                    "negative_prompt",
                    "image",
                    "height",
                    "width",
                    "strength",
                    "guidance_scale",
                    "max_sequence_length",
                },
            ),
            (
                root / "chroma" / "pipeline_chroma_inpainting.py",
                "ChromaInpaintPipeline",
                CHROMA_INPAINT_SOURCE_SHA256,
                {
                    "prompt",
                    "negative_prompt",
                    "true_cfg_scale",
                    "image",
                    "mask_image",
                    "height",
                    "width",
                    "padding_mask_crop",
                    "strength",
                    "guidance_scale",
                    "max_sequence_length",
                },
            ),
            (
                root / "ltx2" / "pipeline_ltx2.py",
                "LTX2Pipeline",
                LTX2_SOURCE_SHA256,
                {
                    "prompt",
                    "negative_prompt",
                    "height",
                    "width",
                    "num_frames",
                    "frame_rate",
                    "guidance_scale",
                    "audio_guidance_scale",
                    "output_type",
                    "max_sequence_length",
                },
            ),
        )
        for source_path, class_name, digest, required_parameters in cases:
            with self.subTest(pipeline=class_name):
                self.assertTrue(source_path.is_file())
                self.assertEqual(hashlib.sha256(source_path.read_bytes()).hexdigest(), digest)
                self.assertTrue(
                    required_parameters.issubset(_parameter_names(_pinned_call_node(source_path, class_name)))
                )

        inpaint_call = _pinned_call_node(cases[1][0], "ChromaInpaintPipeline")
        true_cfg_reads = [
            node
            for node in ast.walk(inpaint_call)
            if isinstance(node, ast.Name) and node.id == "true_cfg_scale" and isinstance(node.ctx, ast.Load)
        ]
        self.assertEqual(true_cfg_reads, [])
        self.assertIn("LTX2PipelineOutput(frames=video, audio=audio)", cases[2][0].read_text(encoding="utf-8"))

    def test_exact_adapters_reuse_only_the_admitted_artifacts(self):
        img2img = IMAGE_PIPELINE_ADAPTERS["ChromaImg2ImgPipeline"]
        inpaint = IMAGE_PIPELINE_ADAPTERS["ChromaInpaintPipeline"]
        ltx2 = VIDEO_PIPELINE_ADAPTERS["LTX2Pipeline"]

        self.assertEqual(img2img.modes, frozenset({"edit_image"}))
        self.assertEqual(inpaint.modes, frozenset({"inpaint", "outpaint"}))
        self.assertEqual(inpaint.rejected_input_fields, ("true_cfg_scale",))
        self.assertTrue(img2img.safe_serialization_required)
        self.assertTrue(inpaint.safe_serialization_required)
        self.assertEqual(img2img.default_repo, CHROMA1_HD_REPO)
        self.assertEqual(inpaint.default_repo, CHROMA1_HD_REPO)
        self.assertEqual(ltx2.default_repo, LTX2_REPO)
        self.assertEqual(ltx2.modes, ("text_to_video",))
        self.assertEqual(ltx2.output_media, ("video", "audio"))

        self.assertEqual(catalog_repository_pin(CHROMA1_HD_REPO)["revision"], CHROMA_REVISION)
        self.assertEqual(catalog_repository_pin(CHROMA1_HD_REPO)["license"], "apache-2.0")
        self.assertEqual(catalog_repository_pin(LTX2_REPO)["revision"], LTX2_REVISION)
        self.assertEqual(
            image_pipeline_contract(img2img, "edit_image")["actions"],
            {"Edit": ["edit_image"]},
        )
        self.assertEqual(
            image_pipeline_contract(inpaint, "outpaint")["actions"],
            {"Inpaint": ["inpaint", "outpaint"]},
        )
        self.assertEqual(
            get_video_mode_field_contract(ltx2, "text_to_video").visible_fields,
            ("frame_rate",),
        )

    def test_exact_hidden_candidate_specs_reuse_artifacts_and_generic_contracts(self):
        self.assertEqual(len(validate_studio_execution_specs(module_registry.MODULE_MAP)), 187)
        capabilities = studio_capability_definitions()
        cases = {
            ("ChromaImg2ImgPipeline", "edit_image"): (
                "chroma1-hd-img2img:edit-image:v1",
                "chroma1-hd-img2img:direct",
            ),
            ("ChromaInpaintPipeline", "inpaint"): (
                "chroma1-hd-inpaint:inpaint:v1",
                "chroma1-hd-inpaint:direct",
            ),
            ("ChromaInpaintPipeline", "outpaint"): (
                "chroma1-hd-inpaint:outpaint:v1",
                "chroma1-hd-inpaint:direct",
            ),
            ("LTX2Pipeline", "text_to_video"): (
                "ltx2-standard:text-to-video:v1",
                "ltx2-standard:direct",
            ),
        }
        for pair, (spec_id, profile_id) in cases.items():
            with self.subTest(pair=pair):
                specification = studio_execution_spec_for_pair(*pair)
                capability = capabilities[pair[0]]
                self.assertEqual(specification["id"], spec_id)
                self.assertEqual(specification["executionProfileId"], profile_id)
                self.assertEqual(specification["pipelineClass"], pair[0])
                self.assertEqual(capability["executionStatus"], "expert_only")
                self.assertFalse(capability["autoEligible"])
                self.assertTrue(capability["templateEligible"])
                self.assertFalse(capability["galleryEligible"])
                self.assertFalse(capability["liveProof"])

        edit = studio_execution_spec_for_pair("ChromaImg2ImgPipeline", "edit_image")
        inpaint = studio_execution_spec_for_pair("ChromaInpaintPipeline", "inpaint")
        outpaint = studio_execution_spec_for_pair("ChromaInpaintPipeline", "outpaint")
        for specification in (edit, inpaint, outpaint):
            capability = capabilities[specification["modelType"]]
            self.assertEqual(capability["downloadFiles"], CHROMA1_HD_DIFFUSERS_FILES)
            self.assertEqual(capability["revisionCandidates"], [CHROMA_REVISION])
            self.assertFalse(capability["supportsLora"])
            self.assertNotIn(
                ("diffusersImageEdit", "reference_strength", "conditioningScale"),
                specification["bindings"],
            )
            self.assertNotIn(
                ("diffusersImageInpaint", "reference_strength", "conditioningScale"),
                specification["bindings"],
            )
            self.assertTrue(
                all(parameter != "true_cfg_scale" for _role, parameter, _source in specification["bindings"])
            )
        self.assertIn(("outpaintCanvas", "mask_image", "diffusersImageInpaint", "mask_image"), outpaint["edges"])

        ltx2 = studio_execution_spec_for_pair("LTX2Pipeline", "text_to_video")
        ltx2_capability = capabilities["LTX2Pipeline"]
        self.assertNotIn("downloadFiles", ltx2_capability)
        self.assertEqual(ltx2_capability["revisionCandidates"], [LTX2_REVISION])
        self.assertEqual(ltx2_capability["outputMedia"], ["video", "audio"])
        self.assertIn(
            ("wanGenerate", "audio", "videoExport", "audio"),
            ltx2["edges"],
        )
        self.assertIn(
            ("videoExport", "modules.Video.ExportWithAudio", 640, -80),
            ltx2["roles"],
        )

    def test_optional_runtime_declares_every_exact_direct_symbol(self):
        profile = OPTIONAL_RUNTIME_PROFILES[TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID]
        expected = {"ChromaImg2ImgPipeline", "ChromaInpaintPipeline", "LTX2Pipeline"}
        self.assertTrue(expected.issubset(profile.pipeline_adapter_symbols))
        self.assertTrue(expected.issubset(profile.required_diffusers_symbols))

    def test_chroma_img2img_passes_the_exact_generic_edit_contract(self):
        source = Image.new("RGB", (48, 32), "red")
        pipeline = _tag_chroma(ChromaImg2ImgPipeline(), "ChromaImg2ImgPipeline", "edit_image")
        result = Edit("chroma-edit-test").execute(
            pipeline=pipeline,
            image=source,
            prompt="restore the faded photograph",
            negative_prompt="scratches",
            width=64,
            height=48,
            seed=7,
            num_inference_steps=20,
            guidance_scale=5.0,
            strength=0.65,
            max_sequence_length=256,
            output_type="pil",
        )

        self.assertEqual((result["width_out"], result["height_out"]), (64, 48))
        call = pipeline.calls[0]
        self.assertIs(call["image"], source)
        self.assertEqual(call["prompt"], "restore the faded photograph")
        self.assertEqual(call["negative_prompt"], "scratches")
        self.assertEqual(call["guidance_scale"], 5.0)
        self.assertEqual(call["strength"], 0.65)
        self.assertEqual(call["max_sequence_length"], 256)

    def test_chroma_inpaint_exact_call_uses_working_guidance_and_rejects_inert_true_cfg(self):
        source = Image.new("RGB", (64, 48), "red")
        mask = Image.new("L", source.size, 0)
        for x in range(32, 64):
            for y in range(48):
                mask.putpixel((x, y), 255)
        pipeline = _tag_chroma(ChromaInpaintPipeline(), "ChromaInpaintPipeline", "inpaint")
        result = Inpaint("chroma-inpaint-test").execute(
            pipeline=pipeline,
            image=source,
            mask_image=mask,
            prompt="replace the damaged right half",
            negative_prompt="seams",
            width=64,
            height=48,
            seed=11,
            num_inference_steps=18,
            guidance_scale=6.5,
            strength=0.7,
            padding_mask_crop=16,
            max_sequence_length=192,
            output_type="pil",
        )

        call = pipeline.calls[0]
        self.assertIs(call["image"], source)
        self.assertIs(call["mask_image"], mask)
        self.assertIs(call["true_cfg_scale"], _UNSET)
        self.assertEqual(call["guidance_scale"], 6.5)
        self.assertEqual(call["padding_mask_crop"], 16)
        self.assertEqual(result["images"][0].getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(result["images"][0].getpixel((63, 0)), (0, 128, 0))

        with self.assertRaisesRegex(ValueError, "rejects stale or inert upstream input field.*true_cfg_scale"):
            Inpaint("chroma-inpaint-stale-test").execute(
                pipeline=pipeline,
                image=source,
                mask_image=mask,
                prompt="stale graph",
                true_cfg_scale=2.0,
            )
        self.assertEqual(len(pipeline.calls), 1)

    def test_chroma_outpaint_uses_the_same_bounded_generic_inpaint_action(self):
        source = Image.new("RGB", (64, 48), "red")
        mask = Image.new("L", source.size, 255)
        pipeline = _tag_chroma(ChromaInpaintPipeline(), "ChromaInpaintPipeline", "outpaint")

        result = Inpaint("chroma-outpaint-test").execute(
            pipeline=pipeline,
            image=source,
            mask_image=mask,
            prompt="extend the scene",
            width=64,
            height=48,
        )

        self.assertEqual((result["width_out"], result["height_out"]), (64, 48))
        self.assertEqual(len(pipeline.calls), 1)

    def test_chroma_loaders_use_exact_pin_and_safe_serialization_without_downloads(self):
        for pipeline_class, mode in (
            ("ChromaImg2ImgPipeline", "edit_image"),
            ("ChromaInpaintPipeline", "inpaint"),
        ):
            with self.subTest(pipeline=pipeline_class):
                calls = []

                def from_pretrained(cls, repository, **kwargs):
                    calls.append((repository, kwargs))
                    return cls()

                fake_pipeline_class = type(pipeline_class, (), {"from_pretrained": classmethod(from_pretrained)})
                loader = LoadImagePipeline(f"{pipeline_class}-loader-test")
                loader.progress = Mock()
                loader.mm_add = Mock()
                loader.diffusers_loading_progress = lambda: nullcontext()
                with (
                    patch("modules.DiffusersImage.main.pipeline_class_from_name", return_value=fake_pipeline_class),
                    patch(
                        "modules.DiffusersImage.main.apply_pipeline_offload",
                        return_value=SimpleNamespace(method="test"),
                    ),
                    patch("modules.DiffusersRuntime.main.apply_execution_recipe_to_pipeline", return_value={}),
                    patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("download attempted")),
                    patch("huggingface_hub.snapshot_download", side_effect=AssertionError("download attempted")),
                ):
                    result = loader.execute(
                        pipeline_class=pipeline_class,
                        mode=mode,
                        model_id={"source": "hub", "value": CHROMA1_HD_REPO},
                        revision="",
                        dtype="bfloat16",
                        device="cpu",
                        quantization_mode="none",
                        quantized_components=[],
                        auto_offload=True,
                        offload_mode="model_cpu",
                    )

                self.assertIsInstance(result["pipeline"], fake_pipeline_class)
                self.assertEqual(calls[0][0], CHROMA1_HD_REPO)
                self.assertEqual(calls[0][1]["revision"], CHROMA_REVISION)
                self.assertTrue(calls[0][1]["use_safetensors"])
                self.assertNotIn("trust_remote_code", calls[0][1])
                self.assertEqual(result["pipeline"]._modiff_image_pipeline_class, pipeline_class)
                self.assertEqual(result["pipeline"]._modiff_image_mode, mode)

    def test_ltx2_loader_uses_exact_class_pin_and_safe_serialization_without_downloads(self):
        calls = []

        class FakeLTX2Pipeline:
            @classmethod
            def from_pretrained(cls, repository, **kwargs):
                calls.append((repository, kwargs))
                return cls()

        loader = LoadVideoPipeline("ltx2-direct-loader-test")
        loader.progress = Mock()
        loader.mm_add = Mock()
        with (
            patch("diffusers.LTX2Pipeline", FakeLTX2Pipeline, create=True),
            patch("modules.DiffusersRuntime.main.apply_execution_recipe_to_pipeline", return_value={}),
            patch("modules.DiffusersVideo.main.apply_pipeline_offload"),
            patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("download attempted")),
            patch("huggingface_hub.snapshot_download", side_effect=AssertionError("download attempted")),
        ):
            result = loader.execute(
                pipeline_class="LTX2Pipeline",
                model_id={"source": "hub", "value": LTX2_REPO},
                revision="",
                dtype="bfloat16",
                device="cpu",
                auto_offload=True,
                offload_mode="sequential_cpu",
            )

        self.assertIsInstance(result["pipeline"], FakeLTX2Pipeline)
        self.assertEqual(calls[0][0], LTX2_REPO)
        self.assertEqual(calls[0][1]["revision"], LTX2_REVISION)
        self.assertTrue(calls[0][1]["use_safetensors"])
        self.assertNotIn("trust_remote_code", calls[0][1])
        self.assertEqual(result["pipeline"]._modiff_video_pipeline_class, "LTX2Pipeline")
        self.assertEqual(result["pipeline"]._modiff_video_revision, LTX2_REVISION)

    def test_ltx2_direct_text_call_returns_truthful_video_and_singular_audio(self):
        pipeline = LTX2Pipeline()
        result = GenerateVideoAudio("ltx2-direct-generate-test").execute(
            pipeline=pipeline,
            mode="text_to_video",
            prompt="A quiet tram crosses a rain-wet bridge with synchronized city ambience.",
            negative_prompt="flicker",
            width=768,
            height=512,
            num_frames=121,
            frame_rate=24,
            num_inference_steps=30,
            guidance_scale=3.0,
            seed=17,
            max_sequence_length=512,
            output_type="pil",
        )

        self.assertEqual(result["video_out"], ["frame-a", "frame-b"])
        self.assertEqual(result["frames_out"], 2)
        self.assertEqual(result["sample_rate_out"], 24000)
        self.assertEqual(result["audio"]["channels"], 2)
        self.assertEqual(result["audio"]["samples"].shape, (2, 240))
        call = pipeline.calls[0]
        self.assertNotIn("conditions", call)
        self.assertEqual(call["frame_rate"], 24.0)
        self.assertEqual(call["max_sequence_length"], 512)

    def test_ltx2_direct_text_call_rejects_every_conditioning_input_before_execution(self):
        cases = (
            ({"reference_images": [Image.new("RGB", (32, 32), "black")]}, "image or video conditions"),
            ({"video": [Image.new("RGB", (32, 32), "black")]}, "image or video conditions"),
            ({"mask": [Image.new("L", (32, 32), 255)]}, "generic mask input"),
        )
        for extra, message in cases:
            pipeline = LTX2Pipeline()
            with self.subTest(extra=tuple(extra)), self.assertRaisesRegex(ValueError, message):
                GenerateVideoAudio("ltx2-condition-rejection-test").execute(
                    pipeline=pipeline,
                    mode="text_to_video",
                    prompt="unconditioned text generation",
                    width=768,
                    height=512,
                    **extra,
                )
            self.assertEqual(pipeline.calls, [])


if __name__ == "__main__":
    unittest.main()
