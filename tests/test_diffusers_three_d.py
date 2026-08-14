import importlib.util
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from modules.DiffusersThreeD.main import (
    DEFAULT_THREE_D_CONTRACT,
    SHAP_E_MODE,
    SHAP_E_PIPELINE_CLASS,
    SHAP_E_RENDERER_SAFE_SUBFOLDER,
    SHAP_E_REPO,
    SHAP_E_REVISION,
    GenerateRenderedArtifact,
    LoadPipeline,
    _load_safe_shap_e_components,
    _require_exact_selection,
)


requires_transformers = unittest.skipUnless(
    importlib.util.find_spec("transformers"),
    "requires the staged optional Transformers runtime",
)


class FakeShapEPipeline:
    _modiff_three_d_pipeline_class = SHAP_E_PIPELINE_CLASS
    _modiff_three_d_mode = SHAP_E_MODE
    _modiff_three_d_repo = SHAP_E_REPO
    _modiff_three_d_revision = SHAP_E_REVISION
    device = "cpu"

    def __init__(self, frames=None, **components):
        self.components = components
        self.frames = frames or [Image.new("RGB", (256, 256)) for _ in range(20)]
        self.call_kwargs = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise AssertionError("The reviewed Shap-E loader must assemble safe components explicitly.")

    def __call__(self, **kwargs):
        self.call_kwargs = kwargs
        return SimpleNamespace(images=[self.frames])


class DiffusersThreeDTests(unittest.TestCase):
    def test_exact_artifact_and_contract_are_fail_closed(self):
        self.assertEqual(
            _require_exact_selection(
                {"source": "hub", "value": SHAP_E_REPO},
                SHAP_E_REVISION,
            ),
            (SHAP_E_REPO, SHAP_E_REVISION),
        )
        self.assertEqual(DEFAULT_THREE_D_CONTRACT["outputContract"], "rendered_orbit")
        self.assertEqual(DEFAULT_THREE_D_CONTRACT["fieldParams"]["frame_size"]["max"], 256)
        for model_id, revision in (
            ({"source": "local", "value": SHAP_E_REPO}, SHAP_E_REVISION),
            ({"source": "hub", "value": "owner/other"}, SHAP_E_REVISION),
            ({"source": "hub", "value": SHAP_E_REPO, "extra": True}, SHAP_E_REVISION),
            ({"source": "hub", "value": SHAP_E_REPO}, "0" * 40),
        ):
            with self.subTest(model_id=model_id, revision=revision):
                with self.assertRaises(ValueError):
                    _require_exact_selection(model_id, revision)

    @patch("modules.DiffusersThreeD.main.local_files_only", return_value=False)
    @patch("diffusers.pipelines.shap_e.renderer.ShapERenderer.from_pretrained")
    @patch("diffusers.HeunDiscreteScheduler.from_pretrained")
    @patch("transformers.CLIPTokenizer.from_pretrained")
    @patch("transformers.CLIPTextModelWithProjection.from_pretrained")
    @patch("diffusers.PriorTransformer.from_pretrained")
    @requires_transformers
    def test_safe_component_assembly_never_selects_legacy_renderer_pickle(
        self,
        prior_loader,
        text_loader,
        tokenizer_loader,
        scheduler_loader,
        renderer_loader,
        _offline,
    ):
        components = _load_safe_shap_e_components(
            SHAP_E_REPO,
            SHAP_E_REVISION,
            dtype="float16",
            low_cpu_mem_usage=True,
        )
        self.assertEqual(set(components), {"prior", "text_encoder", "tokenizer", "scheduler", "shap_e_renderer"})
        for loader in (prior_loader, text_loader, renderer_loader):
            kwargs = loader.call_args.kwargs
            self.assertTrue(kwargs["use_safetensors"])
            self.assertEqual(kwargs["variant"], "fp16")
            self.assertEqual(kwargs["revision"], SHAP_E_REVISION)
            self.assertTrue(kwargs["low_cpu_mem_usage"])
        self.assertEqual(prior_loader.call_args.kwargs["subfolder"], "prior")
        self.assertEqual(text_loader.call_args.kwargs["subfolder"], "text_encoder")
        self.assertEqual(renderer_loader.call_args.kwargs["subfolder"], SHAP_E_RENDERER_SAFE_SUBFOLDER)
        self.assertNotEqual(renderer_loader.call_args.kwargs["subfolder"], "shap_e_renderer")
        self.assertEqual(tokenizer_loader.call_args.kwargs["subfolder"], "tokenizer")
        self.assertEqual(scheduler_loader.call_args.kwargs["subfolder"], "scheduler")

    def test_loader_assembles_pipeline_without_pipeline_from_pretrained(self):
        components = {"component": object()}
        node = LoadPipeline("three-d-loader")
        node.progress = Mock()
        node.mm_add = Mock()
        node.diffusers_loading_progress = Mock(return_value=nullcontext())
        with (
            patch("diffusers.ShapEPipeline", FakeShapEPipeline),
            patch("modules.DiffusersThreeD.main._load_safe_shap_e_components", return_value=components) as loader,
            patch(
                "modules.DiffusersRuntime.main.loader_runtime_options",
                return_value=({}, "cpu", "none", {}),
            ),
            patch("modules.DiffusersThreeD.main.apply_pipeline_offload") as apply_offload,
        ):
            result = node.execute(
                model_id={"source": "hub", "value": SHAP_E_REPO},
                revision=SHAP_E_REVISION,
                pipeline_class=SHAP_E_PIPELINE_CLASS,
                mode=SHAP_E_MODE,
                dtype="float16",
                low_cpu_mem_usage=True,
            )
        loader.assert_called_once()
        self.assertEqual(result["resolved_artifact"], SHAP_E_REPO)
        self.assertEqual(result["pipeline"].components, components)
        self.assertEqual(result["pipeline"]._modiff_three_d_revision, SHAP_E_REVISION)
        apply_offload.assert_called_once()
        node.mm_add.assert_called_once_with(result["pipeline"], priority=2)

    def test_loader_rejects_on_load_quantization_or_device_maps(self):
        node = LoadPipeline("three-d-loader-recipe")
        with patch(
            "modules.DiffusersRuntime.main.loader_runtime_options",
            return_value=({}, "cuda", "none", {"device_map": "cuda"}),
        ):
            with self.assertRaisesRegex(ValueError, "does not support on-load quantization or device maps"):
                node.execute(
                    model_id={"source": "hub", "value": SHAP_E_REPO},
                    revision=SHAP_E_REVISION,
                    pipeline_class=SHAP_E_PIPELINE_CLASS,
                    mode=SHAP_E_MODE,
                )

    def test_generation_uses_exact_bounded_rendered_orbit_signature(self):
        pipeline = FakeShapEPipeline()
        result = GenerateRenderedArtifact("three-d-generate").execute(
            pipeline=pipeline,
            three_d_contract=DEFAULT_THREE_D_CONTRACT,
            prompt="a wooden toy sailboat",
            seed=17,
            num_inference_steps=64,
            guidance_scale=15,
            frame_size=256,
        )
        self.assertEqual(result["video"], pipeline.frames)
        self.assertEqual((result["width_out"], result["height_out"], result["frames_out"]), (256, 256, 20))
        self.assertEqual(
            {key: value for key, value in pipeline.call_kwargs.items() if key != "generator"},
            {
                "prompt": "a wooden toy sailboat",
                "num_images_per_prompt": 1,
                "num_inference_steps": 64,
                "guidance_scale": 15.0,
                "frame_size": 256,
                "output_type": "pil",
                "return_dict": True,
            },
        )

    def test_generation_rejects_stale_identity_and_unbounded_output(self):
        stale = FakeShapEPipeline()
        stale._modiff_three_d_revision = "0" * 40
        with self.assertRaisesRegex(ValueError, "artifact identity"):
            GenerateRenderedArtifact().execute(
                pipeline=stale,
                three_d_contract=DEFAULT_THREE_D_CONTRACT,
                prompt="test",
            )
        malformed = FakeShapEPipeline(frames=[Image.new("RGB", (128, 256))])
        with self.assertRaisesRegex(ValueError, "square frame contract"):
            GenerateRenderedArtifact().execute(
                pipeline=malformed,
                three_d_contract=DEFAULT_THREE_D_CONTRACT,
                prompt="test",
                frame_size=256,
            )


if __name__ == "__main__":
    unittest.main()
