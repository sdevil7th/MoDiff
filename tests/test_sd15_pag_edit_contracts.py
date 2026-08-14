from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from modiff.diffusers_profiles import CONTRACT_ONLY_DIFFUSERS_PIPELINES
from modiff.model_artifact_catalog import catalog_revision
from modules.DiffusersImage import Edit, Inpaint, LoadPipeline
from modules.DiffusersImage.main import (
    IMAGE_PIPELINE_ADAPTERS,
    SD15_BASE_REPO,
    _tag_image_pipeline,
    image_pipeline_contract,
    resolve_image_model_selection,
    resolve_image_pipeline_revision,
)


PAG_EDIT_PIPELINES = {
    "StableDiffusionPAGImg2ImgPipeline": {
        "mode": "edit_image",
        "action": "Edit",
        "artifact_classes": (
            "StableDiffusionPipeline",
            "StableDiffusionPAGImg2ImgPipeline",
        ),
        "visible_fields": {
            "negative_prompt",
            "guidance_scale",
            "strength",
            "pag_scale",
            "pag_adaptive_scale",
        },
    },
    "StableDiffusionPAGInpaintPipeline": {
        "mode": "inpaint",
        "action": "Inpaint",
        "artifact_classes": (
            "StableDiffusionPipeline",
            "StableDiffusionPAGInpaintPipeline",
        ),
        "visible_fields": {
            "negative_prompt",
            "width",
            "height",
            "guidance_scale",
            "strength",
            "padding_mask_crop",
            "pag_scale",
            "pag_adaptive_scale",
        },
    },
}


def _tagged_pipeline(pipeline, pipeline_class, mode):
    _tag_image_pipeline(
        pipeline,
        IMAGE_PIPELINE_ADAPTERS[pipeline_class],
        mode,
        SD15_BASE_REPO,
        "hub",
        catalog_revision(SD15_BASE_REPO),
    )
    return pipeline


class StableDiffusion15PagEditContractTests(unittest.TestCase):
    def test_adapters_seal_exact_generic_fields_bounds_and_source(self):
        declared = {
            pipeline_class: (media_kind, repository, modes)
            for pipeline_class, media_kind, repository, modes in CONTRACT_ONLY_DIFFUSERS_PIPELINES
        }
        revision = catalog_revision(SD15_BASE_REPO)

        for pipeline_class, expected in PAG_EDIT_PIPELINES.items():
            with self.subTest(pipeline=pipeline_class):
                adapter = IMAGE_PIPELINE_ADAPTERS[pipeline_class]
                mode = expected["mode"]
                contract = image_pipeline_contract(adapter, mode)
                visible_fields = {
                    field
                    for field, params in contract["fieldParams"].items()
                    if not params["hidden"]
                }

                self.assertEqual(adapter.mode_options, (mode,))
                self.assertEqual(contract["modes"], [mode])
                self.assertEqual(contract["actions"], {expected["action"]: [mode]})
                self.assertEqual(visible_fields, expected["visible_fields"])
                self.assertEqual(adapter.default_repo, SD15_BASE_REPO)
                self.assertEqual(adapter.model_filter_classes, expected["artifact_classes"])
                self.assertEqual(adapter.load_pipeline_class, pipeline_class)
                self.assertEqual(adapter.allowed_runtime_classes, (pipeline_class,))
                self.assertTrue(adapter.safe_serialization_required)
                self.assertEqual(adapter.max_inference_steps, 100)
                self.assertEqual(
                    (adapter.min_output_side, adapter.max_output_side, adapter.output_side_step),
                    (16, 2048, 16),
                )
                self.assertEqual(adapter.max_output_pixels, 2048 * 2048)
                self.assertEqual(adapter.max_reference_images, 1)
                self.assertEqual(adapter.max_reference_pixels, 16 * 1024 * 1024)
                self.assertEqual(
                    resolve_image_model_selection(adapter, None),
                    {"source": "hub", "value": SD15_BASE_REPO},
                )
                self.assertEqual(
                    resolve_image_pipeline_revision(
                        {"source": "hub", "value": SD15_BASE_REPO},
                        "",
                    ),
                    revision,
                )
                self.assertEqual(
                    declared[pipeline_class],
                    ("image", SD15_BASE_REPO, (mode,)),
                )

    def test_generic_actions_pass_only_the_reviewed_pag_arguments(self):
        image = Image.new("RGB", (32, 48), "black")
        mask = Image.new("L", (32, 48), "white")
        received = {}

        class StableDiffusionPAGImg2ImgPipeline:
            _execution_device = "cpu"

            def __call__(
                self,
                prompt=None,
                image=None,
                strength=0.8,
                num_inference_steps=50,
                guidance_scale=7.5,
                negative_prompt=None,
                generator=None,
                output_type="pil",
                return_dict=True,
                pag_scale=3.0,
                pag_adaptive_scale=0.0,
            ):
                received["edit"] = {
                    "prompt": prompt,
                    "image": image,
                    "strength": strength,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "negative_prompt": negative_prompt,
                    "seed": generator.initial_seed(),
                    "output_type": output_type,
                    "return_dict": return_dict,
                    "pag_scale": pag_scale,
                    "pag_adaptive_scale": pag_adaptive_scale,
                }
                return SimpleNamespace(images=[Image.new("RGB", (32, 48), "white")])

        class StableDiffusionPAGInpaintPipeline:
            _execution_device = "cpu"

            def __call__(
                self,
                prompt=None,
                image=None,
                mask_image=None,
                height=None,
                width=None,
                padding_mask_crop=None,
                strength=0.9999,
                num_inference_steps=50,
                guidance_scale=7.5,
                negative_prompt=None,
                generator=None,
                output_type="pil",
                return_dict=True,
                pag_scale=3.0,
                pag_adaptive_scale=0.0,
            ):
                received["inpaint"] = {
                    "prompt": prompt,
                    "image": image,
                    "mask_image": mask_image,
                    "height": height,
                    "width": width,
                    "padding_mask_crop": padding_mask_crop,
                    "strength": strength,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "negative_prompt": negative_prompt,
                    "seed": generator.initial_seed(),
                    "output_type": output_type,
                    "return_dict": return_dict,
                    "pag_scale": pag_scale,
                    "pag_adaptive_scale": pag_adaptive_scale,
                }
                return SimpleNamespace(images=[Image.new("RGB", (32, 48), "white")])

        common = {
            "prompt": "restore the reviewed fixture",
            "negative_prompt": "artifact",
            "width": 32,
            "height": 48,
            "num_inference_steps": 2,
            "guidance_scale": 4.0,
            "strength": 0.65,
            "pag_scale": 2.5,
            "pag_adaptive_scale": 0.25,
            "seed": 19,
            "output_type": "pil",
        }
        with patch("modules.DiffusersImage.main.add_progress_callback"):
            Edit().execute(
                pipeline=_tagged_pipeline(
                    StableDiffusionPAGImg2ImgPipeline(),
                    "StableDiffusionPAGImg2ImgPipeline",
                    "edit_image",
                ),
                image=image,
                **common,
            )
            Inpaint().execute(
                pipeline=_tagged_pipeline(
                    StableDiffusionPAGInpaintPipeline(),
                    "StableDiffusionPAGInpaintPipeline",
                    "inpaint",
                ),
                image=image,
                mask_image=mask,
                padding_mask_crop=16,
                **common,
            )

        self.assertEqual(
            received["edit"],
            {
                "prompt": common["prompt"],
                "image": image,
                "strength": 0.65,
                "num_inference_steps": 2,
                "guidance_scale": 4.0,
                "negative_prompt": "artifact",
                "seed": 19,
                "output_type": "pil",
                "return_dict": True,
                "pag_scale": 2.5,
                "pag_adaptive_scale": 0.25,
            },
        )
        self.assertEqual(
            received["inpaint"],
            {
                "prompt": common["prompt"],
                "image": image,
                "mask_image": mask,
                "height": 48,
                "width": 32,
                "padding_mask_crop": 16,
                "strength": 0.65,
                "num_inference_steps": 2,
                "guidance_scale": 4.0,
                "negative_prompt": "artifact",
                "seed": 19,
                "output_type": "pil",
                "return_dict": True,
                "pag_scale": 2.5,
                "pag_adaptive_scale": 0.25,
            },
        )

    def test_loader_uses_the_shared_immutable_safetensors_artifact_without_downloads(self):
        loaded = {}

        def fake_pipeline_class(name):
            def from_pretrained(_class, repository, **kwargs):
                loaded[name] = (repository, kwargs)
                return _class()

            return type(name, (), {"from_pretrained": classmethod(from_pretrained)})

        node = LoadPipeline("sd15-pag-edit-loader")
        node.progress = lambda *_args, **_kwargs: None
        node.mm_add = lambda *_args, **_kwargs: None
        classes = {name: fake_pipeline_class(name) for name in PAG_EDIT_PIPELINES}

        with (
            patch(
                "modules.DiffusersImage.main.pipeline_class_from_name",
                side_effect=lambda name: classes[name],
            ),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
        ):
            for pipeline_class, expected in PAG_EDIT_PIPELINES.items():
                with self.subTest(pipeline=pipeline_class):
                    result = node.execute(
                        model_id=None,
                        pipeline_class=pipeline_class,
                        mode=expected["mode"],
                        dtype="float32",
                        device="cpu",
                        auto_offload=False,
                        offload_mode="none",
                    )
                    self.assertEqual(result["resolved_artifact"], SD15_BASE_REPO)
                    self.assertEqual(
                        result["pipeline"]._modiff_image_pipeline_class,
                        pipeline_class,
                    )

        for pipeline_class in PAG_EDIT_PIPELINES:
            repository, kwargs = loaded[pipeline_class]
            self.assertEqual(repository, SD15_BASE_REPO)
            self.assertEqual(kwargs["revision"], catalog_revision(SD15_BASE_REPO))
            self.assertTrue(kwargs["use_safetensors"])
            self.assertNotIn("trust_remote_code", kwargs)

    def test_routes_fail_closed_for_outpaint_and_invalid_pag_bounds(self):
        class StableDiffusionPAGImg2ImgPipeline:
            def __call__(self, **_kwargs):
                raise AssertionError("inference must not run")

        pipeline = _tagged_pipeline(
            StableDiffusionPAGImg2ImgPipeline(),
            "StableDiffusionPAGImg2ImgPipeline",
            "edit_image",
        )
        with self.assertRaisesRegex(ValueError, "pag_scale"):
            Edit().execute(
                pipeline=pipeline,
                image=Image.new("RGB", (16, 16)),
                pag_scale=20.01,
            )
        with self.assertRaisesRegex(ValueError, "does not support outpaint"):
            LoadPipeline._validate_mode("StableDiffusionPAGInpaintPipeline", "outpaint")


if __name__ == "__main__":
    unittest.main()
