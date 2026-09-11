import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from modiff.model_artifact_catalog import catalog_revision
from modiff.studio_execution_specs import (
    LCM_DREAMSHAPER_DIFFUSERS_FILES,
    studio_capability_definitions,
)
from modules.DiffusersImage.main import Edit, IMAGE_PIPELINE_ADAPTERS, LoadPipeline


class LatentImageDownloadSelectionTests(unittest.TestCase):
    def test_lcm_selection_is_safe_code_free_diffusers_snapshot(self):
        selected = set(LCM_DREAMSHAPER_DIFFUSERS_FILES)
        capability = studio_capability_definitions()["LatentConsistencyModelPipeline"]
        adapter = IMAGE_PIPELINE_ADAPTERS["LatentConsistencyModelPipeline"]
        img2img_adapter = IMAGE_PIPELINE_ADAPTERS["LatentConsistencyModelImg2ImgPipeline"]

        self.assertEqual(
            capability["downloadFiles"],
            LCM_DREAMSHAPER_DIFFUSERS_FILES,
        )
        self.assertEqual(len(selected), 17)
        self.assertTrue(adapter.safe_serialization_required)
        self.assertEqual(img2img_adapter.default_repo, adapter.default_repo)
        self.assertEqual(
            img2img_adapter.artifact_pipeline_classes,
            ("LatentConsistencyModelPipeline", "LatentConsistencyModelImg2ImgPipeline"),
        )
        self.assertTrue(img2img_adapter.safe_serialization_required)
        self.assertEqual(img2img_adapter.max_inference_steps, 50)
        self.assertEqual(img2img_adapter.max_reference_pixels, 2048 * 2048)
        self.assertIn("safety_checker/model.safetensors", selected)
        self.assertIn("text_encoder/model.safetensors", selected)
        self.assertIn("unet/diffusion_pytorch_model.safetensors", selected)
        self.assertIn("vae/diffusion_pytorch_model.safetensors", selected)
        self.assertNotIn("LCM_Dreamshaper_v7_4k.safetensors", selected)
        self.assertFalse(any(path.endswith(".onnx") for path in selected))
        self.assertFalse(any(path.endswith(".py") for path in selected))
        self.assertFalse(any(path.startswith("images/") for path in selected))

    def test_lcm_img2img_loads_and_executes_only_the_reviewed_generic_surface(self):
        loaded = {}
        received = {}

        class LatentConsistencyModelImg2ImgPipeline:
            _execution_device = "cpu"

            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                loaded.update({"repo": repo, "kwargs": kwargs})
                return cls()

            def __call__(
                self,
                prompt=None,
                image=None,
                num_inference_steps=4,
                strength=0.8,
                original_inference_steps=None,
                timesteps=None,
                guidance_scale=8.5,
                num_images_per_prompt=1,
                generator=None,
                latents=None,
                prompt_embeds=None,
                ip_adapter_image=None,
                ip_adapter_image_embeds=None,
                output_type="pil",
                return_dict=True,
                cross_attention_kwargs=None,
                clip_skip=None,
                callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=None,
            ):
                received.update(
                    {
                        "prompt": prompt,
                        "image": image,
                        "num_inference_steps": num_inference_steps,
                        "strength": strength,
                        "guidance_scale": guidance_scale,
                        "generator": generator,
                        "output_type": output_type,
                        "return_dict": return_dict,
                    }
                )
                return SimpleNamespace(images=[Image.new("RGB", image.size, "white")])

        loader = LoadPipeline("lcm-img2img-load-probe")
        loader.progress = lambda *args, **kwargs: None
        loader.mm_add = lambda *args, **kwargs: None
        with (
            patch(
                "modules.DiffusersImage.main.pipeline_class_from_name",
                return_value=LatentConsistencyModelImg2ImgPipeline,
            ) as resolve_pipeline,
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
            patch(
                "modules.DiffusersRuntime.main.apply_execution_recipe_to_pipeline",
                return_value={},
            ),
        ):
            result = loader.execute(
                model_id="SimianLuo/LCM_Dreamshaper_v7",
                pipeline_class="LatentConsistencyModelImg2ImgPipeline",
                mode="edit_image",
                revision=catalog_revision("SimianLuo/LCM_Dreamshaper_v7"),
                dtype="float32",
                auto_offload=False,
                offload_mode="none",
            )

        resolve_pipeline.assert_called_once_with("LatentConsistencyModelImg2ImgPipeline")
        self.assertEqual(loaded["repo"], "SimianLuo/LCM_Dreamshaper_v7")
        self.assertEqual(
            loaded["kwargs"]["revision"],
            "a85df6a8bd976cdd08b4fd8f3b73f229c9e54df5",
        )
        self.assertTrue(loaded["kwargs"]["use_safetensors"])
        self.assertNotIn("trust_remote_code", loaded["kwargs"])
        self.assertNotIn("variant", loaded["kwargs"])

        source = Image.new("RGB", (64, 64), "black")
        with patch("modules.DiffusersImage.main.add_progress_callback"):
            output = Edit("lcm-img2img-edit-probe").execute(
                pipeline=result["pipeline"],
                prompt="turn the ridge into a watercolor",
                negative_prompt="must remain hidden",
                image=source,
                width=1024,
                height=1024,
                num_inference_steps=4,
                guidance_scale=8.0,
                strength=0.75,
                output_type="pil",
            )

        self.assertEqual(
            set(received),
            {
                "prompt",
                "image",
                "num_inference_steps",
                "strength",
                "guidance_scale",
                "generator",
                "output_type",
                "return_dict",
            },
        )
        self.assertIs(received["image"], source)
        self.assertEqual(received["strength"], 0.75)
        self.assertEqual(received["guidance_scale"], 8.0)
        self.assertEqual((output["width_out"], output["height_out"]), source.size)

        with self.assertRaisesRegex(ValueError, "between 1 and 50"):
            Edit("lcm-img2img-step-bound").execute(
                pipeline=result["pipeline"],
                prompt="bounded",
                image=source,
                num_inference_steps=51,
            )
        with self.assertRaisesRegex(ValueError, "4194304-pixel cumulative input limit"):
            Edit("lcm-img2img-pixel-bound").execute(
                pipeline=result["pipeline"],
                prompt="bounded",
                image=Image.new("L", (2049, 2049)),
                num_inference_steps=4,
            )


if __name__ == "__main__":
    unittest.main()
