import unittest

from modiff.studio_execution_specs import (
    SD15_SHARED_DIFFUSERS_FILES,
    studio_capability_definitions,
)
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS


class StableDiffusion15DownloadSelectionTests(unittest.TestCase):
    def test_shared_selection_covers_image_and_animatediff_safe_variants(self):
        selected = set(SD15_SHARED_DIFFUSERS_FILES)
        capabilities = studio_capability_definitions()

        self.assertEqual(len(selected), 21)
        for model_type in (
            "StableDiffusionPipeline",
            "StableDiffusionPAGPipeline",
            "AnimateDiffPipeline",
            "AnimateLCMPipeline",
        ):
            with self.subTest(model_type=model_type):
                self.assertEqual(
                    capabilities[model_type]["downloadFiles"],
                    SD15_SHARED_DIFFUSERS_FILES,
                )

        for component, filename in (
            ("safety_checker", "model"),
            ("text_encoder", "model"),
            ("unet", "diffusion_pytorch_model"),
            ("vae", "diffusion_pytorch_model"),
        ):
            with self.subTest(component=component):
                self.assertIn(f"{component}/{filename}.safetensors", selected)
                self.assertIn(f"{component}/{filename}.fp16.safetensors", selected)

        for model_type in (
            "StableDiffusionPipeline",
            "StableDiffusionControlNetPipeline",
            "StableDiffusionImg2ImgPipeline",
            "StableDiffusionInpaintPipeline",
            "StableDiffusionPAGPipeline",
        ):
            with self.subTest(adapter=model_type):
                self.assertTrue(
                    IMAGE_PIPELINE_ADAPTERS[model_type].safe_serialization_required
                )

        self.assertNotIn("unet/diffusion_pytorch_model.non_ema.safetensors", selected)
        self.assertFalse(any("v1-5-pruned" in path for path in selected))
        self.assertFalse(any(path.endswith(".yaml") for path in selected))
        self.assertFalse(
            any(path.endswith((".bin", ".ckpt", ".pt", ".pth")) for path in selected)
        )


if __name__ == "__main__":
    unittest.main()
