import unittest

from modiff.studio_execution_specs import (
    COGVIEW4_6B_DIFFUSERS_FILES,
    ERNIE_IMAGE_TURBO_DIFFUSERS_FILES,
    GLM_IMAGE_DIFFUSERS_FILES,
    studio_capability_definitions,
)
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS


class LargeImageDownloadSelectionTests(unittest.TestCase):
    def test_large_image_selections_are_complete_and_safe(self):
        capabilities = studio_capability_definitions()
        cases = {
            "CogView4Pipeline": (COGVIEW4_6B_DIFFUSERS_FILES, 21),
            "ErnieImagePipeline": (ERNIE_IMAGE_TURBO_DIFFUSERS_FILES, 24),
            "GlmImagePipeline": (GLM_IMAGE_DIFFUSERS_FILES, 27),
        }

        for model_type, (files, count) in cases.items():
            with self.subTest(model_type=model_type):
                selected = set(files)
                self.assertEqual(capabilities[model_type]["downloadFiles"], files)
                self.assertEqual(len(selected), count)
                self.assertTrue(
                    IMAGE_PIPELINE_ADAPTERS[model_type].safe_serialization_required
                )
                self.assertIn("model_index.json", selected)
                self.assertIn("scheduler/scheduler_config.json", selected)
                self.assertIn("transformer/config.json", selected)
                self.assertIn("vae/diffusion_pytorch_model.safetensors", selected)
                self.assertFalse(
                    any(
                        path.endswith((".bin", ".ckpt", ".pt", ".pth"))
                        for path in selected
                    )
                )

        self.assertIn(
            "text_encoder/model.safetensors.index.json",
            COGVIEW4_6B_DIFFUSERS_FILES,
        )
        self.assertIn("pe/model.safetensors", ERNIE_IMAGE_TURBO_DIFFUSERS_FILES)
        self.assertIn(
            "vision_language_encoder/model.safetensors.index.json",
            GLM_IMAGE_DIFFUSERS_FILES,
        )


if __name__ == "__main__":
    unittest.main()
