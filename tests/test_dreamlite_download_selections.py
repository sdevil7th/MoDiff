import unittest

from modiff.studio_execution_specs import (
    DREAMLITE_DIFFUSERS_FILES,
    studio_capability_definitions,
)
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS


class DreamLiteDownloadSelectionTests(unittest.TestCase):
    def test_dreamlite_selections_are_complete_and_safe(self):
        selected = set(DREAMLITE_DIFFUSERS_FILES)
        capabilities = studio_capability_definitions()

        self.assertEqual(len(selected), 27)
        self.assertIn("processor/preprocessor_config.json", selected)
        self.assertIn("processor/tokenizer.json", selected)
        self.assertIn("tokenizer/tokenizer.json", selected)
        self.assertIn("text_encoder/model.safetensors", selected)
        self.assertIn("unet/diffusion_pytorch_model.safetensors", selected)
        self.assertIn("vae/diffusion_pytorch_model.safetensors", selected)
        self.assertFalse(
            any(path.endswith((".bin", ".ckpt", ".pt", ".pth")) for path in selected)
        )

        for model_type in ("DreamLitePipeline", "DreamLiteMobilePipeline"):
            with self.subTest(model_type=model_type):
                self.assertEqual(
                    capabilities[model_type]["downloadFiles"],
                    DREAMLITE_DIFFUSERS_FILES,
                )
                self.assertTrue(
                    IMAGE_PIPELINE_ADAPTERS[model_type].safe_serialization_required
                )


if __name__ == "__main__":
    unittest.main()
