import unittest

from modiff.studio_execution_specs import (
    JOYIMAGE_EDIT_DIFFUSERS_FILES,
    JOYIMAGE_EDIT_PLUS_DIFFUSERS_FILES,
    studio_capability_definitions,
)
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS


class JoyImageDownloadSelectionTests(unittest.TestCase):
    def test_joyimage_selections_are_component_only_and_safe(self):
        capabilities = studio_capability_definitions()
        cases = {
            "JoyImageEditPipeline": (JOYIMAGE_EDIT_DIFFUSERS_FILES, 38),
            "JoyImageEditPlusPipeline": (JOYIMAGE_EDIT_PLUS_DIFFUSERS_FILES, 29),
        }
        for model_type, (files, count) in cases.items():
            with self.subTest(model_type=model_type):
                selected = set(files)
                self.assertEqual(capabilities[model_type]["downloadFiles"], files)
                self.assertEqual(len(selected), count)
                self.assertTrue(
                    IMAGE_PIPELINE_ADAPTERS[model_type].safe_serialization_required
                )
                self.assertIn("text_encoder/model.safetensors.index.json", selected)
                self.assertIn(
                    "transformer/diffusion_pytorch_model.safetensors.index.json",
                    selected,
                )
                self.assertIn("vae/diffusion_pytorch_model.safetensors", selected)
                self.assertFalse(
                    any(
                        path.startswith(("examples/", "test_images/"))
                        for path in selected
                    )
                )
                self.assertFalse(any(path.endswith(".py") for path in selected))
                self.assertFalse(
                    any(
                        path.endswith((".bin", ".ckpt", ".pt", ".pth"))
                        for path in selected
                    )
                )


if __name__ == "__main__":
    unittest.main()
