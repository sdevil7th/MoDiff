import unittest

from modiff.server import STUDIO_MODEL_CAPABILITIES
from modiff.studio_execution_specs import Z_IMAGE_DIFFUSERS_FILES
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS


class ZImageDownloadSelectionTests(unittest.TestCase):
    def test_z_image_selection_is_component_only_and_safe(self):
        selected = set(Z_IMAGE_DIFFUSERS_FILES)
        capability = STUDIO_MODEL_CAPABILITIES["ZImageModularPipeline"]

        self.assertEqual(capability["downloadFiles"], Z_IMAGE_DIFFUSERS_FILES)
        self.assertEqual(len(selected), 21)
        self.assertIn("text_encoder/model.safetensors.index.json", selected)
        self.assertIn(
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            selected,
        )
        self.assertIn("vae/diffusion_pytorch_model.safetensors", selected)
        self.assertFalse(any(path.startswith("assets/") for path in selected))
        self.assertFalse(
            any(path.endswith((".bin", ".ckpt", ".pt", ".pth")) for path in selected)
        )

        for model_type in (
            "ZImagePipeline",
            "ZImageImg2ImgPipeline",
            "ZImageInpaintPipeline",
        ):
            with self.subTest(model_type=model_type):
                self.assertTrue(
                    IMAGE_PIPELINE_ADAPTERS[model_type].safe_serialization_required
                )


if __name__ == "__main__":
    unittest.main()
