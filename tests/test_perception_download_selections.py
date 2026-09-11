import unittest

from modiff.studio_execution_specs import (
    MARIGOLD_DEPTH_DIFFUSERS_FILES,
    studio_capability_definitions,
)
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS


class PerceptionDownloadSelectionTests(unittest.TestCase):
    def test_marigold_selection_matches_safe_float32_loader_surface(self):
        selected = set(MARIGOLD_DEPTH_DIFFUSERS_FILES)
        capability = studio_capability_definitions()["MarigoldDepthPipeline"]
        adapter = IMAGE_PIPELINE_ADAPTERS["MarigoldDepthPipeline"]

        self.assertEqual(capability["downloadFiles"], MARIGOLD_DEPTH_DIFFUSERS_FILES)
        self.assertEqual(len(selected), 14)
        self.assertTrue(adapter.safe_serialization_required)
        self.assertIsNone(adapter.weight_variant)
        self.assertIn("text_encoder/model.safetensors", selected)
        self.assertIn("unet/diffusion_pytorch_model.safetensors", selected)
        self.assertIn("vae/diffusion_pytorch_model.safetensors", selected)
        self.assertNotIn("unet/diffusion_pytorch_model.fp16.safetensors", selected)
        self.assertFalse(
            any(path.endswith((".bin", ".ckpt", ".pt", ".pth")) for path in selected)
        )


if __name__ == "__main__":
    unittest.main()
