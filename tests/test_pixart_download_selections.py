import unittest

from modiff.studio_execution_specs import (
    PIXART_SIGMA_DIFFUSERS_FILES,
    studio_capability_definitions,
)
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS


class PixArtDownloadSelectionTests(unittest.TestCase):
    def test_pixart_sigma_selection_is_component_only_and_safe(self):
        selected = set(PIXART_SIGMA_DIFFUSERS_FILES)
        capability = studio_capability_definitions()["PixArtSigmaPipeline"]
        adapter = IMAGE_PIPELINE_ADAPTERS["PixArtSigmaPipeline"]

        self.assertEqual(capability["downloadFiles"], PIXART_SIGMA_DIFFUSERS_FILES)
        self.assertEqual(len(selected), 15)
        self.assertTrue(adapter.safe_serialization_required)
        self.assertIn("text_encoder/model.safetensors.index.json", selected)
        self.assertIn(
            "transformer/diffusion_pytorch_model.safetensors",
            selected,
        )
        self.assertIn("vae/diffusion_pytorch_model.safetensors", selected)
        self.assertFalse(any(path.startswith("asset/") for path in selected))
        self.assertFalse(
            any(path.endswith((".bin", ".ckpt", ".pt", ".pth")) for path in selected)
        )


if __name__ == "__main__":
    unittest.main()
