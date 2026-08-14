import unittest

from modiff.studio_execution_specs import (
    LCM_DREAMSHAPER_DIFFUSERS_FILES,
    studio_capability_definitions,
)
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS


class LatentImageDownloadSelectionTests(unittest.TestCase):
    def test_lcm_selection_is_safe_code_free_diffusers_snapshot(self):
        selected = set(LCM_DREAMSHAPER_DIFFUSERS_FILES)
        capability = studio_capability_definitions()["LatentConsistencyModelPipeline"]
        adapter = IMAGE_PIPELINE_ADAPTERS["LatentConsistencyModelPipeline"]

        self.assertEqual(
            capability["downloadFiles"],
            LCM_DREAMSHAPER_DIFFUSERS_FILES,
        )
        self.assertEqual(len(selected), 17)
        self.assertTrue(adapter.safe_serialization_required)
        self.assertIn("safety_checker/model.safetensors", selected)
        self.assertIn("text_encoder/model.safetensors", selected)
        self.assertIn("unet/diffusion_pytorch_model.safetensors", selected)
        self.assertIn("vae/diffusion_pytorch_model.safetensors", selected)
        self.assertNotIn("LCM_Dreamshaper_v7_4k.safetensors", selected)
        self.assertFalse(any(path.endswith(".onnx") for path in selected))
        self.assertFalse(any(path.endswith(".py") for path in selected))
        self.assertFalse(any(path.startswith("images/") for path in selected))


if __name__ == "__main__":
    unittest.main()
