import unittest

from modiff.studio_execution_specs import (
    SHAP_E_SAFE_COMPONENT_FILES,
    studio_capability_definitions,
)


class ThreeDDownloadSelectionTests(unittest.TestCase):
    def test_shap_e_selection_matches_explicit_safe_component_assembly(self):
        selected = set(SHAP_E_SAFE_COMPONENT_FILES)
        capability = studio_capability_definitions()["ShapEPipeline"]

        self.assertEqual(capability["downloadFiles"], SHAP_E_SAFE_COMPONENT_FILES)
        self.assertEqual(len(selected), 14)
        self.assertIn("prior/diffusion_pytorch_model.fp16.safetensors", selected)
        self.assertIn("renderer/diffusion_pytorch_model.fp16.safetensors", selected)
        self.assertIn("text_encoder/model.fp16.safetensors", selected)
        self.assertNotIn("shap_e_renderer/config.json", selected)
        self.assertNotIn("shap_e_renderer/diffusion_pytorch_model.bin", selected)
        self.assertFalse(
            any(path.endswith((".bin", ".ckpt", ".pt", ".pth")) for path in selected)
        )


if __name__ == "__main__":
    unittest.main()
