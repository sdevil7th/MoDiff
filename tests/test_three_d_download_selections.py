import unittest

from modiff.studio_execution_specs import (
    SHAP_E_IMG2IMG_SAFE_COMPONENT_FILES,
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

    def test_shap_e_image_selection_is_complete_safe_and_exact(self):
        selected = set(SHAP_E_IMG2IMG_SAFE_COMPONENT_FILES)
        capability = studio_capability_definitions()["ShapEImg2ImgPipeline"]

        self.assertEqual(capability["downloadFiles"], SHAP_E_IMG2IMG_SAFE_COMPONENT_FILES)
        self.assertEqual(len(selected), 11)
        self.assertIn("image_encoder/model.fp16.safetensors", selected)
        self.assertIn("prior/diffusion_pytorch_model.fp16.safetensors", selected)
        self.assertIn("renderer/diffusion_pytorch_model.fp16.safetensors", selected)
        self.assertNotIn("shap_e_renderer/diffusion_pytorch_model.bin", selected)
        self.assertFalse(any(path.endswith((".bin", ".ckpt", ".pt", ".pth")) for path in selected))
        reviewed_weights = {
            "image_encoder/model.fp16.safetensors": (
                606_409_288,
                "4523175837d6342fd2d3ca38ce84912dbc722b556747c883b2842157e9a0f035",
            ),
            "prior/diffusion_pytorch_model.fp16.safetensors": (
                631_964_448,
                "a6d92d12597b298dfb84b32ccfc74a94d2f25d7600f330b39722bd0f0835862e",
            ),
            "renderer/diffusion_pytorch_model.fp16.safetensors": (
                452_597_858,
                "394d52e159447b2bfb944d3bfa4f438180b72ba40644dc4c730282e7a9b62c65",
            ),
        }
        self.assertEqual(sum(size for size, _sha in reviewed_weights.values()), 1_690_971_594)
        self.assertEqual(
            set(reviewed_weights),
            {path for path in selected if path.endswith(".safetensors")},
        )


if __name__ == "__main__":
    unittest.main()
