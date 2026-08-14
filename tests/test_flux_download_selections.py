import unittest

from modiff.studio_execution_specs import (
    FLUX2_KLEIN_DIFFUSERS_FILES,
    studio_capability_definitions,
)
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS


class FluxDownloadSelectionTests(unittest.TestCase):
    def test_flux2_klein_selection_is_component_only_and_safe(self):
        selected = set(FLUX2_KLEIN_DIFFUSERS_FILES)
        capability = studio_capability_definitions()["Flux2KleinPipeline"]

        self.assertEqual(capability["downloadFiles"], FLUX2_KLEIN_DIFFUSERS_FILES)
        self.assertEqual(len(selected), 21)
        for model_type in ("Flux2KleinPipeline", "Flux2KleinInpaintPipeline"):
            with self.subTest(model_type=model_type):
                self.assertTrue(
                    IMAGE_PIPELINE_ADAPTERS[model_type].safe_serialization_required
                )
        self.assertIn("text_encoder/model.safetensors.index.json", selected)
        self.assertIn(
            "transformer/diffusion_pytorch_model.safetensors",
            selected,
        )
        self.assertIn("vae/diffusion_pytorch_model.safetensors", selected)
        self.assertNotIn("flux-2-klein-4b.safetensors", selected)
        self.assertFalse(any(path.endswith(".jpg") for path in selected))
        self.assertFalse(
            any(path.endswith((".bin", ".ckpt", ".pt", ".pth")) for path in selected)
        )


if __name__ == "__main__":
    unittest.main()
