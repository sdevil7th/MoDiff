import unittest

from modiff.studio_execution_specs import (
    SANA_600M_FP16_DIFFUSERS_FILES,
    SANA_SPRINT_DIFFUSERS_FILES,
    studio_capability_definitions,
)
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS


class SanaDownloadSelectionTests(unittest.TestCase):
    def test_sana_selection_matches_reviewed_fp16_loader_variant(self):
        selected = set(SANA_600M_FP16_DIFFUSERS_FILES)
        capability = studio_capability_definitions()["SanaPipeline"]
        adapter = IMAGE_PIPELINE_ADAPTERS["SanaPipeline"]

        self.assertEqual(capability["downloadFiles"], SANA_600M_FP16_DIFFUSERS_FILES)
        self.assertEqual(len(selected), 17)
        self.assertTrue(adapter.safe_serialization_required)
        self.assertEqual(adapter.weight_variant, "fp16")
        self.assertIn("text_encoder/model.safetensors.index.fp16.json", selected)
        self.assertIn(
            "text_encoder/model.fp16-00001-of-00002.safetensors",
            selected,
        )
        self.assertIn(
            "text_encoder/model.fp16-00002-of-00002.safetensors",
            selected,
        )
        self.assertIn(
            "transformer/diffusion_pytorch_model.fp16.safetensors",
            selected,
        )
        self.assertIn("vae/diffusion_pytorch_model.fp16.safetensors", selected)
        self.assertNotIn("text_encoder/model.safetensors.index.json", selected)
        self.assertNotIn("transformer/diffusion_pytorch_model.safetensors", selected)
        self.assertNotIn("vae/diffusion_pytorch_model.safetensors", selected)
        self.assertFalse(
            any(path.endswith((".bin", ".ckpt", ".pt", ".pth")) for path in selected)
        )

    def test_sana_sprint_selection_matches_the_safe_bfloat16_snapshot(self):
        selected = set(SANA_SPRINT_DIFFUSERS_FILES)
        capability = studio_capability_definitions()["SanaSprintPipeline"]

        self.assertEqual(capability["downloadFiles"], SANA_SPRINT_DIFFUSERS_FILES)
        self.assertEqual(len(selected), 17)
        for pipeline_class in ("SanaSprintPipeline", "SanaSprintImg2ImgPipeline"):
            with self.subTest(pipeline_class=pipeline_class):
                self.assertTrue(
                    IMAGE_PIPELINE_ADAPTERS[pipeline_class].safe_serialization_required
                )
        self.assertIn("text_encoder/model.safetensors.index.json", selected)
        self.assertIn(
            "transformer/diffusion_pytorch_model.safetensors",
            selected,
        )
        self.assertIn("vae/diffusion_pytorch_model.safetensors", selected)
        self.assertFalse(
            any(path.endswith((".bin", ".ckpt", ".pt", ".pth")) for path in selected)
        )


if __name__ == "__main__":
    unittest.main()
