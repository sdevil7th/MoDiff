import unittest

from modiff.studio_execution_specs import (
    CONSISTENCY_IMAGENET64_DIFFUSERS_FILES,
    DDPM_CIFAR10_DIFFUSERS_FILES,
    studio_capability_definitions,
)
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS


class UnconditionalDownloadSelectionTests(unittest.TestCase):
    def test_ddpm_and_ddim_share_one_safe_code_free_selection(self):
        selected = set(DDPM_CIFAR10_DIFFUSERS_FILES)
        capabilities = studio_capability_definitions()

        self.assertEqual(len(selected), 6)
        for model_type in ("DDPMPipeline", "DDIMPipeline"):
            with self.subTest(model_type=model_type):
                self.assertEqual(
                    capabilities[model_type]["downloadFiles"],
                    DDPM_CIFAR10_DIFFUSERS_FILES,
                )
                self.assertTrue(
                    IMAGE_PIPELINE_ADAPTERS[model_type].safe_serialization_required
                )
        self.assertIn("diffusion_pytorch_model.safetensors", selected)
        self.assertNotIn("diffusion_pytorch_model.bin", selected)
        self.assertNotIn("modeling_ddpm.py", selected)
        self.assertFalse(any(path.startswith("images/") for path in selected))

    def test_consistency_model_selection_excludes_legacy_duplicate(self):
        selected = set(CONSISTENCY_IMAGENET64_DIFFUSERS_FILES)
        capability = studio_capability_definitions()["ConsistencyModelPipeline"]
        adapter = IMAGE_PIPELINE_ADAPTERS["ConsistencyModelPipeline"]

        self.assertEqual(
            capability["downloadFiles"],
            CONSISTENCY_IMAGENET64_DIFFUSERS_FILES,
        )
        self.assertEqual(len(selected), 6)
        self.assertTrue(adapter.safe_serialization_required)
        self.assertIn("unet/diffusion_pytorch_model.safetensors", selected)
        self.assertNotIn("unet/diffusion_pytorch_model.bin", selected)
        self.assertFalse(
            any(path.endswith((".bin", ".ckpt", ".pt", ".pth")) for path in selected)
        )


if __name__ == "__main__":
    unittest.main()
