import unittest

from modiff.studio_execution_specs import (
    SDXL_BASE_FP16_DIFFUSERS_FILES,
    SDXL_INSTRUCT_PIX2PIX_DIFFUSERS_FILES,
    SDXL_TURBO_FP16_DIFFUSERS_FILES,
    studio_capability_definitions,
)
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS


class SDXLDownloadSelectionTests(unittest.TestCase):
    def test_base_selection_matches_every_base_runtime_adapter(self):
        selected = set(SDXL_BASE_FP16_DIFFUSERS_FILES)
        capabilities = studio_capability_definitions()

        self.assertEqual(len(selected), 21)
        for model_type in (
            "StableDiffusionXLPipeline",
            "StableDiffusionXLControlNetPipeline",
            "StableDiffusionXLAdapterPipeline",
            "StableDiffusionXLPAGPipeline",
        ):
            with self.subTest(capability=model_type):
                self.assertEqual(
                    capabilities[model_type]["downloadFiles"],
                    SDXL_BASE_FP16_DIFFUSERS_FILES,
                )
        for adapter_name in (
            "StableDiffusionXLPipeline",
            "StableDiffusionXLImg2ImgPipeline",
            "StableDiffusionXLInpaintPipeline",
            "StableDiffusionXLControlNetPipeline",
            "StableDiffusionXLAdapterPipeline",
            "StableDiffusionXLPAGPipeline",
            "StableDiffusionXLPAGImg2ImgPipeline",
            "StableDiffusionXLPAGInpaintPipeline",
        ):
            with self.subTest(adapter=adapter_name):
                adapter = IMAGE_PIPELINE_ADAPTERS[adapter_name]
                self.assertTrue(adapter.safe_serialization_required)
                self.assertEqual(adapter.weight_variant, "fp16")

        self.assertEqual(
            capabilities["StableDiffusionXLPipeline"]["defaultDtype"],
            "float16",
        )
        self.assertIn("unet/diffusion_pytorch_model.fp16.safetensors", selected)
        self.assertIn("vae/diffusion_pytorch_model.fp16.safetensors", selected)
        self.assertNotIn("unet/diffusion_pytorch_model.safetensors", selected)
        self.assertNotIn("vae_1_0/diffusion_pytorch_model.fp16.safetensors", selected)
        self.assertNotIn("sd_xl_base_1.0.safetensors", selected)
        self.assertFalse(
            any(path.endswith((".bin", ".ckpt", ".pt", ".pth")) for path in selected)
        )

    def test_turbo_selection_matches_the_fp16_loader_variant(self):
        selected = set(SDXL_TURBO_FP16_DIFFUSERS_FILES)
        capability = studio_capability_definitions()["StableDiffusionXLTurboPipeline"]
        adapter = IMAGE_PIPELINE_ADAPTERS["StableDiffusionXLTurboPipeline"]

        self.assertEqual(
            capability["downloadFiles"],
            SDXL_TURBO_FP16_DIFFUSERS_FILES,
        )
        self.assertEqual(len(selected), 21)
        self.assertTrue(adapter.safe_serialization_required)
        self.assertEqual(adapter.weight_variant, "fp16")
        self.assertIn("text_encoder_2/model.fp16.safetensors", selected)
        self.assertNotIn("text_encoder_2/model.safetensors", selected)
        self.assertNotIn("sd_xl_turbo_1.0_fp16.safetensors", selected)

    def test_instruct_pix2pix_selection_is_component_only_and_safe(self):
        selected = set(SDXL_INSTRUCT_PIX2PIX_DIFFUSERS_FILES)
        capability = studio_capability_definitions()[
            "StableDiffusionXLInstructPix2PixPipeline"
        ]
        adapter = IMAGE_PIPELINE_ADAPTERS[
            "StableDiffusionXLInstructPix2PixPipeline"
        ]

        self.assertEqual(
            capability["downloadFiles"],
            SDXL_INSTRUCT_PIX2PIX_DIFFUSERS_FILES,
        )
        self.assertEqual(len(selected), 20)
        self.assertTrue(adapter.safe_serialization_required)
        self.assertIsNone(adapter.weight_variant)
        self.assertIn("unet/diffusion_pytorch_model.safetensors", selected)
        self.assertIn("vae/diffusion_pytorch_model.safetensors", selected)
        self.assertFalse(
            any(path.startswith("validation_images/") for path in selected)
        )
        self.assertFalse(
            any(path.endswith((".bin", ".ckpt", ".pt", ".pth")) for path in selected)
        )


if __name__ == "__main__":
    unittest.main()
