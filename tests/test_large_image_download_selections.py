import unittest

from modiff.studio_execution_specs import (
    COGVIEW3_PLUS_DIFFUSERS_FILES,
    COGVIEW4_6B_DIFFUSERS_FILES,
    ERNIE_IMAGE_TURBO_DIFFUSERS_FILES,
    GLM_IMAGE_DIFFUSERS_FILES,
    HUNYUAN_DIT_DISTILLED_DIFFUSERS_FILES,
    KANDINSKY3_FP16_DIFFUSERS_FILES,
    studio_capability_definitions,
)
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS


class LargeImageDownloadSelectionTests(unittest.TestCase):
    def test_large_image_selections_are_complete_and_safe(self):
        capabilities = studio_capability_definitions()
        cases = {
            "CogView3PlusPipeline": (COGVIEW3_PLUS_DIFFUSERS_FILES, 20),
            "CogView4Pipeline": (COGVIEW4_6B_DIFFUSERS_FILES, 21),
            "ErnieImagePipeline": (ERNIE_IMAGE_TURBO_DIFFUSERS_FILES, 24),
            "GlmImagePipeline": (GLM_IMAGE_DIFFUSERS_FILES, 27),
        }

        for model_type, (files, count) in cases.items():
            with self.subTest(model_type=model_type):
                selected = set(files)
                self.assertEqual(capabilities[model_type]["downloadFiles"], files)
                self.assertEqual(len(selected), count)
                self.assertTrue(
                    IMAGE_PIPELINE_ADAPTERS[model_type].safe_serialization_required
                )
                self.assertIn("model_index.json", selected)
                self.assertIn("scheduler/scheduler_config.json", selected)
                self.assertIn("transformer/config.json", selected)
                self.assertIn("vae/diffusion_pytorch_model.safetensors", selected)
                self.assertFalse(
                    any(
                        path.endswith((".bin", ".ckpt", ".pt", ".pth"))
                        for path in selected
                    )
                )

        self.assertIn(
            "text_encoder/model.safetensors.index.json",
            COGVIEW3_PLUS_DIFFUSERS_FILES,
        )
        self.assertNotIn("configuration.json", COGVIEW3_PLUS_DIFFUSERS_FILES)
        self.assertNotIn("README_zh.md", COGVIEW3_PLUS_DIFFUSERS_FILES)
        self.assertIn(
            "text_encoder/model.safetensors.index.json",
            COGVIEW4_6B_DIFFUSERS_FILES,
        )
        self.assertIn("pe/model.safetensors", ERNIE_IMAGE_TURBO_DIFFUSERS_FILES)
        self.assertIn(
            "vision_language_encoder/model.safetensors.index.json",
            GLM_IMAGE_DIFFUSERS_FILES,
        )

    def test_kandinsky_and_hunyuan_selections_match_loader_serialization(self):
        capabilities = studio_capability_definitions()
        cases = {
            "Kandinsky3Pipeline": (KANDINSKY3_FP16_DIFFUSERS_FILES, 19, "fp16"),
            "HunyuanDiTPipeline": (HUNYUAN_DIT_DISTILLED_DIFFUSERS_FILES, 20, None),
            "HunyuanDiTControlNetPipeline": (
                HUNYUAN_DIT_DISTILLED_DIFFUSERS_FILES,
                20,
                None,
            ),
        }

        for model_type, (files, count, variant) in cases.items():
            with self.subTest(model_type=model_type):
                selected = set(files)
                adapter = IMAGE_PIPELINE_ADAPTERS[model_type]
                self.assertEqual(capabilities[model_type]["downloadFiles"], files)
                self.assertEqual(len(selected), count)
                self.assertTrue(adapter.safe_serialization_required)
                self.assertEqual(adapter.weight_variant, variant)
                self.assertFalse(
                    any(
                        path.endswith((".bin", ".ckpt", ".pt", ".pth"))
                        for path in selected
                    )
                )

        self.assertIn(
            "text_encoder/model.safetensors.index.fp16.json",
            KANDINSKY3_FP16_DIFFUSERS_FILES,
        )
        self.assertIn(
            "unet/diffusion_pytorch_model.fp16.safetensors",
            KANDINSKY3_FP16_DIFFUSERS_FILES,
        )
        kandinsky_img2img = IMAGE_PIPELINE_ADAPTERS[
            "Kandinsky3Img2ImgPipeline"
        ]
        self.assertTrue(kandinsky_img2img.safe_serialization_required)
        self.assertEqual(kandinsky_img2img.weight_variant, "fp16")
        self.assertFalse(
            any(path.startswith("assets/") for path in KANDINSKY3_FP16_DIFFUSERS_FILES)
        )
        self.assertIn(
            "text_encoder_2/model.safetensors.index.json",
            HUNYUAN_DIT_DISTILLED_DIFFUSERS_FILES,
        )
        self.assertIn(
            "transformer/diffusion_pytorch_model.safetensors",
            HUNYUAN_DIT_DISTILLED_DIFFUSERS_FILES,
        )


if __name__ == "__main__":
    unittest.main()
