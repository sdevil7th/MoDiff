import unittest

from modiff.server import STUDIO_MODEL_CAPABILITIES
from modiff.studio_execution_specs import (
    QWEN_IMAGE_2512_DIFFUSERS_FILES,
    QWEN_IMAGE_EDIT_2511_DIFFUSERS_FILES,
    QWEN_IMAGE_EDIT_DIFFUSERS_FILES,
    QWEN_IMAGE_LAYERED_DIFFUSERS_FILES,
)
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS


class QwenDownloadSelectionTests(unittest.TestCase):
    def test_qwen_image_selections_are_complete_and_safe(self):
        cases = {
            "QwenImageModularPipeline": (QWEN_IMAGE_2512_DIFFUSERS_FILES, 30),
            "QwenImageEditModularPipeline": (QWEN_IMAGE_EDIT_DIFFUSERS_FILES, 39),
            "QwenImageEditPlusModularPipeline": (
                QWEN_IMAGE_EDIT_2511_DIFFUSERS_FILES,
                35,
            ),
            "QwenImageLayeredModularPipeline": (
                QWEN_IMAGE_LAYERED_DIFFUSERS_FILES,
                33,
            ),
        }
        for model_type, (files, count) in cases.items():
            with self.subTest(model_type=model_type):
                selected = set(files)
                self.assertEqual(STUDIO_MODEL_CAPABILITIES[model_type]["downloadFiles"], files)
                self.assertEqual(len(selected), count)
                self.assertIn("model_index.json", selected)
                self.assertIn("text_encoder/model.safetensors.index.json", selected)
                self.assertIn(
                    "transformer/diffusion_pytorch_model.safetensors.index.json",
                    selected,
                )
                self.assertIn("vae/diffusion_pytorch_model.safetensors", selected)
                self.assertFalse(
                    any(
                        path.endswith((".bin", ".ckpt", ".pt", ".pth"))
                        for path in selected
                    )
                )

        self.assertNotIn("processor/tokenizer.json", QWEN_IMAGE_2512_DIFFUSERS_FILES)
        self.assertIn("processor/tokenizer.json", QWEN_IMAGE_EDIT_DIFFUSERS_FILES)
        self.assertIn("processor/tokenizer.json", QWEN_IMAGE_EDIT_2511_DIFFUSERS_FILES)
        self.assertIn("processor/tokenizer.json", QWEN_IMAGE_LAYERED_DIFFUSERS_FILES)

    def test_standard_qwen_adapters_require_safe_serialization(self):
        for pipeline_class in (
            "QwenImagePipeline",
            "QwenImageImg2ImgPipeline",
            "QwenImageInpaintPipeline",
            "QwenImageEditPipeline",
            "QwenImageEditInpaintPipeline",
            "QwenImageEditPlusPipeline",
        ):
            with self.subTest(pipeline_class=pipeline_class):
                self.assertTrue(
                    IMAGE_PIPELINE_ADAPTERS[pipeline_class].safe_serialization_required
                )


if __name__ == "__main__":
    unittest.main()
