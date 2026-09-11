import json
from pathlib import Path
import unittest

from modiff.studio_execution_specs import (
    LONGCAT_IMAGE_DIFFUSERS_FILES,
    LONGCAT_IMAGE_EDIT_DIFFUSERS_FILES,
    studio_capability_definitions,
)


ROOT = Path(__file__).resolve().parents[1]


class LongCatDownloadSelectionTests(unittest.TestCase):
    def test_longcat_selections_cover_package_components_without_demo_assets(self):
        review = json.loads((ROOT / "data" / "longcat-image-artifact-review.json").read_text())
        capabilities = studio_capability_definitions()
        cases = {
            "LongCatImagePipeline": (
                LONGCAT_IMAGE_DIFFUSERS_FILES,
                31,
                29_316_541_013,
                review["repositories"]["textToImage"],
            ),
            "LongCatImageEditPipeline": (
                LONGCAT_IMAGE_EDIT_DIFFUSERS_FILES,
                32,
                29_316_540_813,
                review["repositories"]["imageEdit"],
            ),
        }

        for model_type, (files, count, byte_size, receipt) in cases.items():
            with self.subTest(model_type=model_type):
                selected = set(files)
                self.assertEqual(capabilities[model_type]["downloadFiles"], files)
                self.assertEqual(len(selected), count)
                self.assertEqual(receipt["selectedDownloadFileCount"], count)
                self.assertEqual(receipt["selectedDownloadBytes"], byte_size)
                self.assertIn("text_processor/preprocessor_config.json", selected)
                self.assertIn("tokenizer/tokenizer.json", selected)
                self.assertNotIn("config.json", selected)
                self.assertFalse(any(path.startswith("assets/") for path in selected))
                self.assertFalse(any(path.endswith((".bin", ".ckpt", ".pt", ".pth")) for path in selected))

        self.assertNotIn("text_encoder/preprocessor_config.json", LONGCAT_IMAGE_DIFFUSERS_FILES)
        self.assertIn("text_encoder/preprocessor_config.json", LONGCAT_IMAGE_EDIT_DIFFUSERS_FILES)


if __name__ == "__main__":
    unittest.main()
