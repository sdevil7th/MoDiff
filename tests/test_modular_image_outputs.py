import sys
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.Image.main import Preview  # noqa: E402
from modules.ModularDiffusers.latents import (  # noqa: E402
    ImageEncode,
    flatten_pil_images,
    prepare_image_for_vae_pipeline,
)


class ModularImageOutputTests(unittest.TestCase):
    def test_qwen_layered_vae_input_adds_an_opaque_alpha_channel(self):
        class QwenImageLayeredModularPipeline:
            pass

        source = Image.new("RGB", (8, 8), "red")
        prepared = prepare_image_for_vae_pipeline(source, QwenImageLayeredModularPipeline)

        self.assertEqual(prepared.mode, "RGBA")
        self.assertEqual(prepared.getpixel((0, 0)), (255, 0, 0, 255))
        self.assertEqual(source.mode, "RGB")

    def test_qwen_layered_vae_input_normalizes_nested_pil_batches(self):
        class QwenImageLayeredModularPipeline:
            pass

        source = [Image.new("RGB", (8, 8), "red"), (Image.new("RGBA", (8, 8), "blue"),)]
        prepared = prepare_image_for_vae_pipeline(source, QwenImageLayeredModularPipeline)

        self.assertEqual(prepared[0].mode, "RGBA")
        self.assertEqual(prepared[1][0].mode, "RGBA")

    def test_other_vae_pipelines_keep_their_source_unchanged(self):
        class QwenImageEditModularPipeline:
            pass

        source = Image.new("RGB", (8, 8), "red")

        self.assertIs(prepare_image_for_vae_pipeline(source, QwenImageEditModularPipeline), source)

    def test_image_encode_applies_layered_rgba_contract_before_running_pipeline(self):
        class QwenImageLayeredModularPipeline:
            pass

        observed = {}

        class FakePipeline:
            def __call__(self, **kwargs):
                observed.update(kwargs)
                return {"latents": "encoded"}

        class FakeBlocks:
            component_names = []
            input_names = ["image"]

            @staticmethod
            def init_pipeline(repo_id, components_manager):
                self.assertEqual(repo_id, "fixture/layered")
                return FakePipeline()

        node_config = {
            "params": {},
            "model_input_names": [],
            "input_names": ["image"],
            "output_names": ["latents"],
        }
        source = Image.new("RGB", (8, 8), "red")
        node = ImageEncode()
        node._pipeline_class = QwenImageLayeredModularPipeline

        with patch(
            "modules.ModularDiffusers.latents.pipeline_class_to_modiff_node_config",
            return_value=(FakeBlocks(), node_config),
        ):
            result = node.execute(vae={"repo_id": "fixture/layered"}, image=source)

        self.assertEqual(result["latents"], "encoded")
        summary = json.loads(result["encode_summary_data"])
        self.assertEqual(summary["schemaVersion"], 1)
        self.assertEqual(summary["status"], "encoded")
        self.assertGreaterEqual(summary["elapsedSeconds"], 0)
        self.assertEqual(observed["image"].mode, "RGBA")
        self.assertEqual(observed["image"].getpixel((0, 0)), (255, 0, 0, 255))
        self.assertEqual(source.mode, "RGB")

    def test_flattens_nested_layered_diffusers_pil_batches(self):
        first = Image.new("RGB", (8, 8), "red")
        second = Image.new("RGBA", (8, 8), "blue")

        self.assertEqual(flatten_pil_images([[first], (second,)]), [first, second])

    def test_does_not_relabel_non_image_values_as_decoded_images(self):
        self.assertIsNone(flatten_pil_images([[Image.new("RGB", (8, 8))], object()]))

    def test_preview_missing_vae_respects_declared_output_contract(self):
        result = Preview.execute(object(), image=object(), export="", vae=None, device="cpu")

        self.assertEqual(result, {"output": None, "filtered": None})


if __name__ == "__main__":
    unittest.main()
