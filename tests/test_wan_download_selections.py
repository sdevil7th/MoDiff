import unittest

from modiff.server import STUDIO_MODEL_CAPABILITIES
from modiff.studio_execution_specs import (
    WAN_22_I2V_A14B_DIFFUSERS_FILES,
    WAN_22_TI2V_5B_DIFFUSERS_FILES,
    WAN_T2V_1_3B_DIFFUSERS_FILES,
    WAN_VACE_1_3B_DIFFUSERS_FILES,
)


class WanDownloadSelectionTests(unittest.TestCase):
    def test_admitted_wan_routes_use_component_only_safe_selections(self):
        cases = {
            "WanVideoPipeline": (WAN_T2V_1_3B_DIFFUSERS_FILES, 21),
            "WanVACEPipeline": (WAN_VACE_1_3B_DIFFUSERS_FILES, 19),
            "WanImageToVideoPipeline": (WAN_22_I2V_A14B_DIFFUSERS_FILES, 43),
            "WanTI2VPipeline": (WAN_22_TI2V_5B_DIFFUSERS_FILES, 22),
        }
        for model_type, (files, count) in cases.items():
            with self.subTest(model_type=model_type):
                selected = set(files)
                capability = STUDIO_MODEL_CAPABILITIES[model_type]
                self.assertEqual(capability["downloadFiles"], files)
                self.assertEqual(len(selected), count)
                self.assertFalse(
                    any(path.startswith(("assets/", "examples/")) for path in selected)
                )
                self.assertFalse(
                    any(
                        path.endswith((".bin", ".ckpt", ".pt", ".pth"))
                        for path in selected
                    )
                )
                self.assertIn("text_encoder/model.safetensors.index.json", selected)
                self.assertIn(
                    "transformer/diffusion_pytorch_model.safetensors.index.json",
                    selected,
                )
                self.assertIn("vae/diffusion_pytorch_model.safetensors", selected)

        i2v = set(WAN_22_I2V_A14B_DIFFUSERS_FILES)
        self.assertIn(
            "transformer_2/diffusion_pytorch_model.safetensors.index.json",
            i2v,
        )


if __name__ == "__main__":
    unittest.main()
